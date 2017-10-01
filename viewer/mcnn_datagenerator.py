import numpy as np
import pylidc as pl
import dicom
from sqlalchemy import or_
from skimage import draw
import math, os, sys
#import pylab

# Path to DICOM images
DICOM_PATH = ''

# Standard CT scan size. This will
# be needed to build the mask.
STANDARD_SHAPE = (512, 512)

EXTRACT_SIZE = 25

def create_mask(coordinates, shape=STANDARD_SHAPE):
    '''
    `coordinates` as represented in the Pylidc database for
    outlining the boundary of individual nodules.
    '''
    coords = coordinates.split('\n')
    coords_split = [c.split(',') for c in coords]
    rows = [int(c[0]) for c in coords_split]
    cols = [int(c[1]) for c in coords_split]

    rows, cols = draw.polygon(rows, cols, shape)
    
    mask = np.zeros(shape, dtype=np.bool)
    mask[rows, cols] = True
    
    return mask

def get_dicom_file_name_adjustments(file_name):
    file_name = file_name.rjust(10, '0')
    return file_name

def extract_image(image, mid_x, mid_y):
    '''
    image: 2-D array representing the slice
    mid_x, mid_y: Mid point of the image
    '''
    extracted_image = image[int(math.floor(mid_y)) - EXTRACT_SIZE:int(math.floor(mid_y)) + EXTRACT_SIZE + 1, int(math.floor(mid_x)) - EXTRACT_SIZE:int(math.floor(mid_x)) + EXTRACT_SIZE + 1]
    return extracted_image

def get_middle_contours(contours, base_path, return_all=False):
    '''
    This function returns the middle 
    'n' contours from the set of available
    contours in the DB.
    '''
    if return_all:
        return contours
    
    # print ('Contours Available = ', len(contours))
    
    incl_contours = []
    for c in contours:
        file_name = c.dicom_file_name
        file_name = get_dicom_file_name_adjustments(file_name)
        file_path = base_path + '/' + file_name
        
        # Ensure that the contour is an inclusion contour
        # and that the corresponding DICOM file path actually 
        # exists in the File System DICOM downloads
        if c.inclusion and os.path.exists(file_path):
            incl_contours.append(c)
    
    contours = incl_contours
    length = len(contours)
    if length < 3:
        return []
    
    if length % 2 == 0:
        # We delete the last element to make the list an 
        # Odd elements list
        del contours[-1]
        length = length - 1
        
    if length == 3:
        return contours

    mid_point = int(math.floor(length / 2))
    mid_contours = []
    
    for i in range(mid_point - 1, mid_point + 2):
        mid_contours.append(contours[i])
    
    return mid_contours

from sqlalchemy import and_

# annotations = pl.query(pl.Annotation).filter(and_(pl.Annotation.id >= 4640, pl.Annotation.id <= 4641))
# Fetch and process all the annotation data there is in the system
annotations = pl.query(pl.Annotation)
annotations_count = annotations.count()

qualified_ann_count = 0

max_xrange = 0
max_yrange = 0
min_xrange = 100000
min_yrange = 100000

training_data = []
target_data = []

for ann in annotations:
    ann_id = str(ann.id)
    ann_id = ann_id.rjust(8, ' ')
    sys.stdout.flush()
    # print ('Processing Annotation ID = ' + ann_id)
    scan = ann.scan
    contours = ann.contours

    if len(contours) > 2:
        '''
        These are the annotations that should figure in the final 
        generated data for training and testing of the model.
        '''
        qualified_ann_count += 1
        sorted_contours = sorted(contours, key=lambda c: c.image_z_position)
        base_path = scan.get_path_to_dicom_files(checkpath=False)
        mid_contours = get_middle_contours(sorted_contours, base_path, return_all=False)
        if not (os.path.exists(base_path)):
            continue
        
        if len(mid_contours) >= 3:
            slices = []
            # Once we have all the contours, we will recreate
            # masks and then apply them to original DICOM images
            for contour in mid_contours:
                ctr_coords = contour.to_matrix()
                xmax = np.amax(ctr_coords[:, 0])
                xmin = np.amin(ctr_coords[:, 0])
                ymax = np.amax(ctr_coords[:, 1])
                ymin = np.amin(ctr_coords[:, 1])
                
                xrange = (xmax - xmin)
                yrange = (ymax - ymin)
                
                if(xrange > max_xrange):
                    max_xrange = xrange
                    
                if(yrange > max_yrange):
                    max_yrange = yrange
                    
                if(xrange < min_xrange):
                    min_xrange = xrange
                
                if(yrange < min_yrange):
                    min_yrange = yrange
                
                # print ('    X Range = %d', (xmax - xmin))
                # print ('    Y Range = %d', (ymax - ymin))
                
                # (xcentroid, ycentroid) is the center of the
                # ROI from which the nodule will be extracted
                # into a square of 32x32 pixels
                xcentroid = (xmin + xmax) / 2
                ycentroid = (ymin + ymax) / 2
                
                mask = create_mask(contour.coords)
                file_name = contour.dicom_file_name
                
                # Adjust the filename to fit to the LIDC 
                # 10 character format
                file_name = file_name.rjust(10, '0')
                
                # Now building the file path to load the 
                # DCM image and do the needful processing
                file_path = base_path + '/' + file_name
                ds = dicom.read_file(file_path)
                pixel_array = np.copy(ds.pixel_array)

                rows, cols = np.where(mask == False)
                mask = np.dstack((rows, cols))
                sh = np.shape(mask)
                mask = np.reshape(mask, (sh[1], sh[2]))
                for ind in mask:
                    pixel_array[ind[1]][ind[0]] = 0

                side_by_side = np.concatenate((pixel_array, ds.pixel_array), axis=1)
                
                #pylab.imshow(side_by_side, cmap=pylab.cm.bone)
                
                red_rows = [512 + x for x in ctr_coords[:, 0]]
                # red_rows = ctr_coords[:,0]
                red_cols = ctr_coords[:, 1]
                #pylab.plot(red_rows, red_cols, 'r')
                
                #pylab.show()
                
                extracted_image = extract_image(pixel_array, xcentroid, ycentroid)
                x_dim = extracted_image.shape[0]
                y_dim = extracted_image.shape[1]

                x_diff = 0
                y_diff = 0

                if x_dim < 51:
                    x_diff = 51 - x_dim
                
                if y_dim < 51:
                    y_diff = 51 - y_dim
                
                shape_padding = ((0, x_diff), (0, y_diff))
                # print ('Shape Padding = ' + str(shape_padding))
                extracted_image_2 = np.pad(extracted_image, shape_padding, mode='constant', constant_values = 0)

                if extracted_image_2.shape == extracted_image.shape:
                    equality = (extracted_image_2 == extracted_image)
                    indices = np.where(equality == False)
                    print ('Annotation ID = ' + str(ann.id) + '  .. Unequal = ' + str(indices))

                #pylab.imshow(extracted_image, cmap=pylab.cm.bone)
                #pylab.show()
                slices.append(extracted_image_2)
                
            print ('Adding Training Data for Annotation ID = ' + str(ann.id))
            if ann.malignancy == 1 or ann.malignancy == 2:
                target_data.append(np.asarray([1,0,0]))
                training_data.append(slices)
            elif ann.malignancy == 3:
                target_data.append(np.asarray([0,1,0]))
                training_data.append(slices)
            elif ann.malignancy == 4 or ann.malignancy == 5:
                target_data.append(np.asarray([0,0,1]))
                training_data.append(slices)
            else:
                print ('Unrecognized Output data')
        else:
            print ('Skipping Annotation ', ann.id, ' as not enough contours found ..')

print ('Total Annotations = ', annotations_count)
print ('Qualified Annotations = ', qualified_ann_count)

training_data_array = np.asarray(training_data)
target_data_array = np.asarray(target_data)

print ('Full Data Size = ' + str(training_data_array.shape))
print ('Full Target Data Size = ' + str(target_data_array.shape))

np.savez('../data/2809/unscaled_training_data_range.npz', data = training_data_array)
np.savez('../data/2809/target_data_range.npz', data = target_data_array)

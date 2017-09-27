import numpy as np
import pylidc as pl
import dicom
from sqlalchemy import or_
from skimage import draw
import math, os, sys
import pylab

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
    print (image[math.floor(mid_x) - EXTRACT_SIZE:math.floor(mid_x) + EXTRACT_SIZE + 1, math.floor(mid_y) - EXTRACT_SIZE:math.floor(mid_y) + EXTRACT_SIZE + 1])
    extracted_image = image[math.floor(mid_y) - EXTRACT_SIZE:math.floor(mid_y) + EXTRACT_SIZE + 1, math.floor(mid_x) - EXTRACT_SIZE:math.floor(mid_x) + EXTRACT_SIZE + 1]
    return extracted_image

def get_middle_contours(contours, base_path, return_all=False):
    '''
    This function returns the middle 
    'n' contours from the set of available
    contours in the DB.
    '''
    if return_all:
        return contours
    
    print ('Contours Available = ', len(contours))
    
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

annotations = pl.query(pl.Annotation).filter(pl.Annotation.id == 39)
annotations_count = annotations.count()

qualified_ann_count = 0

max_xrange = 0
max_yrange = 0
min_xrange = 100000
min_yrange = 100000

for ann in annotations:
    ann_id = str(ann.id)
    ann_id = ann_id.rjust(8, ' ')
    sys.stdout.flush()
    print ('Processing Annotation ID = ' + ann_id)
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
        mid_contours = get_middle_contours(sorted_contours, base_path, return_all=True)
        if not (os.path.exists(base_path)):
            continue
        
        if len(mid_contours) >= 3:
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
                
                pylab.imshow(side_by_side, cmap=pylab.cm.bone)
                
                print ('File = ', file_path)
                print ('Contour Id = ', contour.id)
                print ('Annotation = ', ann.id)
                print ('Z Index = ', contour.image_z_position)
                
                red_rows = [512 + x for x in ctr_coords[:, 0]]
                # red_rows = ctr_coords[:,0]
                red_cols = ctr_coords[:, 1]
                pylab.plot(red_rows, red_cols, 'r')
                
                pylab.show()
                
                extracted_image = extract_image(pixel_array, xcentroid, ycentroid)
                pylab.imshow(extracted_image, cmap=pylab.cm.bone)
                pylab.show()
        else:
            print ('Skipping Annotation ', ann.id, ' as not enough contours found ..')

print ('Maximum X Range = %d, Y Range = %d', max_xrange, max_yrange)
print ('Minimum X Range = %d, Y Range = %d', min_xrange, min_yrange)


print ('Total Annotations = ', annotations_count)
print ('Qualified Annotations = ', qualified_ann_count)

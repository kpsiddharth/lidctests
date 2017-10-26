import pylidc as pl
import pylab
import math
from skimage import draw
import numpy as np
import os
import dicom
from skimage.transform import rescale, resize
from numpy import float32

load_annotations = True

scans = pl.query(pl.Scan)

STANDARD_SHAPE = (512, 512)
EXTRACT_SIZE = 14

to_be_added = []

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
    
    return mask, rows, cols

def get_dicom_file_name_adjustments(file_name):
    file_name = file_name.rjust(10, '0')
    return file_name

def extract_image(image, mid_x, mid_y, min_x, min_y, max_x, max_y):
    '''
    image: 2-D array representing the slice
    mid_x, mid_y: Mid point of the image
    '''
    extracted_image = image[int(min_y):int(max_y + 1), int(min_x):int(max_x + 1)]
    xrange = max_x - min_x + 1
    yrange = max_y - min_y + 1
    
    bigger, smaller, axis = (xrange, yrange, 1) if xrange > yrange else (yrange, xrange, 0)
    min_size = 28
    if bigger > min_size:
        if axis == 0:
            new_xsize = int((float(min_size)/float(bigger)) * float(smaller))
            new_ysize = min_size
        else:
            new_xsize = min_size
            new_ysize = int((float(min_size)/float(bigger)) * float(smaller))
            
        print ('Newly determined size = ' + str(new_ysize) + ',' + str(new_xsize))
    
        extracted_image2 = resize(extracted_image, (new_ysize, new_xsize))
        
        extracted_image = extracted_image2
    
    return extracted_image

def construct_patch_from_contour(ann, contour):
    scan = ann.scan
    
    base_path = scan.get_path_to_dicom_files(checkpath=False)
    
    ctr_coords = contour.to_matrix()
    xmax = np.amax(ctr_coords[:, 0])
    xmin = np.amin(ctr_coords[:, 0])
    ymax = np.amax(ctr_coords[:, 1])
    ymin = np.amin(ctr_coords[:, 1])
    
    xrange = (xmax - xmin)
    yrange = (ymax - ymin)
    
    xcentroid = (xmin + xmax) / 2
    ycentroid = (ymin + ymax) / 2
    
    mask, r, c = create_mask(contour.coords)

    file_name = contour.dicom_file_name
    
    # Adjust the filename to fit to the LIDC 
    # 10 character format
    file_name = file_name.rjust(10, '0')
    
    # Now building the file path to load the 
    # DCM image and do the needful processing
    file_path = base_path + '/' + file_name
    if not os.path.exists(file_path):
        print ('File not found: ' + file_path)
        return None

    ds = dicom.read_file(file_path)
    pixel_array = np.copy(ds.pixel_array)

    rows, cols = np.where(mask == False)
    mask = np.dstack((rows, cols))
    sh = np.shape(mask)
    mask = np.reshape(mask, (sh[1], sh[2]))
    for ind in mask:
        pixel_array[ind[1]][ind[0]] = 0

    red_rows = [512 + x for x in ctr_coords[:, 0]]
    red_cols = ctr_coords[:, 1]
    pylab.plot(red_rows, red_cols, 'r')

    #side_by_side = np.concatenate((pixel_array, ds.pixel_array), axis=1)
    #pylab.imshow(side_by_side, cmap=pylab.cm.bone)
    #pylab.show()

    extracted_image = extract_image(pixel_array, xcentroid, ycentroid, xmin, ymin, xmax, ymax)
    return extracted_image

def get_malignancy(level):
    if level == 1 or level == 2:
        return 0
    elif level == 3:
        return 2
    else:
        return 1


if not load_annotations:
    for scan in scans:
        a = scan.annotations
        c = scan.cluster_annotations()
        if len(c) == 0:
            # Error case -- when are there no clusters
            continue
    
        for cluster in c:
            if len(cluster) >= 1:
                malignancy = get_malignancy(cluster[0].malignancy)
                mismatch = False
                for a in cluster:
                    m = get_malignancy(a.malignancy)
                    if malignancy == 2:
                        malignancy = m
                        
                    if malignancy == m:
                        continue
                    else:
                        mismatch = True
            
                if malignancy == 2:
                    continue
            
                if not mismatch:
                    print ('Accepting Annotation .. ' + str(cluster[0].id))
                    to_be_added.append(cluster[0])
                else:
                    print ('Discarding Annotations = ' + str(cluster[0]))
                    pass
    
    print ('Total Annotations = ' + str(len(to_be_added)))
    np.savez('annotations_filtered_all.npz', data=to_be_added)

annotations = np.load('annotations_filtered_all.npz')['data']
patches = []

for ann in annotations:
    contours = pl.query(pl.Contour).filter(pl.Contour.annotation_id == ann.id)
    for c in contours:
        print (type(c))
        try:
            print ('Displaying Contour ID = ' + str(c.id))
            patch = construct_patch_from_contour(ann, c)
            patches.append(patch)
            pylab.imshow(patch, cmap=pylab.cm.bone)
            pylab.show()
        except Exception as e:
            pass
print ('Total Data Set Size = ' + str(len(patches)))


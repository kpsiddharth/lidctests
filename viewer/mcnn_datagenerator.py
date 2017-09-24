import numpy as np
import pylidc as pl
from sqlalchemy import or_
from skimage import draw

STANDARD_SHAPE = (32, 32)

def create_mask(coordinates, shape = STANDARD_SHAPE):
    '''
    `coordinates` as represented in the Pylidc database for
    outlining the boundary of individual nodules.
    '''
    coords = coordinates.split('\n')
    coords_split = [c.split(',') for c in coords]
    rows = [int(c[0]) for c in coords_split]
    cols = [int(c[1]) for c in coords_split]

    min_x = np.amin(rows)
    min_y = np.amin(cols)
    
    rows = [r - min_x for r in rows]
    cols = [c - min_y for c in cols]
    
    rows, cols = draw.polygon(rows, cols, shape)
    
    mask = np.zeros(shape, dtype = np.bool)
    mask[rows, cols] = True
    
    return mask

def get_middle_contours(contours):
    '''
    This function returns the middle 
    'n' contours from the set of available
    contours in the DB.
    '''
    incl_contours = []
    for c in contours:
        if c.inclusion:
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

    mid_point = length / 2
    mid_contours = []
    
    for i in range(mid_point - 1, mid_point + 2):
        mid_contours.append(contours[i])
    
    return mid_contours

annotations = pl.query(pl.Annotation)
annotations_count = annotations.count()

qualified_ann_count = 0

for ann in annotations:
    scan = ann.scan
    contours = ann.contours
    if len(contours) > 2:
        '''
        These are the annotations that should figure in the final 
        generated data for training and testing of the model.
        '''
        qualified_ann_count += 1
        # print ('Contour [0] = ', contours[0].id)
        sorted_contours = sorted(contours, key=lambda c: c.image_z_position)
        mid_contours = get_middle_contours(sorted_contours)
        
        if len(mid_contours) == 3:
            for contour in mid_contours:
                ctr_coords = contour.to_matrix()
                # print ('File Name = ', contour.dicom_file_name, ' \nCoords = ', ctr_coords)
        else:
            print ('Skipping Annotation ', ann.id, ' as not enough contours found ..')

print ('Total Annotations = ', annotations_count)
print ('Qualified Annotations = ', qualified_ann_count)

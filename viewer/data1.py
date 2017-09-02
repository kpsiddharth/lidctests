import numpy as np
import pylidc as pl
from sqlalchemy import or_

ubbox_radius = 25
uside_length = 25

#Collecting all annotation object.
annotations = pl.query(pl.Annotation)

#Taking only 3 annotations-If need to check data
#annotations = annotations.filter(or_(pl.Annotation.scan_id == 17, pl.Annotation.scan_id == 27,pl.Annotation.scan_id == 10))

#Counting Total annotation/Nodule
annotations_count = annotations.count()
#annotations_count = 10

#Creating numpy array for nodule data.
#It will be used as training input data in CNN.
resampled_nodule_data = np.zeros((annotations_count,uside_length,uside_length,uside_length))

#Creating numpy array for maligancy of each nodule.
#It will be used as target data in CNN.
target_data = np.zeros((annotations_count, 2))

counter = 0

bbox_size = np.zeros((annotations_count, 1))

#Updating nodule and target data with actual value 
#by iterating on each annotation/nodule.
for count, annotation in enumerate(annotations):
    try:
        bboxd = annotation.bbox_dimensions(image_coords = True)
        bbox_size[count][0] = bboxd.max()
        if bboxd.max() >= uside_length:
            print 'Annotation ID = ', annotation.id, ' : Skipping .. Malignancy = ', annotation.malignancy
            continue
        print 'Annotation ID = ', annotation.id, ' : Processing .. Malignancy = ', annotation.malignancy
        ann_vol, ann_seg = annotation.uniform_cubic_resample(side_length = (uside_length-1))
        resampled_nodule_data[count] = ann_vol
        rows, cols, heights = np.where(ann_seg == False)
        mask = np.dstack((rows, cols, heights))
        sh = np.shape(mask)
        mask = np.reshape(mask, (sh[1], sh[2]))
        #print 'Shape = ', np.shape(mask)
        i = 0
        for ind in mask:
            i = i + 1
        #    print 'Setting (', ind[0], ',', ind[1], ',', ind[2], ') to 0 ..'
        #    ann_vol[ind[0]][ind[1]][ind[2]] = 0
        print 'i = ', i
        print '\n\nAnn Vol = ', np.shape(ann_vol)
        target_data[count][0] = annotation.malignancy
        target_data[count][1] = annotation.estimate_diameter()
    except Exception as e:
        print (e.message, annotation.scan_id)

    counter = counter+1
    print ('Counter = ', counter)
    if counter > annotations_count:
        break

resampled_nodule_data_in_2D = np.delete(resampled_nodule_data_in_2D, np.s_[i:annotation_count])
print 'Resized the Resampled Array to = ', np.shape(resampled_nodule_data_in_2D)

#Converting 4D nodule data into 2D.
#As we can't save nD(n>2) array as it is into file.
#For this either we will have to reshape this array into 1D or 2D array.
#or will have to save in binary form.
resampled_nodule_data_in_2D = np.reshape(resampled_nodule_data,(np.shape(resampled_nodule_data_in_2D)[0],uside_length*uside_length*uside_length))
print 'Resized Resample Array to 2D = ', np.shape(resampled_nodule_data_in_2D)

#Converting array data type float64 to int8.

#storing nodule data and target data in CSV format.
np.savez('training_data.npz', data = resampled_nodule_data_in_2D)
np.savez('training_targets.npz', data = target_data)
np.savez('bbox_sizes.npz', data = bbox_size)

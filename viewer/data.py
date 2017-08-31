import numpy as np
import pylidc as pl
from sqlalchemy import or_

#Collecting all annotation object.
annotations = pl.query(pl.Annotation)

#Taking only 3 annotations-If need to check data
#annotations = annotations.filter(or_(pl.Annotation.scan_id == 17, pl.Annotation.scan_id == 27,pl.Annotation.scan_id == 10))

#Counting Total annotation/Nodule
annotations_count = annotations.count()
# annotations_count = 1500

#Creating numpy array for nodule data.
#It will be used as training input data in CNN.
resampled_nodule_data = np.zeros((annotations_count,40,40,40))

#Creating numpy array for maligancy of each nodule.
#It will be used as target data in CNN.
target_data = np.zeros(annotations_count)

counter = 0

#Updating nodule and target data with actual value 
#by iterating on each annotation/nodule.
for count, annotation in enumerate(annotations):
    try:
        ann_vol, ann_seg = annotation.uniform_cubic_resample(side_length = 39)
        resampled_nodule_data[count] = ann_vol
        target_data[count] = annotation.malignancy
        print 'Annotation [', count, '] Malignancy = ', target_data[count]
    except Exception as e:
        print  e.message, annotation.scan_id

    counter = counter+1
    print 'Counter = ', counter
    if counter > annotations_count:
        break

#Converting 4D nodule data into 2D.
#As we can't save nD(n>2) array as it is into file.
#For this either we will have to reshape this array into 1D or 2D array.
#or will have to save in binary form.
resampled_nodule_data_in_2D = np.reshape(resampled_nodule_data,(annotations_count,40*40*40))

#Converting array data type float64 to int8.
resampled_nodule_data_in_2D_int8 = resampled_nodule_data_in_2D.astype(np.int8)

#storing nodule data and target data in CSV format.
#np.savetxt("final_data_1.csv", resampled_nodule_data_in_2D_int8, delimiter=",")
#np.savetxt("target_data_1.csv", target_data, delimiter=",")

np.savez('training_data.npz', data = resampled_nodule_data_in_2D)
np.savez('training_targets.npz', data = target_data)

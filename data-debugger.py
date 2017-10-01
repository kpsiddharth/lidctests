import pylidc as pl
import dicom
import pylab
from numpy import shape
import os
import numpy

query = pl.query(pl.Contour)
print ('Total Contours = ' + str(query.count()))

def strip_leading_zeros(file_name):
    length = len(file_name)
    split_index = 0
    i = 0
    for i in range(length):
        if file_name[i] == '0':
            continue
        else:
            break
    return file_name[i:]

qann = pl.query(pl.Annotation)
for ann in qann:
    scan = ann.scan
    contours = ann.contours
    base_path = scan.get_path_to_dicom_files(checkpath=False)
    
    z_to_file_mapping = {}
    for filename in os.listdir(base_path):
        if filename.endswith('.dcm'):
            ds = dicom.read_file(base_path + '/' + filename)
            z = ds.SliceLocation
            z_to_file_mapping[z] = filename
            #print ('Mapped ' + str(z) + ' -> ' + filename)

    for contour in contours:
        file_name = contour.dicom_file_name
        file_name = file_name.rjust(10, '0')
        file_path = base_path + '/' + file_name
        if os.path.exists(file_path):
            ds = dicom.read_file(file_path)
            actual_z = ds.SliceLocation
            db_z = contour.image_z_position
            if not (db_z == actual_z):
                if db_z in z_to_file_mapping:
                    #print ('Actual File Name = ' + file_name + ' - File Name should be = ' + z_to_file_mapping[db_z])
                    should_be_file = z_to_file_mapping[db_z]
                    should_be_file = strip_leading_zeros(should_be_file)
                    print ('UPDATE CONTOURS SET DICOM_FILE_NAME = \'' + should_be_file + '\' WHERE ID = ' + str(contour.id) + ';')
            
            
'''
for contour in query:
    ann = contour.annotation
    scan = ann.scan

    file_name = contour.dicom_file_name
    file_name = file_name.rjust(10, '0')

    base_path = scan.get_path_to_dicom_files(checkpath=False)
    file_path = base_path + '/' + file_name
    
    if os.path.exists(file_path):
        ds = dicom.read_file(file_path)
        z = ds.SliceLocation
        
        if not (z == contour.image_z_position):
            print ('... Z-Axis Mismatch: ' + file_path + ' and Contour ID = ' + str(contour.id) + ' Z = ' + str(z) + ' Contour Z = ' + str(contour.image_z_position))
'''

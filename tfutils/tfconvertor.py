from datasource.datasource import get_training_filenames
import tensorflow as tf
import numpy
import dicom

# Define the name of the file to which the input data is to be persisted
tfrecords_filename = 'D:/Data/ml/medical/LIDC/lidc_batches.tfrecords'

# Creating a TF Records writer object
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

# Code has been inspired by: http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
# Good, practical read on how to use tfrecords in TensorFlow


# Utility functions to enable creating of Feature records 
# in TFRecords files
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Utility functions to enable creating of Feature records 
# in TFRecords files
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Peruse the contents of the training files and then prepare
# for serialization
def convert_and_persist():
    filenames = get_training_filenames(False)
    for fullpath in filenames:
        ds = dicom.read_file(fullpath)
        image = ds.pixel_array
        height = numpy.shape(image)[0]
        width = numpy.shape(image)[1]
        
        raw_image_string = image.tostring()
        
        # Recreate the image using the original dtype
        # image2 = numpy.fromstring(raw_image_string, dtype=ds.pixel_array.dtype)
        # image2 = numpy.reshape(image2, (height, width))

        # Prepare row that is to be written
        row = tf.train.Example(features = tf.train.Features(feature = {
            'height' : _int64_feature(height),
            'width' : _int64_feature(width),
            'image' : _bytes_feature(raw_image_string)}))
        
        writer.write(row.SerializeToString())
        
    writer.close()
    
if __name__ == '__main__':
    convert_and_persist()
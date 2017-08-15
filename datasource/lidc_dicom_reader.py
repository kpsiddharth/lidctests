import dicom
import tensorflow as tf

def read_lidc(filename_queue):
    class DicomLIDCRecord(object):
        pass
    
    dicom_reader = tf.WholeFileReader()
    _, dicom_file = dicom_reader.read(filename_queue)
    
    # tf.image.decode_jpeg(contents, channels, ratio, fancy_upscaling, try_recover_truncated, acceptable_fraction, dct_method, name)
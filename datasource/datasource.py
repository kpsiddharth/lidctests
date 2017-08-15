import tensorflow as tf
import os
import dicom, pylab, numpy
from datasource.constants import STANDARD_WIDTH, STANDARD_HEIGHT
from xml.dom.minidom import parse, parseString
from datasource import constants

FLAGS = tf.app.flags.FLAGS

image_uid2ds_map = {}
sopId2fileNames_map = {}

#path = 'D:/Data/ml/medical/LIDC/DICOM/DOI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.175012972118199124641098335511/1.3.6.1.4.1.14519.5.2.1.6279.6001.141365756818074696859567662357'
path = 'D:/Data/ml/medical/LIDC/DICOM/DOI/LIDC-IDRI-0003/1.3.6.1.4.1.14519.5.2.1.6279.6001.101370605276577556143013894866/1.3.6.1.4.1.14519.5.2.1.6279.6001.170706757615202213033480003264'
#path = 'D:/Data/ml/medical/LIDC/DICOM/DOI'

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', path,
                           """Path to the LIDC data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_string('path_separator', '/',
                            """File system path separator.""")

def parse_lidc_truth_xmls(path):
    faulty_paths = []
    
    found_count = 0
    missing_count = 0
    
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith('xml'):
                dom = parse(root + FLAGS.path_separator + name)
                imageSOP_UID_nodes = dom.getElementsByTagName(constants.ImageSOP_UID)
                imageUids = []
                for node in imageSOP_UID_nodes:
                    sopUid = node.childNodes[0].nodeValue
                    imageUids.append(sopUid)

                    if not image_uid2ds_map.get(sopUid):
                        missing_count += 1
                    else:
                        #print ('Found image UID = ', image_uid2ds_map[sopUid].SOPInstanceUID)
                        found_count += 1
                
                #if not(len(set(imageUids)) == 1):
                    # All well here - not more than a single image UID referred here
                #    print ('In-Correct Path = ', root + '/' + name)
                #    faulty_paths.append(root + '/' + name)
                #else:
                #    print ('Correct Path = ', root + '/' + name)
    
    # Print the list of all faulty paths
    print ('Number of Images Found = ', found_count, '. Number of Images Missing = ', missing_count, '.')

def get_training_filenames(plot = False):
    filenames = []
    filtered_filenames = []
    
    sop_instance_id = []
    
    path = FLAGS.data_dir
    large_count = 0
    small_count = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith('dcm'):
                fullpath = root + FLAGS.path_separator + name
                ds = dicom.read_file(fullpath)
               
                sop_instance_id.append(ds.SOPInstanceUID)
                sopId2fileNames_map[ds.SOPInstanceUID] = name
                
                # cache a UID to image mapping for future lookups
                # these UIDs are referenced by the annotated XMLs
                # and hence are relevant to when we are processing 
                # the truth data.
                image_uid2ds_map[ds.SOPInstanceUID] = ds
                
                # print ('Full Path - ', fullpath)
                # ignore all the x-rays
                if numpy.shape(ds.pixel_array)[0] > 2000:
                    large_count += 1
                    filtered_filenames.append(fullpath)
                else:
                    small_count += 1
                    pixel_array = numpy.shape(ds._pixel_data_numpy())
                    if pixel_array[0] == STANDARD_HEIGHT and pixel_array[1] == STANDARD_WIDTH:
                        if plot:
                            pylab.imshow(ds.pixel_array, cmap = pylab.cm.bone)
                            pylab.show()
                            # print ('Type = ', type(ds.pixel_array))
                        filenames.append(fullpath)
                    else:
                        filtered_filenames.append(fullpath)
                        pass

    print ('Total files = ', len(filenames), '. Filtered Files = ', 
           len(filtered_filenames), '. Validating file availability.')
    check_file_availability(filenames)
    
    print ('Duplicates in SOP Instance Ids = ', not(len(sop_instance_id) == len(set(sop_instance_id))))
    
    return filenames

def prepare_dataset_for_training(filenames):
    filename_queue = tf.train.string_input_producer(filenames)

def check_file_availability(filenames):
    for f in filenames:
        if not tf.gfile.Exists(f):
          raise ValueError('Failed to find file: ' + f)

    print ('Successfully validated existence of all files ..')
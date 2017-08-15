import datasource.datasource as ds
from datasource.datasource import FLAGS

if __name__ == '__main__':
    print ('Attempting to load data set ..')
    ds.get_training_filenames(plot = False)
    ds.parse_lidc_truth_xmls(FLAGS.data_dir)
    pass
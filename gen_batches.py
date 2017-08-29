import csv
import numpy as np

def generate_batches(file_name, out_dir='gen_data/', out_file='databatch_', batch_size=500):
    f = open(file_name, 'rb')
    reader = csv.reader(f, delimiter=',')

    current_batch = 1
    batch_as_list = []

    for i, line in enumerate(reader):
        row = np.array(line, dtype=type(np.inf))
        batch_as_list.append(row)
        if (i+1) % batch_size == 0:
            batch_array = np.array(batch_as_list)
            print 'Saving Batch [', current_batch, '] with ', np.shape(batch_array)[0], ' rows'
            np.savez(out_dir + out_file + str(current_batch) + '.npz', data = batch_array)
            current_batch = current_batch + 1

            del batch_as_list
            del batch_array

            batch_as_list = []

    if len(batch_as_list) > 0:
        batch_array = np.array(batch_as_list)
        print 'Saving Batch [', str(current_batch), '] .. Last Batch .. with ', np.shape(batch_array)[0], ' rows'
        np.savez(out_dir + out_file + str(current_batch) + '.npz', data = batch_array)

    print 'Finished Batch Generation ...'

print 'Generating Training Data Batches .. '
generate_batches('datafull/target_data.csv', out_file='testbatch_')
generate_batches('datafull/final_data.csv')


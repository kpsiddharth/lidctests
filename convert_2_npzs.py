import numpy as np
import pandas as p

data_file_name = 'datafull/final_data.csv'
targets_file_name = 'datafull/target_data.csv'

data_npz_name = 'final_data.npz'
target_npz_name = 'final_targets.npz'

pandas_df = p.read_csv(data_file_name, header=None)
csv_data_array = pandas_df.as_matrix()
del pandas_df
#csv_data_array = np.loadtxt(data_file_name, delimiter = ',')
csv_targets_array = np.loadtxt(targets_file_name, delimiter = ',')
csv_targets_array = np.reshape(csv_targets_array, (np.shape(csv_targets_array)[0], 1))

print 'Data Shape = ', np.shape(csv_data_array)
print 'Targets Shape = ', np.shape(csv_targets_array)

csv_data_array = np.concatenate((csv_data_array, csv_targets_array), axis = 1)

del csv_targets_array

print ('Net shape of concatenated array = ', np.shape(csv_data_array))

np.random.shuffle(csv_data_array)

csv_targets_array = csv_data_array[:, -1]
np.savez(target_npz_name, data = csv_targets_array)

csv_data_array = csv_data_array[:, :np.shape(csv_data_array)[1] - 1]
print 'Size of data array being saved = ', np.shape(csv_data_array)
np.savez(data_npz_name, data = csv_data_array)

del csv_data_array




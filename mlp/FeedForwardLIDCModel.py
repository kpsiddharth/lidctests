'''
Created on Aug 23, 2017

@author: spurkayastha
'''

import numpy as np
import sys
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.externals import joblib
from datetime import datetime

train_data = 'datafull/final_data.csv'
result_data = 'datafull/target_data.csv'

training_data = []
target_data = []

data_size = 6589
num_classes = 6
num_iter = 1000
learning_rate = 0.001

def load_filtered_data_from_npz():
    print 'Loading Filtered data from NPZ ..'
    sys.stdout.flush()
    training_trgt_set = 'gen_data/full/reclassed_encoded_targets.npz'
    training_data_set = 'gen_data/full/scaled_data.npz'
    #test_data_set = 'gen_data/full/test_data.npz'
    #test_trgt_set = 'gen_data/full/test_targets_data.npz'

    training_data = np.load(training_data_set)
    training_data = training_data['data']
    trn_trgt_data = np.load(training_trgt_set)
    trn_trgt_data = trn_trgt_data['data']

    test_data     = training_data[6000:, :]
    test_trgt     = trn_trgt_data[6000:, :]

    training_data = training_data[:6000, :]
    trn_trgt_data = trn_trgt_data[:6000, :]

    # training_data, trn_trgt_data = preprocess_training_data(training_data, trn_trgt_data)
    # test_data, test_trgt = preprocess_training_data(test_data, test_trgt)
    return training_data, trn_trgt_data, test_data, test_trgt


def load_data_from_npz():
    print 'Loading data from NPZ ..'
    sys.stdout.flush()
    training_data_set = 'gen_data/full/train_data.npz'
    training_trgt_set = 'gen_data/full/train_targets_data.npz'
    test_data_set = 'gen_data/full/test_data.npz'
    test_trgt_set = 'gen_data/full/test_targets_data.npz'

    training_data = np.load(training_data_set)
    training_data = training_data['data']
    trn_trgt_data = np.load(training_trgt_set)
    trn_trgt_data = trn_trgt_data['data']

    training_data = training_data[:3000, :]
    trn_trgt_data = trn_trgt_data[:3000]

    test_data     = np.load(test_data_set)
    test_data     = test_data['data']
    test_trgt     = np.load(test_trgt_set)
    test_trgt     = test_trgt['data']

    training_data, trn_trgt_data = preprocess_training_data(training_data, trn_trgt_data)
    test_data, test_trgt = preprocess_training_data(test_data, test_trgt)
    return training_data, trn_trgt_data, test_data, test_trgt

def load_data():
    print ('Loading Input Data ..')
    sys.stdout.flush()
    training_data = np.loadtxt(train_data, delimiter = ',')
    target_data = np.loadtxt(result_data)

    training_data = training_data.astype(int)
    new_target_data = np.zeros((data_size, num_classes))

    print ('Transforming truth data ..')
    sys.stdout.flush()
    i = range(np.shape(target_data)[0] - 1)
    for j in i:
        new_target_data[j, target_data[j]] = 1

    print ('New Target Data = \n\n', new_target_data)
    sys.stdout.flush()
    training_data = training_data.astype(type(np.inf))
    target_data = target_data.astype(type(np.inf))

    min_max_scaler = preprocessing.MinMaxScaler()
    training_data = min_max_scaler.fit_transform(training_data)

    test_data = training_data[5589:][:]
    test_target_data = new_target_data[5589:][:]

    training_data = training_data[:5589][:]
    new_target_data = new_target_data[:5589][:]

    return training_data, new_target_data, test_data, test_target_data

def build_mlp_classifier(training_data, target_data, test_data, test_target_data):
    print ('Building MLP Classifier .. ')
    sys.stdout.flush()
    #mlp = MLPClassifier(solver='lbfgs', max_iter=num_iter, validation_fraction=0.1, alpha=1e-5, hidden_layer_sizes=(512, 256, 64, 6))
    mlp = MLPClassifier(solver='lbfgs', learning_rate_init = learning_rate, verbose=True, max_iter=num_iter, learning_rate='adaptive', validation_fraction=0.1, alpha=1e-5, hidden_layer_sizes=(256, 256, 128, 3))
    print ('Training Data Shape = ', np.shape(training_data))
    print ('Target Data Shape = ', np.shape(target_data))
    sys.stdout.flush()
    mlp.fit(training_data, target_data)
    print ('Built Classifier .. ')
    sys.stdout.flush()

    print ('Test Data Predictions ... ')
    print ('------------------------- ')
    sys.stdout.flush()
    predictions = mlp.predict_proba(test_data)
    i = range(len(test_data))

    print ('Predictions Shape = ', np.shape(predictions))
    print ('\n\nBelow are the actual comparisons for the performance of the ANN versus the actual target values .. ')
    sys.stdout.flush()
    correct_predictions = 0
    incorrect_predictions = 0

    for j in i:
        predicted_value = np.argmax(predictions[j])
        target_value = np.argmax(test_target_data[j])
        print predictions[j], ' :: Prediction = ', predicted_value, ' Target = ', target_value
        if predicted_value == target_value:
            correct_predictions = correct_predictions + 1
        else:
            incorrect_predictions = incorrect_predictions + 1

    print ('Correct Predictions   = ', correct_predictions)
    print ('Incorrect Predictions = ', incorrect_predictions)
    sys.stdout.flush()

    # Now test against the training data set to validate
    print ('Training Data Predictions ...')
    print ('-----------------------------')
    predictions = mlp.predict_proba(training_data)
    i = range(len(training_data))

    print ('Predictions Shape = ', np.shape(predictions))
    print ('\n\nBelow are the actual comparisons for the performance of the ANN versus the actual target values .. ')
    sys.stdout.flush()
    correct_predictions = 0
    incorrect_predictions = 0

    for j in i:
        predicted_value = np.argmax(predictions[j])
        target_value = np.argmax(target_data[j])
        print predictions[j], ' :: Prediction = ', predicted_value, ' Target = ', target_value
        if predicted_value == target_value:
            correct_predictions = correct_predictions + 1
        else:
            incorrect_predictions = incorrect_predictions + 1

    print ('Correct Predictions   = ', correct_predictions)
    print ('Incorrect Predictions = ', incorrect_predictions)
    sys.stdout.flush()

def preprocess_training_data(training_data, target_data):
    training_data = training_data.astype(int)
    target_data = target_data.astype(int)

    new_target_data = np.zeros((np.shape(training_data)[0], num_classes))
    i = range(np.shape(target_data)[0] - 1)
    for j in i:
        new_target_data[j, target_data[j]] = 1

    training_data = training_data.astype(type(np.inf))
    target_data = target_data.astype(type(np.inf))

    min_max_scaler = preprocessing.MinMaxScaler()
    training_data = min_max_scaler.fit_transform(training_data)

    return training_data, new_target_data


def get_batch(batch_number, in_dir='gen_data/', in_file='databatch_', test_file = 'testbatch_', data_key='data'):
    try:
        npz_file = np.load(in_dir + in_file + str(batch_number) + '.npz')
        # print 'Loaded npz file [', npz_file, ']'
        data_array = npz_file[data_key]

        npz_file = np.load(in_dir + test_file + str(batch_number) + '.npz')
        # print 'Loaded test file [', npz_file, ']'
        targets_array = npz_file[data_key]

        del npz_file

        return True, data_array, targets_array
    except Exception as e:
        print 'Unable to load file for batch = ', batch_number, '  ', e.message
        sys.stdout.flush()
        return False, None, None

def train(upper_batch=1000, save = True):
    mlp = MLPClassifier(solver='adam', verbose=True, max_iter=num_iter, validation_fraction=0.1, alpha=1e-5, hidden_layer_sizes=(1024, 512, 256, 128, 6))
    current_batch = 1
    found, data_batch, targets_batch = get_batch(current_batch)
    while found:
        data_batch, targets_batch = preprocess_training_data(data_batch, targets_batch)

        print '---> Shape of Data Array   = ', np.shape(data_batch)
        print '---> Shape of Target Array = ', np.shape(targets_batch)
        sys.stdout.flush()
        mlp.partial_fit(data_batch, targets_batch, classes=[0, 1, 2, 3, 4, 5])
        current_batch = current_batch + 1
        if current_batch <= upper_batch:
            found, data_batch, targets_batch = get_batch(current_batch)
        else:
            break

    if save:
        dt = datetime.now()
        _timestamp = dt.microsecond

        joblib.dump(mlp, 'gen_models/lidc_version_' + str(_timestamp) + '.pkl')
        print 'Successfully printed generated LIDC Model to gen_models directory .. '
        sys.stdout.flush()
    return mlp

def test(mlp):
    if mlp == None:
        raise ValueError('Unspecified MLP Classifier. Provide a valid MLPClassifier object to test.')

    # Load the Test Batch
    found, test_data, test_targets = get_batch(14)

    if not found:
        print 'Could not find test set for batch = 14. Returning from test function.'
        return

    test_data, test_targets = preprocess_training_data(test_data, test_targets)

    print 'Size of Test Data = ', np.shape(test_data)[0]
    print 'Size of Test Targets = ', np.shape(test_targets)[0]
    sys.stdout.flush()
    score = mlp.score(test_data, test_targets)
    print '... Validation Accuracy = ', score
    sys.stdout.flush()

if __name__ == '__main__':
    print ('Executing MLP for Voxel Learning for LIDC .')
    sys.stdout.flush()
    full_training = True
    max_batches = 2 

    if full_training:
        # Full Training Version
        print 'Running in Full Data Mode ..'
        training_data, target_data, test_data, test_target_data = load_filtered_data_from_npz()
        build_mlp_classifier(training_data, target_data, test_data, test_target_data)
    else:
        print 'Running in Batch Mode ..'
        # Batch Training Version
        mlp = train(upper_batch=max_batches, save=True)
        test(mlp)

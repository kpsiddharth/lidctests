'''
Created on Aug 23, 2017

@author: spurkayastha
'''

import numpy as np
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
num_iter = 1

def load_data():
    print ('Loading Input Data ..')
    training_data = np.loadtxt(train_data, delimiter = ',')
    target_data = np.loadtxt(result_data)

    training_data = training_data.astype(int)
    target_data = target_data.astype(int)

    new_target_data = np.zeros((data_size, num_classes))

    print ('Transforming truth data ..')
    i = range(np.shape(target_data)[0] - 1)
    for j in i:
        new_target_data[j, target_data[j]] = 1

    # print ('Sample Data = ', training_data[0][0:30])
    # print ('Sample Data = ', training_data[10][0:30])

    print ('New Target Data = \n\n', new_target_data)

    #cdata = np.concatenate((training_data, target_data), axis = 1)
    #print ('CData Shape = ', np.shape(cdata))
    
    # training_data = np.reshape(training_data, (50, 40, 40, 40))
    # target_data = np.reshape(target_data, (data_size, 1))
    # print ('Training Data Shape = ', np.shape(training_data))
    # print ('Training Results Shape = ', np.shape(target_data))
    
    training_data = training_data.astype(type(np.inf))
    target_data = target_data.astype(type(np.inf))

    min_max_scaler = preprocessing.MinMaxScaler()
    training_data = min_max_scaler.fit_transform(training_data)

    test_data = training_data[5589:][:]
    test_target_data = new_target_data[5589:][:]

    training_data = training_data[:5589][:]
    new_target_data = new_target_data[:5589][:]

    print ('Training Data Shape = ', np.shape(training_data))
    print ('Training Target Shape = ', np.shape(new_target_data))

    print ('Test Data Shape = ', np.shape(test_data))
    print ('Test Target Shape = ', np.shape(test_target_data))

    return training_data, new_target_data, test_data, test_target_data

def build_mlp_classifier(training_data, target_data, test_data, test_target_data):
    print ('Building MLP Classifier .. ')
    #mlp = MLPClassifier(solver='lbfgs', max_iter=num_iter, validation_fraction=0.1, alpha=1e-5, hidden_layer_sizes=(512, 256, 64, 6))
    mlp = MLPClassifier(solver='adam', verbose=True, max_iter=num_iter, validation_fraction=0.1, alpha=1e-5, hidden_layer_sizes=(1024, 512, 256, 128, 6))
    print ('Training Data Shape = ', np.shape(training_data))
    print ('Target Data Shape = ', np.shape(target_data))

    # print ('Number of Outputs = ', mlp.n_outputs)

    mlp.fit(training_data, target_data)
    print ('Built Classifier .. ')

    print ('Training Data Predictions ... ')
    print ('----------------------------- ')

    predictions = mlp.predict_proba(training_data)
    i = range(len(training_data))

    #print ('Number of Outputs = ', mlp.n_outputs)

    weights = mlp.get_params()
    print ('Weights Shape = ', np.shape(weights))

    print ('Predictions Shape = ', np.shape(predictions))

    print ('\n\nBelow are the actual comparisons for the performance of the ANN versus the actual target values .. ')

    correct_predictions = 0
    incorrect_predictions = 0

    print ('Shape of predictions = ', np.shape(predictions))

    for j in i:
        predicted_value = np.argmax(predictions[j])
        target_value = np.argmax(target_data[j])
        print (predictions[j], ' :: Prediction = ', predicted_value, ' Target = ', target_value)
        if predicted_value == target_value:
            correct_predictions = correct_predictions + 1
        else:
            incorrect_predictions = incorrect_predictions + 1

    print ('Correct Predictions   = ', correct_predictions)
    print ('Incorrect Predictions = ', incorrect_predictions)
    print ('Score                 = ', mlp.score(training_data))
    print ('---------------------------- ')
    print ('\n\nTest Data Predictions ...    ')
    print ('---------------------------- ')

    predictions = mlp.predict_proba(test_data)
    i = range(len(test_data))
    print ('\n\nBelow are the actual comparisons for the performance of the ANN versus actual test target values .. ')
    correct_predictions = 0
    incorrect_predictions = 0

    for j in i:
        predicted_value = np.argmax(predictions[j])
        target_value = np.argmax(test_target_data[j])
        print (predictions[j], ' :: Prediction = ', predicted_value, ' Target = ', target_value)
        if predicted_value == target_value:
            correct_predictions = correct_predictions + 1
        else:
            incorrect_predictions = incorrect_predictions + 1

    print ('Correct Predictions   = ', correct_predictions)
    print ('Incorrect Predictions = ', incorrect_predictions)
    print ('Score                 = ', mlp.score(test_data))

    #score = mlp.score(training_data, target_data)
    # print ('Accuracy = ', score)

def preprocess_training_data(training_data, target_data):
    training_data = training_data.astype(int)
    target_data = target_data.astype(int)

    new_target_data = np.zeros((np.shape(training_data)[0], num_classes))
    print ('Transforming truth data ..')
    i = range(np.shape(target_data)[0] - 1)
    for j in i:
        new_target_data[j, target_data[j]] = 1

    print ('New Target Data = \n\n', new_target_data)

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
        return False, None, None

def train(upper_batch=1000, save = True):
    mlp = MLPClassifier(solver='adam', verbose=True, max_iter=num_iter, validation_fraction=0.1, alpha=1e-5, hidden_layer_sizes=(1024, 512, 256, 128, 6))
    current_batch = 1
    found, data_batch, targets_batch = get_batch(current_batch)
    while found:
        data_batch, targets_batch = preprocess_training_data(data_batch, targets_batch)

        print '---> Shape of Data Array   = ', np.shape(data_batch)
        print '---> Shape of Target Array = ', np.shape(targets_batch)

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

    score = mlp.score(test_data, test_targets)
    print '... Validation Accuracy = ', score

if __name__ == '__main__':
    print ('Executing MLP for Voxel Learning for LIDC .')

    full_training = False
    max_batches = 2 

    if full_training:
        # Full Training Version
        print 'Running in Full Data Mode ..'
        training_data, target_data = load_data()
        build_mlp_classifier(training_data, target_data)
    else:
        print 'Running in Batch Mode ..'
        # Batch Training Version
        mlp = train(upper_batch=max_batches, save=True)
        test(mlp)

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
import math

train_data = 'datafull/final_data.csv'
result_data = 'datafull/target_data.csv'

training_data = []
target_data = []

num_iter = 3000
learning_rate = 0.001

test_data_split = 0.1

def load_data():
    print 'Loading Data ..'
    data = np.load('/home/ubuntu/sandboxes/data/2210/training_data.npz')['data']
    targets = np.load('/home/ubuntu/sandboxes/data/2210/targets_data.npz')['data']

    total_length = data.shape[0]
    test_data_length = int(math.floor(total_length * test_data_split))
    train_data_length = total_length - test_data_length

    # Shuffle up all the data
    perm = np.arange(total_length)
    np.random.shuffle(perm)
    data = data[perm]
    targets = targets[perm]

    print ('Training to Test Data split .. ' + str (test_data_split))

    training_data = data[:train_data_length, :]
    training_targets = targets[:train_data_length, :]

    test_data = data[-test_data_length:, :]
    test_targets = targets[-test_data_length:, :]

    return training_data, training_targets, test_data, test_targets

def build_mlp_classifier(training_data, target_data, test_data, test_target_data):
    print ('Building MLP Classifier .. ')
    mlp = MLPClassifier(solver='lbfgs', learning_rate_init = learning_rate, verbose=True, max_iter=num_iter, learning_rate='adaptive', validation_fraction=0.1, alpha=1e-5, hidden_layer_sizes=(2048, 1024, 512, 256, 128, 64, 2))
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

    nclasses = np.shape(target_data)[1]
    test_cm = np.zeros((nclasses,nclasses))

    for j in i:
        predicted_value = np.argmax(predictions[j])
        target_value = np.argmax(test_target_data[j])
        print predictions[j], ' :: Prediction = ', predicted_value, ' Target = ', target_value
        if predicted_value == target_value:
            correct_predictions = correct_predictions + 1
        else:
            incorrect_predictions = incorrect_predictions + 1
        test_cm[target_value][predicted_value] += 1

    test_cm = test_cm/(test_cm.sum(axis = 1, keepdims=True)*1.0)
    print "Test Confusion matrix"
    print test_cm

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

    train_cm = np.zeros((nclasses,nclasses))

    for j in i:
        predicted_value = np.argmax(predictions[j])
        target_value = np.argmax(target_data[j])
        print predictions[j], ' :: Prediction = ', predicted_value, ' Target = ', target_value
        if predicted_value == target_value:
            correct_predictions = correct_predictions + 1
        else:
            incorrect_predictions = incorrect_predictions + 1
        train_cm[target_value][predicted_value] += 1

    train_cm = train_cm/(train_cm.sum(axis = 1, keepdims=True)*1.0)
    print "Train Confusion matrix"
    print train_cm

    print ('Correct Predictions   = ', correct_predictions)
    print ('Incorrect Predictions = ', incorrect_predictions)
    print "correct %age---", (correct_predictions*100.0)/(correct_predictions + incorrect_predictions)
    sys.stdout.flush()

if __name__ == '__main__':
    print ('Executing MLP for Voxel Learning for LIDC .')
    sys.stdout.flush()
    full_training = True

    if full_training:
        # Full Training Version
        print 'Running in Full Data Mode ..'
        training_data, target_data, test_data, test_target_data = load_data()
        build_mlp_classifier(training_data, target_data, test_data, test_target_data)

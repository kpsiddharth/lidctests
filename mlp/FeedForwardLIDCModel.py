'''
Created on Aug 23, 2017

@author: spurkayastha
'''

import numpy as np
from sklearn.neural_network import MLPClassifier

train_data = 'data1500/final_data_1.csv'
result_data = 'data1500/target_data_1.csv'

training_data = []
target_data = []

data_size = 1500
num_classes = 6

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

    print ('New Target Data = \n\n', new_target_data)

    #cdata = np.concatenate((training_data, target_data), axis = 1)
    #print ('CData Shape = ', np.shape(cdata))
    
    # training_data = np.reshape(training_data, (50, 40, 40, 40))
    # target_data = np.reshape(target_data, (data_size, 1))
    print ('Training Data Shape = ', np.shape(training_data))
    print ('Training Results Shape = ', np.shape(target_data))
    
    return training_data, new_target_data

def build_mlp_classifier(training_data, target_data):
    print ('Building MLP Classifier .. ')
    #mlp = MLPClassifier(solver='lbfgs', max_iter=1000, validation_fraction=0.1, alpha=1e-5, hidden_layer_sizes=(1024, 128, 128, 64, 6))
    mlp = MLPClassifier(solver='adam', verbose=True, max_iter=10000, validation_fraction=0.1, alpha=1e-5, hidden_layer_sizes=(1024, 1024, 256, 128, 6))
    print ('Training Data Shape = ', np.shape(training_data))
    mlp.fit(training_data, target_data)
    print ('Built Classifier .. ')

    predictions = mlp.predict(training_data)
    i = range(len(training_data))

    print ('\n\nBelow are the actual comparisons for the performance of the ANN versus the actual target values .. ')

    correct_predictions = 0
    incorrect_predictions = 0

    print ('Shape of predictions = ', np.shape(predictions))

    for j in i:
        predicted_value = np.argmax(predictions[j])
        target_value = np.argmax(target_data[j])
        print ('Prediction = ', predicted_value, ' Target = ', target_value)
        if predicted_value == target_value:
            correct_predictions = correct_predictions + 1
        else:
            incorrect_predictions = incorrect_predictions + 1

    print ('Correct Predictions = ', correct_predictions)
    print ('Incorrect Predictions = ', incorrect_predictions)

    #score = mlp.score(training_data, target_data)
    
    # print ('Accuracy = ', score)

if __name__ == '__main__':
    print ('Executing MLP for Voxel Learning for LIDC .')
    training_data, target_data = load_data()
    build_mlp_classifier(training_data, target_data)

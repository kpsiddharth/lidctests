'''
Created on Aug 23, 2017

@author: spurkayastha
'''

import numpy as np
from sklearn.neural_network import MLPClassifier

train_data = 'D:\\Data\\ml\\medical\\LIDC_Converted\\final_data_1.csv'
result_data = 'D:\\Data\\ml\\medical\\LIDC_Converted\\target_data_1.csv'

training_data = []
target_data = []

def load_data():
    training_data = np.loadtxt(train_data, delimiter = ',')
    target_data = np.loadtxt(result_data)
    
    # training_data = np.reshape(training_data, (50, 40, 40, 40))
    target_data = np.reshape(target_data, (50, 1))
    print ('Training Data Shape = ', np.shape(training_data))
    print ('Training Results Shape = ', np.shape(target_data))
    
    return training_data, target_data

def build_mlp_classifier(training_data, target_data):
    print ('Building MLP Classifier .. ')
    mlp = MLPClassifier(solver='lbfgs', max_iter=250, validation_fraction=0.1, alpha=1e-5, hidden_layer_sizes=(10, 10, 2))
    print ('Training Data Shape = ', np.shape(training_data))
    mlp.fit(training_data, target_data)
    print ('Built Classifier .. ')
    score = mlp.score(training_data, target_data)
    
    print ('Accuracy = ', score)

if __name__ == '__main__':
    training_data, target_data = load_data()
    build_mlp_classifier(training_data, target_data)
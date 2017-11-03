from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import numpy as np
import math

data_base_dir = '/home/ubuntu/ml/data/mcnn_lidc/shuffled'
data_base_dir = '/home/ubuntu/data/mcnn/shuffled'
data_base_dir = '/home/ubuntu/sandboxes/data/0410'
data_base_dir = '/home/ubuntu/sandboxes/data/1210'
data_base_dir = '/home/ubuntu/sandboxes/data/2210'

class DataSet(object):
    def __init__(self, shuffle=True):
        # self._images = np.load(data_base_dir + '/scaled_training_data.npz')
        # self._labels = np.load(data_base_dir + '/shuffled_target_data.npz')

        #self._images = np.load(data_base_dir + '/two_class_training_data.npz')
        #self._labels = np.load(data_base_dir + '/two_class_data.npz')

        self._images = np.load(data_base_dir + '/training_data.npz')
        self._labels = np.load(data_base_dir + '/targets_data.npz')

        self._images = self._images['data']
        self._labels = self._labels['data']

        shuffleorder = np.arange(self._images.shape[0])
        np.random.shuffle(shuffleorder)

        self._images = self._images[shuffleorder]
        self._labels = self._labels[shuffleorder]

        # Commenting out the reshaping from 3 views to a single view
        # self._images = np.reshape(self._images, (self._images.shape[0], 3, 2601))
        # To make things easier for the moment, we will only pick one of the 
        # slices for each nodule and get it working end to end
        # images1 = self._images[:, 1, :]
        # labels1 = self._labels
        
        # images2 = self._images[:, 2, :]
        # labels2 = self._labels
        
        # images3 = self._images[:, 0, :]
        # labels3 = self._labels

        # self._images = np.concatenate((images1, images2, images3))
        # self._labels = np.concatenate((labels1, labels2, labels3))

        print ('Total Images = ' + str(len(self._images)))

        self._epochs_completed = 0
        self._index_in_epoch = 0
        #self._num_examples = 36725
        self._num_examples = len(self._images) 
        
        test_size = int(math.floor(self._num_examples * 0.2))

        self.test_images = self._images[-test_size:]
        self.test_labels = self._labels[-test_size:]

        self._num_examples = self._num_examples - test_size

        self._images = self._images[:self._num_examples]
        self._labels = self._labels[:self._num_examples]

        # Check if the input data is to be shuffled
        # then go ahead and shuffle it up nicely
        if shuffle:
            perm = np.arange(len(self._images))
            np.random.shuffle(perm)
            # Now shuffle this up nicely
            self._images = self._images[perm]
            self._labels = self._labels[perm]

        print ('Length of Training Data Set = ' + str(len(self._images)))
        print ('Length of Test Data Set     = ' + str(len(self.test_images)))

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch

        # Go to the next epoch
        if start + batch_size > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1

          if shuffle:
              shuffleorder = np.arange(self._num_examples)
              np.random.shuffle(shuffleorder)
              self._images = self._images[shuffleorder]
              self._labels = self._labels[shuffleorder]

          # Get the rest examples in this epoch
          rest_num_examples = self._num_examples - start
          images_rest_part = self._images[start:self._num_examples]
          labels_rest_part = self._labels[start:self._num_examples]

          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size - rest_num_examples
          end = self._index_in_epoch
          images_new_part = self._images[start:end]
          labels_new_part = self._labels[start:end]
          return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
          self._index_in_epoch += batch_size
          end = self._index_in_epoch
          return self._images[start:end], self._labels[start:end]

    def test_images(self):
        return self.test_images
    
    def test_labels(self):
        return self.test_labels

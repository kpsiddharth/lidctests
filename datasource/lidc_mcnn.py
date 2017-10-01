from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import numpy as np

data_base_dir = '/home/siddhartha/ml/data/mcnn_lidc/shuffled'

class DataSet(object):
    def __init__(self):
        self._images = np.load(data_base_dir + '/scaled_training_data.npz')
        self._labels = np.load(data_base_dir + '/shuffled_target_data.npz')
        self._images = self._images['data']
        self._labels = self._labels['data']
        self._images = np.reshape(self._images, (self._images.shape[0], 3, 2601))
        # To make things easier for the moment, we will only pick one of the 
        # slices for each nodule and get it working end to end
        images1 = self._images[:, 1, :]
        labels1 = self._labels
        
        images2 = self._images[:, 2, :]
        labels2 = self._labels
        
        images3 = self._images[:, 0, :]
        labels3 = self._labels

        self._images = np.concatenate((images1, images2, images3))
        self._labels = np.concatenate((labels1, labels2, labels3))

        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = self._images.shape[0]
        
        self.test_images = self._images[4990:]
        self.test_labels = self._labels[4990:]

        self._images = self._images[:-600]
        self._labels = self._labels[:-600]

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch

        # Go to the next epoch
        if start + batch_size > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
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
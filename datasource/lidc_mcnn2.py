from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import numpy as np
import math

data_base_dir = '/home/ubuntu/sandboxes/data/1610'

class DataSet(object):
    def __init__(self):
        self._images = np.load(data_base_dir + '/mcnn_training_data.npz')
        self._labels = np.load(data_base_dir + '/mcnn_target_data.npz')
        self._images = self._images['data']
        self._labels = self._labels['data']
        
        self._images = np.reshape(self._images, (self._images.shape[0], 3, 1024))

        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = self._images.shape[0]

        test_size = int(math.floor(self._num_examples * 0.1))
        self._num_examples = self._num_examples - test_size
        
        self.test_images = self._images[-test_size:]
        self.test_labels = self._labels[-test_size:]

        self._images = self._images[self._num_examples:]
        self._labels = self._labels[self._num_examples:]

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
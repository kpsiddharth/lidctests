from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import numpy as np
import math

data_base_dir = '/home/ubuntu/sandboxes/data/1610'

class DataSet(object):
    def __init__(self):
        self._images1 = np.load(data_base_dir + '/mcnn_s1_training_data.npz')['data']
        self._images2 = np.load(data_base_dir + '/mcnn_s2_training_data.npz')['data']
        self._images3 = np.load(data_base_dir + '/mcnn_s3_training_data.npz')['data']

        self._labels = np.load(data_base_dir + '/mcnn_target_data.npz')['data']
 
        print ('Images 1 = ' + str(len(self._images1)))

        #self._images1 = self._images1[0::3, :]
        #self._images2 = self._images2[1::3, :]
        #self._images3 = self._images3[2::3, :]

        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = self._images1.shape[0]

        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images1 = self._images1[perm]
        self._images2 = self._images2[perm]
        self._images3 = self._images3[perm]
        self._labels = self._labels[perm]

        print ('Number of Examples = ' + str(self._num_examples))
        test_size = int(math.floor(self._num_examples * 0.1))
        self._num_examples = self._num_examples - test_size

        self.test_images1 = self._images1[-test_size:]
        self.test_images2 = self._images2[-test_size:]
        self.test_images3 = self._images3[-test_size:]

        self.test_labels = self._labels[-test_size:]

        self._images1 = self._images1[:self._num_examples]
        self._images2 = self._images2[:self._num_examples]
        self._images3 = self._images3[:self._num_examples]
        
        self._labels = self._labels[:self._num_examples]

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch

        # Go to the next epoch
        if start + batch_size > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1

          if shuffle:
              print ('Shuffling number of examples .. ')
              perm = np.arange(self._num_examples)
              np.random.shuffle(perm)
              self._images1 = self._images1[perm]
              self._images2 = self._images2[perm]
              self._images3 = self._images3[perm]
              self._labels = self._labels[perm]

          # Get the rest examples in this epoch
          rest_num_examples = self._num_examples - start
          images_rest_part1 = self._images1[start:self._num_examples]
          images_rest_part2 = self._images2[start:self._num_examples]
          images_rest_part3 = self._images3[start:self._num_examples]

          labels_rest_part = self._labels[start:self._num_examples]

          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size - rest_num_examples
          end = self._index_in_epoch
          images_new_part1 = self._images1[start:end]
          images_new_part2 = self._images2[start:end]
          images_new_part3 = self._images3[start:end]

          labels_new_part = self._labels[start:end]
          return np.concatenate((images_rest_part1, images_new_part1), axis=0) , np.concatenate((images_rest_part2, images_new_part2), axis=0), np.concatenate((images_rest_part3, images_new_part3), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
          self._index_in_epoch += batch_size
          end = self._index_in_epoch
          return self._images1[start:end], self._images2[start:end], self._images3[start:end], self._labels[start:end]

    def test_images(self):
        return self.test_images1, self.test_images2, self.test_images3
    
    def test_labels(self):
        return self.test_labels

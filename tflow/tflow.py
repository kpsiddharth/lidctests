import tensorflow as tf
from tensorflow.examples.tutorials.mnist import mnist, input_data
from datasource.lidc_mcnn import DataSet


def weight_variable(shape):
    """
    Create and return weight variables with normal distributions
    with a slightly positive bias and marginally broken symmetries
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """
    Creates and returns constants with values of 0.1 of a given
    shape, as passed in the method argument.
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
  
def conv2d(x, W):
    """
    Returns a 2D Convolutional layer of the specified input and 
    weights.
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
def max_pool_2x2(x):
    """
    Returns a 2D Max pooling layer of the specified input and 
    weights.
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def resize_input_image(x):
    # Below is creating a 4D tensor to be used to
    # train the model.
    x_image = tf.reshape(x, [-1, 32, 32, 1])
    return x_image

x = tf.placeholder(tf.float32, shape=[None, 1024])
y_ = tf.placeholder(tf.float32, shape=[None, 3])

# Building the first convolutional layer.
# Kernels creation
W_conv1 = weight_variable([3, 3, 1, 32])

# Single bias tensor per kernel
b_conv1 = bias_variable([32])

x_image = resize_input_image(x)
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

print ('HConv 1 = ' + str(h_conv1.shape))
print ('HConv 2 = ' + str(h_conv2.shape))

# Fully connected layer
num_neurons = 1024
W_fc1 = weight_variable([8 * 8 * 64, num_neurons])
b_fc1 = bias_variable([num_neurons])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Output layer
W_fc2 = weight_variable([1024, 3])
b_fc2 = bias_variable([3])

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2


# Run and train the model
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
dataset = DataSet()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        batch = dataset.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1]})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: dataset.test_images, y_: dataset.test_labels}))
    

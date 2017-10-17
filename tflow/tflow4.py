import tensorflow as tf
from tensorflow.examples.tutorials.mnist import mnist, input_data
from datasource.lidc_mcnn import DataSet


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

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
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    
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
y_ = tf.placeholder(tf.float32, shape=[None, 2])

# Building the first convolutional layer.
# Kernels creation
W_conv1 = weight_variable([3, 3, 1, 32])

# Single bias tensor per kernel
b_conv1 = bias_variable([32])

x_image = resize_input_image(x)
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_conv4 = weight_variable([5, 5, 128, 256])
b_conv4 = bias_variable([256])

h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

W_conv5 = weight_variable([5, 5, 256, 512])
b_conv5 = bias_variable([512])

h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
h_pool5 = max_pool_2x2(h_conv5)

print ('hconv_1 shape = ' + str(h_conv1.shape))
print ('hconv_2 shape = ' + str(h_conv2.shape))
print ('hconv_3 shape = ' + str(h_conv3.shape))
print ('hconv_4 shape = ' + str(h_conv4.shape))
print ('hconv_5 shape = ' + str(h_conv5.shape))

# Fully connected layer
num_neurons_1 = 1024
num_neurons_2 = 512

W_fc1 = weight_variable([8 * 8 * 512, num_neurons_1])
b_fc1 = bias_variable([num_neurons_1])

h_pool_flat = tf.reshape(h_pool5, [-1, 8 * 8 * 512])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([num_neurons_1, num_neurons_2])
b_fc2 = bias_variable([num_neurons_2])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Output layer
W_fc3 = weight_variable([num_neurons_2, 2])
b_fc3 = bias_variable([2])

keep_prob_2 = tf.placeholder(tf.float32)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob_2)

y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3


# Run and train the model
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('accuracy', accuracy)

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
dataset = DataSet()
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('summaries/train4', sess.graph)
    for i in range(1000):
        batch = dataset.next_batch(512)
        if i % 100 == 0 or True:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0, keep_prob_2: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        if i % 1000 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: dataset.test_images, y_: dataset.test_labels, keep_prob: 1.0, keep_prob_2: 1.0})
            print ('Milestone Step Accuracy = %g' % (train_accuracy))

        # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, keep_prob_2: 0.5})
        merged = tf.summary.merge_all()
        # train_step.run(feed_dict={x1: batch[0], x2: batch[1], x3: batch[2], y_: batch[3], keep_prob: 0.5, keep_prob_2: 0.5})
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, keep_prob_2: 0.5})
        train_writer.add_summary(summary, i)

   
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: dataset.test_images, y_: dataset.test_labels, keep_prob: 1.0, keep_prob_2: 1.0}))
    train_writer.close() 

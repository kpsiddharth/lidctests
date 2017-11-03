import tensorflow as tf
from tensorflow.examples.tutorials.mnist import mnist, input_data
from datasource.lidc_mcnn2 import DataSet

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

def max_pool_ixz(x):
    #with tf.device("/cpu:0"):
    #    return tf.nn.max_pool(x, ksize=[1, 1, 1, 512],
    #                      strides=[1, 1, 1, 512], padding='SAME')
    return tf.reduce_mean(x, axis=[3], keep_dims=True)

def avg_pool_1x3(x):
    with tf.device("/cpu:0"):
        return tf.nn.avg_pool(x, ksize=[1, 1, 1, 3],
                          strides=[1, 1, 1, 3], padding='SAME')

def max_pool_1x3(x):
    # with tf.device("/cpu:0"):
    #    return tf.nn.max_pool(x, ksize=[1, 1, 1, 3],
    #                      strides=[1, 1, 1, 3], padding='SAME')
    return tf.reduce_mean(x, axis=[3], keep_dims=True)

def resize_input_image(x):
    # Below is creating a 4D tensor to be used to
    # train the model.
    x_image = tf.reshape(x, [-1, 32, 32, 1])
    return x_image


# Inputs: 3 Adjacent layers
x1 = tf.placeholder(tf.float32, shape=[None, 1024])
x2 = tf.placeholder(tf.float32, shape=[None, 1024])
x3 = tf.placeholder(tf.float32, shape=[None, 1024])

# Output
y_ = tf.placeholder(tf.float32, shape=[None, 2])

# Building the first convolutional layer.
# Kernels creation
W_conv1 = weight_variable([3, 3, 1, 32])
variable_summaries(W_conv1)

# Single bias tensor per kernel
b_conv1 = bias_variable([32])

x_image1 = resize_input_image(x1)
x_image2 = resize_input_image(x2)
x_image3 = resize_input_image(x3)

# CNN1 for Image 1
h_conv1_1 = tf.nn.relu(conv2d(x_image1, W_conv1) + b_conv1)

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2_1 = tf.nn.relu(conv2d(h_conv1_1, W_conv2) + b_conv2)

W_conv3 = weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])

h_conv3_1 = tf.nn.relu(conv2d(h_conv2_1, W_conv3) + b_conv3)

W_conv4 = weight_variable([5, 5, 128, 256])
b_conv4 = bias_variable([256])

h_conv4_1 = tf.nn.relu(conv2d(h_conv3_1, W_conv4) + b_conv4)

W_conv5 = weight_variable([5, 5, 256, 512])
b_conv5 = bias_variable([512])

h_conv5_1 = tf.nn.relu(conv2d(h_conv4_1, W_conv5) + b_conv5)
with tf.device("/cpu:0"):
    h_pool5_1 = max_pool_ixz(max_pool_2x2(h_conv5_1))
# ## End of CNN 1 for Image 1

# CNN1 for Image 2
h_conv1_2 = tf.nn.relu(conv2d(x_image2, W_conv1) + b_conv1)
h_conv2_2 = tf.nn.relu(conv2d(h_conv1_2, W_conv2) + b_conv2)
h_conv3_2 = tf.nn.relu(conv2d(h_conv2_2, W_conv3) + b_conv3)
h_conv4_2 = tf.nn.relu(conv2d(h_conv3_2, W_conv4) + b_conv4)
h_conv5_2 = tf.nn.relu(conv2d(h_conv4_2, W_conv5) + b_conv5)

with tf.device("/cpu:0"):
    h_pool5_2 = max_pool_ixz(max_pool_2x2(h_conv5_2))
# ## End of CNN 1 for Image 2

# CNN1 for Image 2
h_conv1_3 = tf.nn.relu(conv2d(x_image3, W_conv1) + b_conv1)
h_conv2_3 = tf.nn.relu(conv2d(h_conv1_3, W_conv2) + b_conv2)
h_conv3_3 = tf.nn.relu(conv2d(h_conv2_3, W_conv3) + b_conv3)
h_conv4_3 = tf.nn.relu(conv2d(h_conv3_3, W_conv4) + b_conv4)
h_conv5_3 = tf.nn.relu(conv2d(h_conv4_3, W_conv5) + b_conv5)

with tf.device("/cpu:0"):
    h_pool5_3 = max_pool_ixz(max_pool_2x2(h_conv5_3))
# ## End of CNN 1 for Image 2

to_be_pooled = tf.concat([h_pool5_1, h_pool5_2, h_pool5_3], axis=3)
with tf.device("/cpu:0"):
    to_be_pooled = max_pool_1x3(to_be_pooled)

print ('hconv_1 shape = ' + str(h_pool5_1.shape))
print ('hconv_2 shape = ' + str(h_pool5_2.shape))
print ('hconv_3 shape = ' + str(h_pool5_3.shape))
print ('to_be_pooled shape = ' + str(to_be_pooled.shape))

# ## Implementing CNN 2
cnn2_W_conv1 = weight_variable([2, 2, 1, 32])
cnn2_b_conv1 = bias_variable([32])

cnn2_h_conv1_1 = tf.nn.relu(conv2d(to_be_pooled, cnn2_W_conv1) + cnn2_b_conv1)

cnn2_W_conv2 = weight_variable([3, 3, 32, 64])
cnn2_b_conv2 = bias_variable([64])

cnn2_h_conv1_2 = tf.nn.relu(conv2d(cnn2_h_conv1_1, cnn2_W_conv2) + cnn2_b_conv2)

cnn2_h_pool5_1 = max_pool_2x2(cnn2_h_conv1_2)
# ## End of CNN 2

# Fully connected layer
num_neurons_1 = 1024
num_neurons_2 = 512

W_fc1 = weight_variable([3 * 3 * 64, num_neurons_1])
b_fc1 = bias_variable([num_neurons_1])

h_pool_flat = tf.reshape(cnn2_h_pool5_1, [-1, 3 * 3 * 64])
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

### ##############################
dataset = DataSet()
merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('summaries/train', sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch = dataset.next_batch(32)
        if not len(batch[0]) == 128:
            print ('---> Bad Batch Length: ' + str(len(batch[0])))
        if i % 100 == 0 or True:
            train_accuracy = accuracy.eval(feed_dict={
                x1: batch[0], x2: batch[1], x3: batch[2], y_: batch[3], keep_prob: 1.0, keep_prob_2: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        if i % 1000 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x1: dataset.test_images1, x2: dataset.test_images2, x3: dataset.test_images3, y_: dataset.test_labels, keep_prob: 1.0, keep_prob_2: 1.0})
            print ('Milestone Step Accuracy = %g' % (train_accuracy))

        merged = tf.summary.merge_all()
        # train_step.run(feed_dict={x1: batch[0], x2: batch[1], x3: batch[2], y_: batch[3], keep_prob: 0.5, keep_prob_2: 0.5})
        summary, _ = sess.run([merged, train_step], feed_dict={x1: batch[0], x2: batch[1], x3: batch[2], y_: batch[3], keep_prob: 0.5, keep_prob_2: 0.5})
        train_writer.add_summary(summary, i)
   
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x1: dataset.test_images1, x2: dataset.test_images2, x3: dataset.test_images3, y_: dataset.test_labels, keep_prob: 1.0, keep_prob_2: 1.0}))
    train_writer.close()
    

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import mnist, input_data
from datasource.lidc_mcnn import DataSet
import numpy as np

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
    x_image = tf.reshape(x, [-1, 1024])
    return x_image


x = tf.placeholder(tf.float32, shape=[None, 1024])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

x_image = resize_input_image(x)

W_fc1= weight_variable([1024, 784])
b_fc1 = bias_variable([784])

h_fc1 = tf.nn.relu(tf.matmul(x_image, W_fc1) + b_fc1)

keep_prob_1 = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_1)

W_fc2 = weight_variable([784, 512])
b_fc2 = bias_variable([512])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

keep_prob_2 = tf.placeholder(tf.float32)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob_2)

W_fc3 = weight_variable([512, 256])
b_fc3 = bias_variable([256])

h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

keep_prob_3 = tf.placeholder(tf.float32)
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob_1)

W_fc4 = weight_variable([256, 64])
b_fc4 = bias_variable([64])

h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)

keep_prob_4 = tf.placeholder(tf.float32)
h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob_4)

W_fc5 = weight_variable([64, 2])
b_fc5 = bias_variable([2])

y_conv = tf.matmul(h_fc4_drop, W_fc5) + b_fc5

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
    for i in range(20000):
        batch = dataset.next_batch(512)
        if i % 50 == 0 or True:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob_1: 1.0, keep_prob_2: 1.0, keep_prob_3: 1.0, keep_prob_4: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        if i % 1000 == 0:
            #train_accuracy = accuracy.eval(feed_dict={
            #    x: dataset.test_images, y_: dataset.test_labels, keep_prob: 1.0, keep_prob_2: 1.0})
            print ('Milestone Step Accuracy = %g' % (1))

        # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, keep_prob_2: 0.5})
        merged = tf.summary.merge_all()
        # train_step.run(feed_dict={x1: batch[0], x2: batch[1], x3: batch[2], y_: batch[3], keep_prob: 0.5, keep_prob_2: 0.5})
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob_1: 0.5, keep_prob_2: 0.5, keep_prob_3: 0.5, keep_prob_4: 0.5})
        train_writer.add_summary(summary, i)

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: dataset.test_images, y_: dataset.test_labels, keep_prob_1: 1.0, keep_prob_2: 1.0, keep_prob_3: 1.0, keep_prob_4: 1.0}))
    
    predictions = sess.run(y_conv, feed_dict={
        x: dataset.test_images, keep_prob_1: 1.0, keep_prob_2: 1.0, keep_prob_3: 1.0, keep_prob_4: 1.0})
    # print (predictions)

    true_class = np.argmax(dataset.test_labels, 1)
    predicted_class = np.argmax(predictions, 1)

    cm = tf.confusion_matrix(predicted_class, true_class)
    print (cm)
    import ipdb
    ipdb.set_trace()
    train_writer.close() 

# -*- coding: utf-8 -*-
"""
ex11_one_layer_nn on Mnist

2019/04/17

"""

# imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

img_h = img_w = 28             # MNIST images are 28x28
img_size_flat = img_h * img_w  # 28x28=784, the total number of pixels
n_classes = 10                 # Number of classes, 0-9 one class per digit

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(mnist.train.labels)))
print("- Test-set:\t\t{}".format(len(mnist.test.labels)))
print("- Validation-set:\t{}".format(len(mnist.validation.labels)))

# weight(beta1) and bais(beta0) wrappers
def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W_' + name,
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)

def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name,
                           dtype=tf.float32,
                           initializer=initial)
#dense layer
def fc_layer(x, num_units, name, use_relu=True):
    """
    Create a fully-connected layer
   
    """
    with tf.variable_scope(name):
        in_dim = x.get_shape()[1]
        W = weight_variable(name, shape=[in_dim, num_units])
        b = bias_variable(name, [num_units])
        layer = tf.matmul(x, W)
        layer += b
        if use_relu:
            layer = tf.nn.relu(layer)
        return layer
#convolutional layer
def conv_layer(x, filter_size, num_filters, stride, name):
    """
    Create a 2D convolution layer
    
    """
    with tf.variable_scope(name):
        num_in_channel = x.get_shape().as_list()[-1]
        shape = [filter_size, filter_size, num_in_channel, num_filters]
        W = weight_variable(name,shape=shape)
        tf.summary.histogram('weight', W)
        b = bias_variable(name,shape=[num_filters])
        tf.summary.histogram('bias', b)
        layer = tf.nn.conv2d(x, W,
                             strides=[1, stride, stride, 1],
                             padding="SAME")
        layer += b
        return tf.nn.relu(layer)
      
def max_pool(x, ksize, stride, name):
    """
    Create a max pooling layer

    """
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding="SAME",
                          name=name)
  
def flatten_layer(layer):
    """
    Flattens the output of the convolutional layer to be fed into fully-connected layer
    """
    with tf.variable_scope('Flatten_layer'):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat

# hyper-parameters
logs_path = "./logs/visualize_graph"  

learning_rate = 0.001  # learning rate
epochs = 10  # Total number of training epochs
batch_size = 100  # Training batch size
display_freq = 300  # Frequency of displaying the training results

# Network Parameters
img_h = img_w = 28 # MNIST are 28 pixels in each dimension.
img_size_flat = img_h * img_w # Images are stored in one-dimensional arrays of this length.
n_classes = 10 # Number of classes, one class for each of 10 digits.
# number of units in the first hidden layer
h1 = 1


# Create network sturcture
# Placeholders for inputs (x), outputs(y)
with tf.variable_scope('Input'):
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X')
    y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')
#fc1 = fc_layer(x, h1, 'Hidden_layer', use_relu=True)
output_logits = fc_layer(x, n_classes, 'Output_layer', use_relu=False)

# Define the loss function, optimizer, and accuracy
with tf.variable_scope('Train'):
    with tf.variable_scope('Loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
    with tf.variable_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
    with tf.variable_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        # prediction
        cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph (session)
sess = tf.InteractiveSession() # using InteractiveSession instead of Session to test network in separate cell
sess.run(init)
train_writer = tf.summary.FileWriter(logs_path, sess.graph)
num_tr_iter = int(mnist.train.num_examples / batch_size)
import time
a = time.time()
#initialize step
global_step = 0
for epoch in range(epochs):
    print('Training epoch: {}'.format(epoch))
    for iteration in range(num_tr_iter):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        global_step += 1
        # Run optimization op (backprop)
        feed_dict_batch = {x: batch_x, y: batch_y}
        _ = sess.run(optimizer, feed_dict=feed_dict_batch)

        if iteration % display_freq == 0:
            # Calculate and display the batch loss and accuracy
            loss_batch, acc_batch = sess.run([loss, accuracy],
                                             feed_dict=feed_dict_batch)
            print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                  format(iteration, loss_batch, acc_batch))

    # Run validation after every epoch
    feed_dict_valid = {x: mnist.validation.images, y: mnist.validation.labels}
    loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
    print('---------------------------------------------------------')
    print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
          format(epoch + 1, loss_valid, acc_valid))

    
print("time cost:", time.time()- a)

# Test the network after training
# Accuracy
feed_dict_test = {x: mnist.test.images, y: mnist.test.labels}
loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
print('---------------------------------------------------------')
print("One layer perception evaluation on Mnist:")

print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
sess.close()

total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()

    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value

    total_parameters += variable_parameters
print("total parameters:", total_parameters)


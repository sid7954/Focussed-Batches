# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Copied from: https://raw.githubusercontent.com/tensorflow/tensorflow/r1.3/tensorflow/examples/tutorials/mnist/mnist_deep.py
A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import tqdm
import os
from scipy.misc import imsave
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import tensorflow as tf

import vae.data_utils as data_utils

FLAGS = None

NUM_CLASSES = 36

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, NUM_CLASSES])
    b_fc2 = bias_variable([NUM_CLASSES])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  # Import data
  data = data_utils.get_dataset('gfonts')
  
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.GradientDescentOptimizer(0.07).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  print (np.shape(data.test.images))
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(25000):
      batch = data.tgen.next_batch(50)
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        if i >= 6000:
            test_acc = accuracy.eval(feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1.0})
            print ('Test acc: ', test_acc)
        print('step %d, training accuracy %g' % (i, train_accuracy))
      if i< 15000 :
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

      if i >= 15000:
          batch = data.tgen.next_batch(500)
          logit_layer = sess.run(y_conv, feed_dict={x: batch[0], y_: batch[1], keep_prob:1.0}) 
          sim = cosine_similarity(np.array(logit_layer))
          #print (sim.shape)
          correct_labels = np.argmax(np.array(batch[1]), 1)
          correct_labels = np.tile(np.expand_dims(correct_labels, 1), (1, len(batch[1])))
          label_pairs = np.concatenate([np.expand_dims(correct_labels, 2), np.expand_dims(np.transpose(correct_labels), 2)], 2)
          correct_label_pairs = (label_pairs[:, :, 0] == label_pairs[:, :, 1])
          unmatch_label_pairs = 1.0 - correct_label_pairs
          sim_unmatch = sim + np.log(unmatch_label_pairs+1e-15)
          sim_unmatch_relevant = sim_unmatch[0:25 , 25:]
          #print (sim_unmatch_relevant.tolist()[:1][:50])
          #print ('SIM unmatch relevant: ', sim_unmatch_relevant.shape)
          argmax_indices = np.argmax(sim_unmatch_relevant, axis=1)
          #print ('ARGMAX indices: ', argmax_indices.shape)
          #print ("Argmax values: ", argmax_indices)
          argmax_indices = [x_ for x_ in argmax_indices]
          batch_new = []
          batch_new.append(batch[0][:25] + np.array(batch[0])[argmax_indices].tolist())
          batch_new.append(batch[1][:25] + np.array(batch[1])[argmax_indices].tolist())
          #print (len(batch_new[0]), len(batch_new[1]))
          #print ('X: ', x, 'y: ', y_, 'keep_prob: ', keep_prob)
          train_step.run(feed_dict={x: batch_new[0], y_: batch_new[1], keep_prob:0.5})


    np_a, np_cp, np_p, np_t = sess.run([accuracy, correct_prediction, tf.argmax(y_conv, 1), tf.argmax(y_, 1)],
                                       feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1.0})
    wrong = np.reshape(np.where(np_cp==0), [-1])
    # for w in tqdm.tqdm(wrong):
    #   imsave("test_wrong/%d_as_%d_%d.png" % (np_t[w], np_p[w], w), np.reshape(mnist.test.images[w], [28, 28]))
    
    # print ("Saved %d images to test_wrong" % len(wrong))
    print ("Test accuracy: %f" % np_a)
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='vae/data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
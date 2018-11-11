# FOCUSED BATCHES BASED ON IMPROVEMENTS ON VAL-DATASET

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tqdm

from collections import OrderedDict

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

FLAGS = None
batch_size = 50
PICK_SIZE = 500

def model_fn(x, y_, wts=None, reuse=None):
  with tf.variable_scope('model_fn', reuse=reuse):
    W = tf.get_variable("W", [784, 20])
    b = tf.get_variable("b", [20])
    W2 = tf.get_variable("W2", [20, 10])
    b2 = tf.get_variable("b2", [10])
  y = tf.matmul(tf.nn.relu(tf.matmul(x, W) + b), W2) + b2

  if wts is None:
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  else:
    cross_entropy = tf.reduce_sum(wts*tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  
  return cross_entropy, y  

def focus_batch2(np_train_grads, np_valid_grads):
  train_grads, valid_grads = tf.constant(np_train_grads), tf.constant(np_valid_grads)
  wt_logits = tf.get_variable("wt_logits", [PICK_SIZE], initializer=tf.zeros_initializer)
  wts = tf.nn.softmax(wt_logits)
  train_grads = tf.reduce_sum(train_grads*wts)
  loss = -tf.reduce_sum(train_grads*valid_grads) + .1*tf.reduce_mean(wt_logits*wt_logits)
  #return tf.train.GradientDescentOptimizer(0.1).minimize(loss, var_list=[wt_logits]), loss, tf.nn.top_k(wts, k=batch_size)[1]
  return tf.train.GradientDescentOptimizer(0.1).minimize(loss, var_list=[wt_logits]), loss, tf.range(batch_size)

def get_focus_batch(sess, feed_dict):
  assert type(feed_dict) == OrderedDict
  
  train_x, train_y, valid_x, valid_y = feed_dict.keys()
  tvars = tf.trainable_variables()

  valid_loss, _ = model_fn(valid_x, valid_y, reuse=True)
  valid_loss = tf.reduce_mean(valid_loss)
  
  valid_grads = tf.gradients(valid_loss, tvars)
  valid_grads = tf.concat([tf.reshape(_vg, [-1]) for _vg in valid_grads if _vg is not None], axis=0)
  
  tgrads = []
      
  train_loss, _ = model_fn(train_x, train_y, reuse=True)
  for i in tqdm.tqdm(range(PICK_SIZE)):
    train_grads = tf.gradients(train_loss[i], tvars)
    train_grads = tf.concat([tf.reshape(_tg, [-1]) for _tg in train_grads if _tg is not None], axis=0)
    tgrads.append(train_grads)
    
  print ("Shape debug")
  print (len(tgrads), tgrads[0].get_shape())

  np_tgrads = sess.run(tgrads, feed_dict)
  np_valid_grads = sess.run(valid_grads, feed_dict)

  opt, loss, focus_indices = focus_batch2(np_tgrads, np_valid_grads)
  ol = sess.run(loss)
  for j in range(100):
    sess.run(opt)
  l = sess.run(loss)
  print ("final loss: %f->%f" % (ol, l))
  return tf.gather(train_x, focus_indices), tf.gather(train_y, focus_indices)

def focus_batch(train_x, train_y, valid_x, valid_y):
  tvars = tf.trainable_variables()
  wt_logits = tf.get_variable("wt_logits", [PICK_SIZE], initializer=tf.random_normal_initializer())
  focus_indices = tf.nn.top_k(wt_logits, k=batch_size)[1]
  wts = tf.ones([PICK_SIZE])*0.1
  wts += tf.reduce_sum(tf.map_fn(lambda _: tf.one_hot(_, PICK_SIZE)*.9, focus_indices, dtype=tf.float32), axis=0)
  
  train_loss, _ = model_fn(train_x, train_y, wts, reuse=True)
  valid_loss, _ = model_fn(valid_x, valid_y, reuse=True)
  
  train_grads = tf.gradients(train_loss, tvars)
  valid_grads = tf.gradients(valid_loss, tvars)
  
  l1 = -sum([tf.reduce_sum(train_grads[_wi]*valid_grads[_wi])/(tf.norm(train_grads[_wi])*tf.norm(valid_grads[_wi])) for _wi in range(len(train_grads))])
  l2 = .1*tf.reduce_mean(wt_logits*wt_logits)
  loss = l1+l2
  return tf.train.GradientDescentOptimizer(1).minimize(loss, var_list=[wt_logits]), loss, tf.nn.top_k(wts, k=batch_size)[1]

def sharpness_based_focus_batch(train_x, train_y):
  eps = .01
  tvars = tf.trainable_variables()
  wt_logits = tf.get_variable("wt_logits", [PICK_SIZE], initializer=tf.ones_initializer())
  wts = tf.nn.softmax(wt_logits)

  unwtd_train_loss, _ = model_fn(train_x, train_y, reuse=True)
  train_loss, _ = model_fn(train_x, train_y, wts, reuse=True)
  g_theta = tf.gradients(train_loss, tvars)

  grad_x = tf.gradients(unwtd_train_loss, train_x)[0]
  
  for ti, tvar in enumerate(tvars):
    tvar.assign(tvar + eps*g_theta[ti])
  loss_post_update, _ = model_fn(train_x, train_y, reuse=True)
  grad_x_post_update = tf.gradients(loss_post_update, train_x)[0]
  H_x_theta_g_theta = (grad_x_post_update - grad_x)/eps
  for ti, tvar in enumerate(tvars):
    tvar.assign(tvar - eps*g_theta[ti])

  total_loss = 0
  _hx = tf.reshape(H_x_theta_g_theta, [PICK_SIZE , -1])
  grad_x = tf.reshape(grad_x, [PICK_SIZE, -1])
  batch_loss = tf.reduce_sum((H_x_theta_g_theta + 2*grad_x) * H_x_theta_g_theta, axis=1)
  batch_loss /= loss_post_update + 1
  total_loss += tf.reduce_mean(batch_loss, axis=0)*100

  total_loss += .1*tf.reduce_mean(wt_logits*wt_logits)
  return tf.train.GradientDescentOptimizer(1).minimize(total_loss, var_list=[wt_logits]), total_loss, tf.nn.top_k(wts, k=batch_size)[1]
    
def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  
  train_x = tf.placeholder(tf.float32, [None, 784])
  train_y = tf.placeholder(tf.float32, [None, 10])
  train_loss, y = model_fn(train_x, train_y)
  
  valid_x = tf.placeholder(tf.float32, [None, 784])
  valid_y = tf.placeholder(tf.float32, [None, 10])
    
  # cross_entropy, y = model_fn(train_x, train_y)
  if not FLAGS.random:
    print ("Invoking focus batch")
    batch_op, loss, focus_indices = focus_batch(train_x, train_y, valid_x, valid_y)
  else:
    focus_indices = tf.random_uniform(shape=[batch_size], minval=0, maxval=PICK_SIZE, dtype=tf.int32)
  
  x, y_ = tf.gather(train_x, focus_indices), tf.gather(train_y, focus_indices)
  x, y_ = train_x, train_y
  train_loss, y = model_fn(x, y_, reuse=True)

  train_op = tf.train.GradientDescentOptimizer(0.01).minimize(train_loss)

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  merged_summaries = tf.summary.merge_all()

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
  config = tf.ConfigProto(gpu_options=gpu_options)
  sess = tf.Session(config=config)
  sess.run(tf.global_variables_initializer())

  # Train
  num_steps = 5000
  for i in tqdm.tqdm(range(num_steps)):
    np_valid_xs, np_valid_ys = mnist.validation.next_batch(PICK_SIZE)
    np_train_xs, np_train_ys = mnist.train.next_batch(PICK_SIZE)

    if not FLAGS.random:
      start_bl, prev_bl = None, 0
      j=0
      while True:
        _, bl = sess.run([batch_op, loss], feed_dict={train_x: np_train_xs, train_y: np_train_ys, valid_x: np_valid_xs, valid_y: np_valid_ys})
        if start_bl is None:
          start_bl = bl
        if abs(bl-prev_bl)<1E-6 or j>100:
          break
        prev_bl = bl
        j += 1
    print ("Batch loss: Number of steps: %d %f -> %f" % (j, start_bl, bl))
    sess.run(train_op, feed_dict={train_x: np_train_xs, train_y: np_train_ys})

    # focus_x, focus_y = get_focus_batch(sess, OrderedDict(
    #   {train_x: np_train_xs, train_y: np_train_ys, valid_x: np_valid_xs, valid_y: np_valid_ys}))
    # sess.run(train_op, feed_dict={train_x: sess.run(focus_x), train_y: sess.run(focus_y)})

    if i%10==0:
      print('Test Acc: (%d) ' % i, sess.run(accuracy, feed_dict={train_x: mnist.test.images,
                                                        train_y: mnist.test.labels}))

  
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='vae/data/mnist',
                      help='Directory for storing input data')
  parser.add_argument('--random', action='store_true', default=False, help='random or focussed batches')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

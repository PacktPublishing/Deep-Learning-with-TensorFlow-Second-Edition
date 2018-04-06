import os
import re
import io
import requests
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

n_inputs = 3
n_neurons = 5

X1 = tf.placeholder(tf.float32, [None, n_inputs])
X2 = tf.placeholder(tf.float32, [None, n_inputs])

Wx = tf.get_variable("Wx", shape=[n_inputs,n_neurons], dtype=tf.float32, initializer=None, regularizer=None, trainable=True, collections=None)

Wy = tf.get_variable("Wy", shape=[n_neurons,n_neurons], dtype=tf.float32, initializer=None, regularizer=None, trainable=True, collections=None)

b = tf.get_variable("b", shape=[1,n_neurons], dtype=tf.float32, initializer=None, regularizer=None, trainable=True, collections=None)

Y1 = tf.nn.relu(tf.matmul(X1, Wx) + b)
Y2 = tf.nn.relu(tf.matmul(Y1, Wy) + tf.matmul(X2, Wx) + b)

init_op = tf.global_variables_initializer()

# Mini-batch: instance 0,instance 1,instance 2,instance 3
X1_batch = np.array([[0, 2, 3], [2, 8, 9], [5, 3, 8], [3, 2, 9]]) # t = 0
X2_batch = np.array([[5, 6, 8], [1, 0, 0], [8, 2, 0], [2, 3, 6]]) # t = 1

with tf.Session() as sess:
	init_op.run()
	Y1_val, Y2_val = sess.run([Y1, Y2], feed_dict={X1: X1_batch, X2: X2_batch})

print(Y1_val) # output at t = 0
print(Y2_val) # output at t = 1

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
n_steps = 2

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
seq_length = tf.placeholder(tf.int32, [None])

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

# Mini-batch: instance 0,instance 1,instance 2,instance 3
X_batch = np.array([
                   [[0, 2, 3], [2, 8, 9]], # instance 0
		   [[5, 6, 8], [0, 0, 0]], # instance 1 (padded with a zero vector)
                   [[6, 7, 8], [6, 5, 4]], # instance 2
                   [[8, 2, 0], [2, 3, 6]], # instance 3
                  ])

seq_length_batch = np.array([3, 4, 3, 5])

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
	init_op.run()
	outputs_val, states_val = sess.run([output_seqs, states], feed_dict={X: X_batch, seq_length: seq_length_batch})

print(outputs_val)
print(states_val)

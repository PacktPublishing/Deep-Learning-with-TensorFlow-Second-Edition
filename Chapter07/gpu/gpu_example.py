import numpy as np
import tensorflow as tf
import datetime

n = 10

A = np.random.rand(10000, 10000).astype('float32')
B = np.random.rand(10000, 10000).astype('float32')


c1 = []
c2 = []

def matpow(M, n):
    if n == 1: 
        return M
    else:
        return tf.matmul(M, matpow(M, n-1))

with tf.device('/gpu:0'):
    a = tf.placeholder(tf.float32, [10000, 10000])
    b = tf.placeholder(tf.float32, [10000, 10000])
    c1.append(matpow(a, n))
    c1.append(matpow(b, n))

with tf.device('/cpu:0'):
  sum = tf.add_n(c1) #Addition of all elements in c1, i.e. A^n + B^n

t1_1 = datetime.datetime.now()
#sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
"""
with tf.Session(config=tf.ConfigProto\
              (log_device_placement=log_device_placement)) as sess:
     sess.run(sum, {a:A, b:B})
"""     
with tf.Session(config=tf.ConfigProto\
                (allow_soft_placement=True,\
                 log_device_placement=True)) as sess:
    sess.run(sum, {a:A, b:B})
t2_1 = datetime.datetime.now()
print("GPU computation time: " + str(t2_1-t1_1))
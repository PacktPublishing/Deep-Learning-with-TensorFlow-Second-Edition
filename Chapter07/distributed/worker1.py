import tensorflow as tf
from main import *

with tf.Session(worker1.target) as sess:
    init = tf.global_variables_initializer()
    a = tf.constant(10.0, dtype=tf.float32)
    add_node = tf.multiply(a,b)
    sess.run(init)
    a = add_node
    print(sess.run(add_node))


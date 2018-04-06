import tensorflow as tf
from main import *

with tf.Session(worker0.target) as sess:
    init = tf.global_variables_initializer()
    add_node = tf.multiply(a,b)
    sess.run(init)
    print(sess.run(add_node))

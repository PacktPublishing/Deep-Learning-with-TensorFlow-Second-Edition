import tensorflow as tf
import numpy as np

array_2d = np.array([(1,2,3),(4,5,6),(7,8,9)])
print(array_2d.shape[0])
#tensor_2d = tf.Variable(array_2d)
tensor_2d = tf.get_variable(array_2d, shape=[array_2d.shape[0], array_2d.shape[0]], dtype=tf.int32, initializer=None, regularizer=None, trainable=True, collections=None)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(tensor_2d.get_shape())
    print(sess.run(tensor_2d))
# Finally, close the TensorFlow session when you're done
sess.close()


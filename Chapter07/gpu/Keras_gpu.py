import tensorflow as tf
import numpy as np
import os
import sys
import datetime
from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import Dense, Reshape, Convolution2D, Activation, MaxPooling2D
from keras.optimizers import *




my_dir= os.getenv
print(my_dir)

mnist = input_data.read_data_sets('dataset/mnist', one_hot=True)
trainimg    = mnist.train.images
trainlabel  = mnist.train.labels
testimg     = mnist.test.images
testlabel   = mnist.test.labels

learning_rate = 0.001
training_epochs = 2
batch_size = 100
display_step = 1


def do_train(device):
    if device == 'gpu':
        device_type = '/gpu:0'
    else:
        device_type = '/cpu:0'
        
    with tf.device(device_type): # <= This is optional
        n_input  = 784
        n_output = 10
        weights  = {
            'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1)),
            'wd1': tf.Variable(tf.random_normal([14*14*64, n_output], stddev=0.1))
        }
        biases   = {
            'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),
            'bd1': tf.Variable(tf.random_normal([n_output], stddev=0.1))
        }
        def conv_simple(_input, _w, _b):
            # Reshape input
            _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])
            # Convolution
            _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
            # Add-bias
            _conv2 = tf.nn.bias_add(_conv1, _b['bc1'])
            # Pass ReLu
            _conv3 = tf.nn.relu(_conv2)
            # Max-pooling
            _pool  = tf.nn.max_pool(_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # Vectorize
            _dense = tf.reshape(_pool, [-1, _w['wd1'].get_shape().as_list()[0]])
            # Fully-connected layer
            _out = tf.add(tf.matmul(_dense, _w['wd1']), _b['bd1'])
            # Return everything
            out = {
                'input_r': _input_r, 'conv1': _conv1, 'conv2': _conv2, 'conv3': _conv3
                , 'pool': _pool, 'dense': _dense, 'out': _out
            }
            return out

        def conv_keras(_input):
            # Reshape input
            _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])
            # Convolution2D(nb_filters, kernal_size[0], kernal_size[1])
            _conv1 = Convolution2D(64,3,3, border_mode='same', input_shape=(28,28,1))(_input_r)
            _relu1 = Activation('relu')(_conv1)
            _pool1 = MaxPooling2D(pool_size=(2,2))(_relu1)
            # Conv layer 2
            _conv2 = Convolution2D(64,3,3, border_mode='same')(_pool1)
            _relu2 = Activation('relu')(_conv2)
            _pool2 = MaxPooling2D(pool_size=(2,2))(_relu2)
            # FC layer 1
            _dense1 = tf.reshape(_pool2, [-1, np.prod(_pool2.get_shape()[1:].as_list())])
            _dense2 = Dense(128, activation='relu')(_dense1)
            preds = Dense(10, activation='softmax')(_dense2)
            return preds

    print ("CNN ready with {}".format(device_type))

    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_output])
   
    with tf.device(device_type):
        _pred = conv_keras(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=_pred))
        optm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        _corr = tf.equal(tf.argmax(_pred,1), tf.argmax(y,1)) # Count corrects
        accr = tf.reduce_mean(tf.cast(_corr, tf.float32)) # Accuracy
        init = tf.global_variables_initializer()
    print ("Network Ready to Go!")
    
    do_train = 1
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(init)
    
    start_time = datetime.datetime.now()
    if do_train == 1:
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # Fit training using batch data
                sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
                # Compute average loss
                avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch

            # Display logs per epoch step
            if epoch % display_step == 0: 
                print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
                train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys})
                print (" Training accuracy: %.3f" % (train_acc))
                test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel})
                print (" Test accuracy: %.3f" % (test_acc))

            # Save Net
#             if epoch % save_step == 0:
#                 saver.save(sess, "nets/cnn_mnist_simple.ckpt-" + str(epoch))
        print ("Optimization Finished.")
        print ("Single {} computaion time : {}".format(device, datetime.datetime.now() - start_time))

        do_train('gpu')

        
do_train('cpu')

import tensorflow as tf
import numpy as np
from datetime import datetime
import EmotionUtils
import os, sys, inspect
from tensorflow.python.framework import ops
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

"""
lib_path = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
"""

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir",\
                       "EmotionDetector/",\
                       "Path to data files")
tf.flags.DEFINE_string("logs_dir",\
                       "logs/EmotionDetector_logs/",\
                       "Path to where log files are to be saved")
tf.flags.DEFINE_string("mode",\
                       "train",\
                       "mode: train (Default)/ test")

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_ITERATIONS = 1001
REGULARIZATION = 1e-2
IMAGE_SIZE = 48
NUM_LABELS = 7
VALIDATION_PERCENT = 0.1

def add_to_regularization_loss(W, b):
    tf.add_to_collection("losses", tf.nn.l2_loss(W))
    tf.add_to_collection("losses", tf.nn.l2_loss(b))

def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], \
                          strides=[1, 2, 2, 1], padding="SAME")

def emotion_cnn(dataset):
    with tf.name_scope("conv1") as scope:
        #W_conv1 = weight_variable([5, 5, 1, 32])
        #b_conv1 = bias_variable([32])
        tf.summary.histogram("W_conv1", weights['wc1'])
        tf.summary.histogram("b_conv1", biases['bc1'])
        conv_1 = tf.nn.conv2d(dataset, weights['wc1'],\
                              strides=[1, 1, 1, 1],\
                              padding="SAME")
        h_conv1 = tf.nn.bias_add(conv_1, biases['bc1'])
        #h_conv1 = conv2d_basic(dataset, W_conv1, b_conv1)
        h_1 = tf.nn.relu(h_conv1)
        h_pool1 = max_pool_2x2(h_1)
        add_to_regularization_loss(weights['wc1'], biases['bc1'])

    with tf.name_scope("conv2") as scope:
        #W_conv2 = weight_variable([3, 3, 32, 64])
        #b_conv2 = bias_variable([64])
        tf.summary.histogram("W_conv2", weights['wc2'])
        tf.summary.histogram("b_conv2", biases['bc2'])
        conv_2 = tf.nn.conv2d(h_pool1, weights['wc2'],\
                              strides=[1, 1, 1, 1],\
                              padding="SAME")
        h_conv2 = tf.nn.bias_add(conv_2, biases['bc2'])
        #h_conv2 = conv2d_basic(h_pool1, weights['wc2'], biases['bc2'])
        h_2 = tf.nn.relu(h_conv2)
        h_pool2 = max_pool_2x2(h_2)
        add_to_regularization_loss(weights['wc2'], biases['bc2'])

    with tf.name_scope("fc_1") as scope:
        prob=0.5
        image_size = IMAGE_SIZE // 4
        h_flat = tf.reshape(h_pool2, [-1, image_size * image_size * 64])
        #W_fc1 = weight_variable([image_size * image_size * 64, 256])
        #b_fc1 = bias_variable([256])
        tf.summary.histogram("W_fc1", weights['wf1'])
        tf.summary.histogram("b_fc1", biases['bf1'])
        h_fc1 = tf.nn.relu(tf.matmul(h_flat, weights['wf1']) + biases['bf1'])
        h_fc1_dropout = tf.nn.dropout(h_fc1, prob)
        
    with tf.name_scope("fc_2") as scope:
        #W_fc2 = weight_variable([256, NUM_LABELS])
        #b_fc2 = bias_variable([NUM_LABELS])
        tf.summary.histogram("W_fc2", weights['wf2'])
        tf.summary.histogram("b_fc2", biases['bf2'])
        #pred = tf.matmul(h_fc1, weights['wf2']) + biases['bf2']
        pred = tf.matmul(h_fc1_dropout, weights['wf2']) +\
               biases['bf2']

    return pred

weights = {
    'wc1': weight_variable([5, 5, 1, 32], name="W_conv1"),
    'wc2': weight_variable([3, 3, 32, 64],name="W_conv2"),
    'wf1': weight_variable([(IMAGE_SIZE//4)*(IMAGE_SIZE//4)*64,256],name="W_fc1"),
    'wf2': weight_variable([256, NUM_LABELS], name="W_fc2")
}

biases = {
    'bc1': bias_variable([32], name="b_conv1"),
    'bc2': bias_variable([64], name="b_conv2"),
    'bf1': bias_variable([256], name="b_fc1"),
    'bf2': bias_variable([NUM_LABELS], name="b_fc2")
}

def loss(pred, label):
    cross_entropy_loss =\
    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2\
                   (logits=pred, labels=label))
    tf.summary.scalar('Entropy', cross_entropy_loss)
    reg_losses = tf.add_n(tf.get_collection("losses"))
    tf.summary.scalar('Reg_loss', reg_losses)
    return cross_entropy_loss + REGULARIZATION * reg_losses

def get_next_batch(images, labels, step):
    offset = (step * BATCH_SIZE) % (images.shape[0] - BATCH_SIZE)
    batch_images = images[offset: offset + BATCH_SIZE]
    batch_labels = labels[offset:offset + BATCH_SIZE]
    return batch_images, batch_labels

def main(argv=None):
    train_images,train_labels,valid_images,valid_labels,test_images = EmotionUtils.read_data(FLAGS.data_dir)
    print("Train size: %s" % train_images.shape[0])
    print('Validation size: %s' % valid_images.shape[0])
    print("Test size: %s" % test_images.shape[0])

    global_step = tf.Variable(0, trainable=False)
    dropout_prob = tf.placeholder(tf.float32)
    input_dataset = tf.placeholder(tf.float32, \
                                   [None, IMAGE_SIZE, IMAGE_SIZE, 1],name="input")
    input_labels = tf.placeholder(tf.float32, [None, NUM_LABELS])

    pred = emotion_cnn(input_dataset)
    output_pred = tf.nn.softmax(pred,name="output")
    loss_val = loss(pred, input_labels)
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_val, global_step)

    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph) 
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model Restored!")

        for step in range(MAX_ITERATIONS):
            batch_image, batch_label = get_next_batch(train_images,\
                                                      train_labels, step)
            feed_dict = {input_dataset: batch_image, \
                         input_labels: batch_label}

            sess.run(train_op, feed_dict=feed_dict)
            if step % 10 == 0:
                train_loss, summary_str = sess.run([loss_val, summary_op],\
                                                   feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=step)
                print("Training Loss: %f" % train_loss)

            if step % 100 == 0:
                valid_loss = sess.run(loss_val, \
                                      feed_dict={input_dataset: valid_images,\
                                                 input_labels: valid_labels})
                print("%s Validation Loss: %f" % (datetime.now(), valid_loss))
                saver.save(sess, FLAGS.logs_dir + 'model.ckpt', global_step=step)


if __name__ == "__main__":
    tf.app.run()



"""
Reading train.csv ...
(4178, 48, 48, 1)
(4178, 7)
Reading test.csv ...
Picking ...
Train size: 3761
Validation size: 417
Test size: 1312
Training Loss: 1.958807
2018-02-24 15:17:45.421344 Validation Loss: 1.962773
Training Loss: 1.916384
Training Loss: 1.874157
Training Loss: 1.860803
Training Loss: 1.796846
Training Loss: 1.872398
Training Loss: 1.854985
Training Loss: 1.834055
Training Loss: 1.921772
Training Loss: 1.802405
Training Loss: 1.844588
2018-02-24 15:19:09.568140 Validation Loss: 1.796418
Training Loss: 1.765655
Training Loss: 1.639217
Training Loss: 1.662861
Training Loss: 1.628877
Training Loss: 1.527493
Training Loss: 1.550197
Training Loss: 1.361452
Training Loss: 1.552609
Training Loss: 1.416391
Training Loss: 1.383318
2018-02-24 15:20:35.122450 Validation Loss: 1.328313
Training Loss: 1.209850
Training Loss: 1.258240
Training Loss: 1.246996
Training Loss: 1.251378
Training Loss: 1.011271
Training Loss: 1.118353
Training Loss: 1.224426
Training Loss: 1.162503
Training Loss: 0.954584
Training Loss: 1.154842
2018-02-24 15:21:58.200816 Validation Loss: 1.120482
Training Loss: 1.142614
Training Loss: 1.017526
Training Loss: 1.016837
Training Loss: 0.916659
Training Loss: 1.112834
Training Loss: 0.831739
Training Loss: 0.961953
Training Loss: 0.959742
Training Loss: 0.870184
Training Loss: 0.774641
2018-02-24 15:23:24.024985 Validation Loss: 1.066049
Training Loss: 0.826836
Training Loss: 0.781013
Training Loss: 0.828705
Training Loss: 0.986543
Training Loss: 0.917876
Training Loss: 0.804076
Training Loss: 0.896962
Training Loss: 0.849673
Training Loss: 0.740196
Training Loss: 0.791801
2018-02-24 15:24:38.838554 Validation Loss: 0.965881
Training Loss: 0.660149
Training Loss: 0.817802
Training Loss: 0.759723
Training Loss: 0.778251
Training Loss: 0.820222
Training Loss: 0.778601
Training Loss: 0.732990
Training Loss: 0.799039
Training Loss: 0.676878
Training Loss: 0.667507
2018-02-24 15:25:54.761599 Validation Loss: 0.953470
Training Loss: 0.778299
Training Loss: 0.712712
Training Loss: 0.692834
Training Loss: 0.646735
Training Loss: 0.721945
Training Loss: 0.635004
Training Loss: 0.591936
Training Loss: 0.675401
Training Loss: 0.562353
Training Loss: 0.609804
2018-02-24 15:27:15.592093 Validation Loss: 0.897236
Training Loss: 0.705211
Training Loss: 0.639610
Training Loss: 0.583716
Training Loss: 0.564106
Training Loss: 0.582983
Training Loss: 0.602146
Training Loss: 0.698023
Training Loss: 0.599001
Training Loss: 0.605951
Training Loss: 0.652705
2018-02-24 15:28:39.881676 Validation Loss: 0.838831
Training Loss: 0.724818
Training Loss: 0.593321
Training Loss: 0.515698
Training Loss: 0.546896
Training Loss: 0.724818
Training Loss: 0.583093
Training Loss: 0.589566
Training Loss: 0.646185
Training Loss: 0.529777
Training Loss: 0.470442
2018-02-24 15:29:53.012461 Validation Loss: 0.910777
Training Loss: 0.495638
Training Loss: 0.491909
Training Loss: 0.611700
Training Loss: 0.533136
Training Loss: 0.552972
Training Loss: 0.566998
Training Loss: 0.555445
Training Loss: 0.531941
Training Loss: 0.537725
Training Loss: 0.437258
2018-02-24 15:31:14.416664 Validation Loss: 0.888537
>>> 
"""

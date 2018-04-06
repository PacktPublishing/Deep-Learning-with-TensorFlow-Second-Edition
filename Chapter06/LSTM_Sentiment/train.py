from data_preparation import Preprocessing
from lstm_network import LSTM_RNN_Network
import pickle
import datetime
import time
import matplotlib.pyplot as plt

import tensorflow as tf
import os
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import warnings

flags = tf.app.flags
FLAGS = flags.FLAGS  

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

data_dir = 'data/' # Data directory containing 'data.csv' file with 'SentimentText' and 'Sentiment\'. Intermediate files will automatically be stored here'
stopwords_file = 'data/stopwords.txt' # Path to stopwords file. If stopwords_file is None, no stopwords will be used'
n_samples= None # Set n_samples=None to use the whole dataset

summaries_dir= 'logs/'# Directory where TensorFlow summaries will be stored'
batch_size = 100 #Batch size
train_steps = 1000 #Number of training steps
hidden_size= 75 # Hidden size of LSTM layer
embedding_size = 75 # Size of embeddings layer

#random_state = 0 # Random state used for data splitting. Default is 0
learning_rate = 0.01
test_size = 0.2
dropout_keep_prob = 0.5 # 0<dropout_keep_prob<=1. Dropout keep-probability
sequence_len = None # Maximum sequence length
validate_every = 100 # Step frequency in order to evaluate the model using a validation set'

# Prepare summaries
summaries_dir = '{0}/{1}'.format(summaries_dir, datetime.datetime.now().strftime('%d_%b_%Y-%H_%M_%S'))
train_writer = tf.summary.FileWriter(summaries_dir + '/train')
validation_writer = tf.summary.FileWriter(summaries_dir + '/validation')

# Prepare model directory
model_name = str(int(time.time()))
model_dir = '{0}/{1}'.format('checkpoints', model_name)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Prepare data and build TensorFlow graph
data_lstm = Preprocessing(data_dir=data_dir,
                          stopwords_file=stopwords_file,
                          sequence_len=sequence_len,
                          test_size=test_size,
                          val_samples=batch_size,
                          n_samples=n_samples,
                          random_state=100)

lstm_model = LSTM_RNN_Network(hidden_size=[hidden_size],
                              vocab_size=data_lstm.vocab_size,
                              embedding_size=embedding_size,
                              max_length=data_lstm.sequence_len,
                              learning_rate=learning_rate)

#   Train model
sess = tf.Session()
# Initializing the variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

saver = tf.train.Saver()
x_val, y_val, val_seq_len = data_lstm.get_val_data()
train_writer.add_graph(lstm_model.input.graph)

# Lists
train_loss_list = []
val_loss_list = []
step_list = []
sub_step_list = []
step = 0

for i in range(train_steps):
    # Perform training step
    x_train, y_train, train_seq_len = data_lstm.next_batch(batch_size)
    train_loss, _, summary = sess.run([lstm_model.loss, lstm_model.train_step, lstm_model.merged],
                                      feed_dict={lstm_model.input: x_train,
                                                 lstm_model.target: y_train,
                                                 lstm_model.seq_len: train_seq_len,
                                                 lstm_model.dropout_keep_prob: dropout_keep_prob})
    train_writer.add_summary(summary, i)  # Write train summary for step i (TensorBoard visualization)
    train_loss_list.append(train_loss)
    step_list.append(i)

    print('{0}/{1} train loss: {2:.4f}'.format(i + 1, train_steps, train_loss))

    # Check validation performance
    if (i + 1) % validate_every == 0:
        val_loss, accuracy, summary = sess.run([lstm_model.loss, lstm_model.accuracy, lstm_model.merged],
                                               feed_dict={lstm_model.input: x_val,
                                                          lstm_model.target: y_val,
                                                          lstm_model.seq_len: val_seq_len,
                                                          lstm_model.dropout_keep_prob: 1})
        validation_writer.add_summary(summary, i)  # Write validation summary for step i (TensorBoard visualization)
        print('   validation loss: {0:.4f} (accuracy {1:.4f})'.format(val_loss, accuracy))
        step = step + 1
        val_loss_list.append(val_loss)
        sub_step_list.append(step)

# Plot loss over time
plt.plot(step_list, train_loss_list, 'r--', label='LSTM training loss per iteration', linewidth=4)
plt.title('LSTM training loss per iteration')
plt.xlabel('Iteration')
plt.ylabel('Training loss')
plt.legend(loc='upper right')
plt.show()

# Plot accuracy over time
plt.plot(sub_step_list, val_loss_list, 'r--', label='LSTM validation loss per validatin interval', linewidth=4)
plt.title('LSTM validation loss per validatin interval')
plt.xlabel('Validatin interval')
plt.ylabel('Validation loss')
plt.legend(loc='upper left')
plt.show()

# Save model
checkpoint_file = '{}/model.ckpt'.format(model_dir)
save_path = saver.save(sess, checkpoint_file)
print('Model saved in: {0}'.format(model_dir))

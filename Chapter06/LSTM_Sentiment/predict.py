from data_preparation import Preprocessing
import pickle

import tensorflow as tf
import os
from tensorflow.python.framework import ops
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

checkpoints_dir = 'checkpoints/1519576475' # Change this path based on the output from '$ python3 train.py' script "Model saved in: checkpoints/1517781236"

data_dir = 'data/' # Data directory containing 'data.csv' file with 'SentimentText' and 'Sentiment\'. Intermediate files will automatically be stored here'
stopwords_file = 'data/stopwords.txt' # Path to stopwords file. If stopwords_file is None, no stopwords will be used'
sequence_len = None # Maximum sequence length
n_samples= None # Set n_samples=None to use the whole dataset
test_size = 0.2
batch_size = 100 #Batch size
random_state = 0 # Random state used for data splitting. Default is 0

if checkpoints_dir is None:
    raise ValueError('Please, a valid checkpoints directory is required (--checkpoints_dir <file name>)')

# Load data
data_lstm = Preprocessing(data_dir=data_dir,
                 stopwords_file=stopwords_file,
                 sequence_len=sequence_len,
                 n_samples=n_samples,
                 test_size=test_size,
                 val_samples=batch_size,
                 random_state=random_state,
                 ensure_preprocessed=True)

# Import graph and evaluate the model using test data
original_text, x_test, y_test, test_seq_len = data_lstm.get_test_data(original_text=True)
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()

    # Import graph and restore its weights
    print('Restoring graph ...')
    saver = tf.train.import_meta_graph("{}/model.ckpt.meta".format(checkpoints_dir))
    saver.restore(sess, ("{}/model.ckpt".format(checkpoints_dir)))

    # Recover input/output tensors
    input = graph.get_operation_by_name('input').outputs[0]
    target = graph.get_operation_by_name('target').outputs[0]
    seq_len = graph.get_operation_by_name('lengths').outputs[0]
    dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
    predict = graph.get_operation_by_name('final_layer/softmax/predictions').outputs[0]
    accuracy = graph.get_operation_by_name('accuracy/accuracy').outputs[0]

    # Perform prediction
    pred, acc = sess.run([predict, accuracy],
                         feed_dict={input: x_test,
                                    target: y_test,
                                    seq_len: test_seq_len,
                                    dropout_keep_prob: 1})
    print("Evaluation done.")

# Print results
print('\nAccuracy: {0:.4f}\n'.format(acc))
for i in range(100):
    print('Sample: {0}'.format(original_text[i]))
    print('Predicted sentiment: [{0:.4f}, {1:.4f}]'.format(pred[i, 0], pred[i, 1]))
    print('Real sentiment: {0}\n'.format(y_test[i]))

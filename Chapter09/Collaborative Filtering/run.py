# Imports for data io operations
from collections import deque
from six import next
import readers
import os
# Main imports for training
import tensorflow as tf
import numpy as np
import model as md
# Evaluate train times per epoch
import time

# Constant seed for replicating training results
np.random.seed(12345)

# Parameters
# ==================================================

# Eval Parameters
user = 1696 # User (default: 1696)
item = 3113, # Movie (default: 3113)
checkpoint_dir = "save/" # Checkpoint directory from training run
is_gpu = True # Want to train model at GPU

# Misc Parameters
allow_soft_placement = True # Allow device soft device placement
log_device_placement = False # Log placement of ops on devices

# Device used for all computations
if is_gpu:
    place_device = "/gpu:0"
else:
    place_device="/cpu:0"

# Clip (limit) the values in an array: given an interval, values outside the interval are clipped to the interval edges. 
#For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.

def clip(x):
    return np.clip(x, 1.0, 5.0) # rating 1 to 5

# Inference using saved model
user=np.array([user])
item=np.array([item])

#Making predicitons
def prediction(users=user,items=item,allow_soft_placement=allow_soft_placement,log_device_placement=log_device_placement,checkpoint_dir=checkpoint_dir):
    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    rating_prediction=[]
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=allow_soft_placement,log_device_placement=log_device_placement)
        with tf.Session(config = session_conf) as sess:
            new_saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_prefix))
            new_saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

            # Get the placeholders from the graph by name
            user_batch = graph.get_operation_by_name("id_user").outputs[0]
            item_batch = graph.get_operation_by_name("id_item").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("svd_inference").outputs[0]
            pred = sess.run(predictions, feed_dict={user_batch: users, item_batch: items})
            pred = clip(pred)
            #print(pred)
        sess.close()
    print(pred)
    return  pred

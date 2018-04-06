import tensorflow as tf

class DQNetwork:
    def __init__(self,\
                 learning_rate=0.01, \
                 state_size=4,\
                 action_size=2, \
                 hidden_size=10,\
                 name='DQNetwork'):

         with tf.variable_scope(name):
            self.inputs_ = \
                         tf.placeholder\
                         (tf.float32,\
                          [None, state_size],\
                          name='inputs')
            
            self.actions_ = tf.placeholder\
                            (tf.int32,[None],\
                             name='actions')
            
            one_hot_actions =tf.one_hot\
                              (self.actions_,\
                               action_size)
            
            self.targetQs_ = tf.placeholder\
                             (tf.float32,[None],\
                              name='target')
            
            self.fc1 =tf.contrib.layers.fully_connected\
                       (self.inputs_,\
                        hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected\
                       (self.fc1,\
                        hidden_size)

            self.output = tf.contrib.layers.fully_connected\
                          (self.fc2,\
                           action_size,activation_fn=None)

            self.Q = tf.reduce_sum(tf.multiply\
                                   (self.output,\
                                    one_hot_actions),\
                                   axis=1)
            
            self.loss = tf.reduce_mean\
                        (tf.square(self.targetQs_ - self.Q))
            
            self.opt = tf.train.AdamOptimizer\
                       (learning_rate).minimize(self.loss)

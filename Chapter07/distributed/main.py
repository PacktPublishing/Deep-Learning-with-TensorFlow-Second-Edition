import tensorflow as tf

cluster = tf.train.ClusterSpec({"ps": ["localhost:2222"],\
                                "worker": ["localhost:2223",\
                                           "localhost:2224"]})

ps = tf.train.Server(cluster, job_name="ps", task_index=0)

worker0 = tf.train.Server(cluster,\
                          job_name="worker",\
                          task_index=0)

worker1 = tf.train.Server(cluster,\
                          job_name="worker",\
                          task_index=1)

with tf.device("/job:ps/task:0"):
    a = tf.constant(3.0, dtype=tf.float32)
    b = tf.constant(4.0)  



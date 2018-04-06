import tensorflow as tf
import gym
import numpy as np
import time
import os
from create_cart_pole_env import *
from DQNetwork import *
from memory import *
from solve_cart_pole import *
from plot_result_DQN import *




# Create the Cart-Pole game environment
env = gym.make('CartPole-v0')
create_cart_pole_env(env)


#Build the deep neural network
tf.reset_default_graph()
deepQN = DQNetwork(name='main', \
                  hidden_size=64, \
                  learning_rate=0.0001)

# Initialize the simulation
env.reset()

# Take one random step to get the pole and cart moving
state, rew, done, _ = env.step(env.action_space.sample())
memory = Memory(max_size=10000)

# Make a bunch of random actions and store the experiences
pretrain_length= 20

for j in range(pretrain_length):
    action = env.action_space.sample()
    next_state, rew, done, _ = \
                env.step(env.action_space.sample())
    if done:
        env.reset()
        memory.build((state, action, rew, np.zeros(state.shape)))
        state, rew, done, _ = \
               env.step(env.action_space.sample())
    else:
        memory.build((state, action, rew, next_state))
        state = next_state
        

# Exploration parameters
# exploration probability at start
start_exp = 1.0
# minimum exploration probability 
stop_exp = 0.01
# exponential decay rate for exploration prob
decay_rate = 0.0001            

# Train the DQN with new experiences
rew_list = []
train_episodes = 100
max_steps=200
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    for ep in range(1, train_episodes):
        tot_rew = 0
        t = 0
        while t < max_steps:
            step += 1
            explore_p = stop_exp + \
                        (start_exp - stop_exp)*\
                        np.exp(-decay_rate*step)
            
            if explore_p > np.random.rand():
                action = env.action_space.sample()
                
            else:
                Qs = sess.run(deepQN.output, \
                              feed_dict={deepQN.inputs_: \
                                         state.reshape\
                                         ((1, *state.shape))})
                action = np.argmax(Qs)

            next_state, rew, done, _ = env.step(action)
            tot_rew += rew
            
            if done:
                next_state = np.zeros(state.shape)
                t = max_steps
               
                print('Episode: {}'.format(ep),
                      'Total rew: {}'.format(tot_rew),
                      'Training loss: {:.4f}'.format(loss),
                      'Explore P: {:.4f}'.format(explore_p))
                
                rew_list.append((ep, tot_rew))
                memory.build((state, action, rew, next_state))
                env.reset()
                state, rew, done, _ = env.step\
                                         (env.action_space.sample())

            else:
                memory.build((state, action, rew, next_state))
                state = next_state
                t += 1

            batch_size = pretrain_length               
            states = np.array([item[0] for item in memory.sample(batch_size)])
            actions = np.array([item[1] for item in memory.sample(batch_size)])
            rews = np.array([item[2] for item in memory.sample(batch_size)])
            next_states = np.array([item[3] for item in memory.sample(batch_size)])
            
            target_Qs = sess.run(deepQN.output, \
                                 feed_dict=\
                                 {deepQN.inputs_: next_states})

            target_Qs[(next_states == \
                       np.zeros(states[0].shape))\
                      .all(axis=1)] = (0, 0)
            
            targets = rews + 0.99 * np.max(target_Qs, axis=1)

            loss, _ = sess.run([deepQN.loss, deepQN.opt],
                                feed_dict={deepQN.inputs_: states,
                                           deepQN.targetQs_: targets,
                                           deepQN.actions_: actions})

    env = gym.make('CartPole-v0')
    solve_cart_pole(env,deepQN,state,sess)
    plot_result(rew_list)
    



import numpy as np

def solve_cart_pole(env,dQN,state,sess):
    test_episodes = 10
    test_max_steps = 400
    env.reset()
    for ep in range(1, test_episodes):
        t = 0
        while t < test_max_steps:
            env.render() 
        
            # Get action from Q-network
            Qs = sess.run(dQN.output, \
                          feed_dict={dQN.inputs_: state.reshape\
                                     ((1, *state.shape))})
            action = np.argmax(Qs)
        
            # Take action, get new state and reward
            next_state, reward, done, _ = env.step(action)
        
            if done:
                t = test_max_steps
                env.reset()
                # Take one random step to get the pole and cart moving
                state, reward, done, _ = env.step(env.action_space.sample())

            else:
                state = next_state
                t += 1     

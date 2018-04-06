import gym
import time

#Run the cart-pole 
def create_cart_pole_env(env):
    env.reset()
    rewards = []
    tic = time.time()
    for _ in range(1000):
        env.render()
        state, reward, done, info = env.step(env.action_space.sample())
        rewards.append(reward)
        if done:
            rewards = []
            env.reset()
    toc = time.time()
    if toc-tic > 10:
        env.close()
    print(rewards)

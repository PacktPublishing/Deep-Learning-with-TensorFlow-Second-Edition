import gym
env = gym.make('CartPole-v0')

obs = env.reset()

for step_idx in range(500):
  env.render()
  obs, reward, done, _ = env.step(env.action_space.sample())

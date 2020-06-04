import gym
import numpy as np

env_name = "CartPole-v0"
gym_spec = gym.spec(env_name)
print("About to MAKE")
gym_env = gym_spec.make()

env = gym_env
experiences = 2
#experiences = 1
actions = 100

total_total_length = []
total_total_reward = []
a = 0
for i in range(experiences):
    obs = env.reset()
    total_reward = 0
    max_job = 0
    for j in range(actions):
        action = env.action_space.sample()  # direct action for test
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    total_total_length.append(j + 1)
    total_total_reward.append(total_reward)
print("average_total_reward", float(sum(total_total_reward)) / experiences)
print("average episode length", float(sum(total_total_length)) / experiences)
print("total actions", sum(total_total_length))
print("std reward", np.std(np.array(total_total_reward)))
print("std episode length", np.std(np.array(total_total_length)))
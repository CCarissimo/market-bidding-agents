import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ddpg import DDPGagent
from utils import *

env = gym.make("Pendulum-v0")  # not being used
observation_space_size = 2
action_space_size = 1
n_players = 100

done = False
episodes = 100
len_episode = 1000

agent = DDPGagent(env)  # env is redundant, but hacked

karma = np.ones(n_players)/n_players
urgency = np.random.random_sample(size=n_players)

noise = OUNoise(env.action_space, min_sigma=0, decay_period=int(episodes*len_episode))
batch_size = 128
rewards = []
avg_rewards = []
actions = []

state = np.array([karma[0], urgency[0]])  # env.reset()
noise.reset()
episode_reward = 0

for t in range(int(episodes*len_episode)):
    action = agent.get_action(state)
    action = noise.get_action(action, t)

    random_player = np.random.randint(0, n_players)
    state2 = np.array([karma[random_player], urgency[random_player]])
    action2 = agent.get_action(state2)
    action2 = noise.get_action(action2, t)

    # pay bid to peer PBP
    if action >= action2:
        reward = 1 * urgency[0]
        r2 = -urgency[random_player]
        karma[random_player] += action * karma[0]
        karma[0] -= action*karma[0]

    elif action < action2:
        reward = -urgency[0]
        r2 = 1 * urgency[random_player]
        karma[0] += action2 * karma[random_player]
        karma[random_player] -= action * karma[random_player]

    new_random_urgency = np.random.random()
    new_random_urgency_2 = np.random.random()
    urgency[0] = new_random_urgency
    urgency[random_player] = new_random_urgency_2
    new_state = np.array([karma[0], new_random_urgency])
    new_state_2 = np.array([karma[random_player], new_random_urgency_2])

    # new_state, reward, done, _ = env.step(action)
    agent.memory.push(state, action, reward, new_state, done)
    agent.memory.push(state2, action2, r2, new_state_2, done)

    if len(agent.memory) > batch_size:
        agent.update(batch_size)

    state = new_state
    episode_reward += reward

    actions.append((action[0], action2[0]))

    if t % len_episode == 0:
        done = True
    if done:
        sys.stdout.write(
            "episode: {}, reward: {}, average _reward: {} \n".format(t/1000, np.round(episode_reward, decimals=2),
                                                                     np.mean(rewards[-10:])))
        done = False
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))
        episode_reward = 0

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

plt.plot(actions, label=["ego_player", "random_player"])
plt.legend()
plt.show()

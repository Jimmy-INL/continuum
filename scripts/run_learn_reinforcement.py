from typing import NamedTuple
from loguru import logger
import numpy as np
import gym
import random
import time
import torch
from torch import nn
from torch.nn import functional as F

env = gym.make("FrozenLake-v0")

action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.zeros((action_space_size, state_space_size))

num_episodes = 10000

max_steps_per_episode = 100
learning_rate = 0.1

exploration_rate = 1
max_exploration_rate = 1
discount_rate = 0.99
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

rewards_all_episodes: list = []


class DQN(nn.Module):

    def __init__(self, img_height, img_width) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=img_height * img_width, out_features=24
        )
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)

    def forward(self, x: torch.Tensor):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


Experience = NamedTuple(
    'Experience', ('state', 'action', 'next_state', 'reward')
)


def run():
    pass
    # for episode in range(num_episodes):

    #     state = env.reset()
    #     done = False
    #     rewards_current_episode = 0
    #     # print("CUNT")
    #     for step in range(max_steps_per_episode):
    #         exploration_rate_threshold = random.uniform(0, 1)
    #         if exploration_rate_threshold > exploration_rate:
    #             action = np.argmax(q_table[state, :])
    #         else:
    #             action = env.action_space.sample()
    #         new_state, reward, done, info = env.step(action)
    #         q_table[action, state] = q_table[action, state] * (1 - learning_rate) + \
    #             learning_rate * (reward + discount_rate * np.max(q_table[:, new_state]))
    #         logger.info(action)
    #         logger.success(reward)
    #         state = new_state
    #         rewards_current_episode += reward
    #         if done:
    #             break


if __name__ == "__main__":
    run()
    # print(q_table)
    # print(rewards_all_episodes)
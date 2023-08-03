import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import namedtuple, deque
import random
import math


class PolicyModel(nn.Module):
    def __init__(self, n_states, n_actions):
        super(PolicyModel, self).__init__()
        self.layer1 = nn.Linear(n_states, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return F.log_softmax(self.layer3(x), dim=1)


class ValueModel(nn.Module):
    def __init__(self, n_states):
        super(ValueModel, self).__init__()
        self.layer1 = nn.Linear(n_states, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


lr_policy = 0.01
lr_value = 0.01
episode_num = 2000
discounting_factor = 0.99

env = gym.make('CartPole-v1', render_mode='human')
state, info = env.reset()
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

policy_net = PolicyModel(n_states, n_actions).to(device)
value_net = ValueModel(n_states).to(device)
policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr_policy)
value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr_value)
value_loss = nn.MSELoss()

loss_list = []
step_list = []
for i in range(episode_num):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).reshape(1, -1)
    state_list = []
    action_list = []
    reward_list = []
    with torch.no_grad():
        for t in count():
            state_list.append(state)
            pi = torch.exp(policy_net(state).reshape(-1))
            action = torch.multinomial(pi, num_samples=1)
            action_list.append(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            reward_list.append(torch.tensor([reward], dtype=torch.float32, device=device))
            if terminated or truncated:
                step_list.append(t + 1)
                break
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device).reshape(1, -1)
                state = next_state

    state_b = torch.cat(state_list, dim=0)
    action_b = torch.cat(action_list, dim=0).reshape(-1)
    return_b = torch.zeros(len(reward_list), device=device, dtype=torch.float32)
    temp = torch.zeros(1, device=device, dtype=torch.float32)
    for j in range(len(reward_list)-1, -1, -1):
        temp = reward_list[j] + discounting_factor * temp
        return_b[j] = temp

    pred_v = value_net(state_b).reshape(-1)
    l = value_loss(pred_v, return_b)
    value_optimizer.zero_grad()
    l.backward()
    value_optimizer.step()

    Adv = return_b - pred_v.detach()
    logpi = policy_net(state_b)
    policy_loss = - torch.mean(logpi[torch.arange(state_b.shape[0]), action_b] * Adv)
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    plt.figure(1)
    plt.clf()
    plt.plot(step_list)
    plt.xlabel('episode_num')
    plt.ylabel('step_num')
    plt.pause(0.001)

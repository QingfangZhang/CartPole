import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import namedtuple, deque
import random
import math


class Model(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(n_states, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class ExperienceDeque(object):
    def __init__(self, maxlen=10000):
        self.expdeque = deque([], maxlen=maxlen)

    def push(self, item):
        self.expdeque.append(item)

    def sample(self, batch_size):
        return random.sample(self.expdeque, batch_size)

    def __len__(self):
        return len(self.expdeque)


def select_action(state):
    global steps_done
    i = random.uniform(0, 1)  # uniformly return a random floating point in [0, 1]
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done/eps_decay)
    if i <= eps_threshold:
        action = torch.tensor([env.action_space.sample()], dtype=torch.int64, device=device)  # return a random int in [0, n_actions]
    else:
        with torch.no_grad():
            action = torch.max(update_net(state), dim=1)[1]
    steps_done += 1
    return action


def optimize_model():
    if len(experiences) < batch_size:
        return
    else:
        batch = experiences.sample(batch_size)
        b = Transition(*zip(*batch))
        b_state = torch.cat(b.state)
        b_action = torch.cat(b.action)
        b_reward = torch.cat(b.reward)
        mask = [i is not None for i in b.next_state]
        b_next_state = torch.cat([i for i in b.next_state if i is not None])

        update_net.train()
        yhat = update_net(b_state)[torch.arange(batch_size), b_action]
        y_temp = torch.zeros(batch_size, device=device)
        with torch.no_grad():
            y_temp[mask] = torch.max(target_net(b_next_state), dim=1)[0]
        y = b_reward + discounting_factor * y_temp
        l = loss(yhat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        loss_list.append(l.detach().cpu())
        target_net_state_dict = target_net.state_dict()
        update_net_state_dict = update_net.state_dict()
        for k in update_net_state_dict.keys():
            target_net_state_dict[k] = Tau * update_net_state_dict[k] + (1 - Tau) * target_net_state_dict[k]
        target_net.load_state_dict(target_net_state_dict)
        # draw_loss_figure(loss_list)


def draw_loss_figure(loss_list):
    plt.figure(2)
    plt.clf()
    plt.plot(loss_list)
    plt.xlabel('iter_num')
    plt.ylabel('loss')
    plt.pause(0.001)


lr = 0.0001
batch_size = 128
eps_start = 0.9
eps_end = 0.05
eps_decay = 1000
episode_num = 600
discounting_factor = 0.9
Tau = 0.005

env = gym.make('CartPole-v1', render_mode='rgb_array')
state, info = env.reset()
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

update_net = Model(n_states, n_actions).to(device)
target_net = Model(n_states, n_actions).to(device)
target_net.load_state_dict(update_net.state_dict())
loss = nn.SmoothL1Loss()
# loss = nn.MSELoss()
optimizer = torch.optim.Adam(update_net.parameters(), lr=lr)
# optimizer = torch.optim.AdamW(update_net.parameters(), lr=lr, amsgrad=True)

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])
experiences = ExperienceDeque(10000)
loss_list = []
step_list = []
steps_done = 0
for i in range(episode_num):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).reshape(1, -1)
    for t in count():
        action = select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action.item())
        reward = torch.tensor([reward], dtype=torch.float32, device=device)
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).reshape(1, -1)
        experiences.push(Transition._make([state, action, reward, next_state]))
        optimize_model()
        if terminated or truncated:
            break
        state = next_state
    step_list.append(t+1)

    plt.figure(1)
    plt.clf()
    plt.plot(step_list)
    plt.xlabel('episode_num')
    plt.ylabel('step_num')
    plt.pause(0.001)

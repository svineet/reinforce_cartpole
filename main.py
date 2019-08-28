import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import gym

import torch
from torch import nn
from torch.nn import functional as F

from collections import deque
from functools import reduce

DEVICE = "cpu"
device = torch.device(DEVICE)


class Policy(nn.Module):
    """
        Map state to action probabilities
    """
    def __init__(self, state_size, num_actions, h1=256, h2=100):
        super().__init__()

        self.fc1 = nn.Linear(state_size, h1)
        self.fc2 = nn.Linear(h1, num_actions)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.softmax(self.fc2(out), dim=1)

        return out


class Agent:
    def __init__(self, env, alpha=0.001,
                       gamma=0.90, lr=0.001):
        self.alpha = alpha
        self.gamma = gamma

        state_size = reduce(lambda x, y: x*y, env.observation_space.shape)
        self.num_actions = env.action_space.n
        self.policy = Policy(state_size, self.num_actions)

        global device
        self.policy.to(device=device)

        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=lr)

    def load_state(self, path):
        pass

    def get_action(self, state):
        """
            Returns (action, log (pi(a_t | s_t)))
        """
        global device
        st = torch.Tensor(state[None, :]).to(device=device)
        p = self.policy(st)
        p2 = p[0].detach().numpy() if DEVICE == "cpu" else p[0].cpu().detach().numpy()
        choice = np.random.choice(range(self.num_actions), p=p2)
        return (choice, p[0, choice].log())

    def start_episode(self):
        self.states = deque([])
        self.rewards = deque([])
        self.actions = deque([])
        self.logprobs = deque([])

    def step(self, state, action, reward, logprob):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)
        self.logprobs.append(logprob)

    def chain_train(self):
        """
            Takes a Monte Carlo episode and trains the policy
            on it

            chain = [ (state, action, reward, done, next_state), ... ]
        """
        discounted_rewards = deque([])
        cum_rew = 0
        full_rew = 0
        for reward in reversed(self.rewards):
            full_rew += reward
            cum_rew = reward + self.gamma*cum_rew
            discounted_rewards.appendleft(cum_rew)

        print("Full reward", full_rew)

        global device
        discounted_rewards = torch.Tensor(discounted_rewards).to(device=device)
        logprobs = torch.stack(tuple(self.logprobs)).to(device=device)
        loss = - torch.sum( discounted_rewards*logprobs )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self):
        torch.save(self.policy.state_dict(), "cartpole_policy_cpu.pkl")

    def load(self):
        if DEVICE == "cuda":
            self.policy.load_state_dict(torch.load("cartpole_policy.pkl"))
        else:
            self.policy.load_state_dict(
                    torch.load("cartpole_policy.pkl", map_location=lambda storage, loc: storage))


RENDER_ENV = True

def main(args):
    env = gym.make('CartPole-v0')
    agent = Agent(env)
    ema_reward = 0
    agent.load()
    for i in range(args["NUM_EPISODES"]):
        print("Starting episode", i)
        state = env.reset()
        done = False
        agent.start_episode()
        total = 0
        while not done:
            action, logprob = agent.get_action(state)
            new_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, logprob)

            if RENDER_ENV:
                env.render()

            state = new_state

        ema_reward = 0.9*ema_reward + 0.1*total
        agent.chain_train()

        if i%10==0:
            # agent.save()
            print("EMA of Reward is", ema_reward)

        if (ema_reward >= 195):
            print("We have arrived in the AI age")
            agent.save()
            # break


if __name__ == '__main__':
    main({
        "NUM_EPISODES": 100,
    })


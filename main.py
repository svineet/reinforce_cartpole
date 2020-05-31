import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import gym

import torch
from torch import nn
from torch.nn import functional as F

from collections import deque
from functools import reduce


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
    def __init__(self, env, gamma=0.99, lr=0.1,
                       device_name="cpu",
                       device=torch.device("cpu")):
        self.gamma = gamma
        self.device = device
        self.device_name = device_name

        state_size = reduce(lambda x, y: x*y, env.observation_space.shape)
        self.num_actions = env.action_space.n
        self.policy = Policy(state_size, self.num_actions)

        self.policy.to(device=device)

        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=lr)

    def load_state(self, path):
        pass

    def get_action(self, state):
        """
            Returns (action, log (pi(a_t | s_t)))
        """
        device = self.device

        st = torch.Tensor(state[None, :]).to(device=device)
        p = self.policy(st)
        p2 = p[0].detach().numpy() if self.device_name == "cpu" else p[0].cpu().detach().numpy()
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
        rewards = np.array(self.rewards)
        for reward in reversed(rewards):
            full_rew += reward
            cum_rew = reward + self.gamma*cum_rew
            discounted_rewards.appendleft(cum_rew)

        device = self.device
        discounted_rewards = torch.Tensor(discounted_rewards).to(device=device)
        logprobs = torch.stack(tuple(self.logprobs)).to(device=device)
        loss = - torch.sum( discounted_rewards*logprobs )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self):
        torch.save(self.policy.state_dict(), "cartpole_policy_cpu.pkl")

    def load(self):
        if self.device_name == "cuda":
            self.policy.load_state_dict(torch.load("cartpole_policy.pkl"))
        else:
            self.policy.load_state_dict(
                    torch.load("cartpole_policy.pkl", map_location=lambda storage, loc: storage))


def train(args):
    env = gym.make(args["ENVIRONMENT"])
    agent = Agent(env, lr=args["LEARNING_RATE"])
    ema_reward = 0

    stats = {
        "reward_ema": deque([])
    }

    if args["LOAD_POLICY"]: agent.load()

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
            total += reward

            if args["RENDER_ENV"]:
                env.render()

            state = new_state

        ema_reward = 0.9*ema_reward + 0.1*total
        agent.chain_train()

        if i%10==0:
            agent.save()
            stats["reward_ema"].append(ema_reward)
            print("EMA of Reward is", ema_reward)

        if args["STOP_AT_THRESHOLD"] and (ema_reward >= env.spec.reward_threshold):
            print("We have arrived in the AI age")
            agent.save()
            break

    return stats


if __name__ == '__main__':
    stats = train({
                    "NUM_EPISODES": 2000,
                    "LEARNING_RATE": 0.001,
                    "ENVIRONMENT": "CartPole-v0",
                    "DEVICE": "cpu",
                    "RENDER_ENV": False,
                    "LOAD_POLICY": False,
                    "STOP_AT_THRESHOLD": False
                })

    plt.plot(stats["reward_ema"])
    plt.show()


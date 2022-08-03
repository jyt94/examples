import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make('CartPole-v1')
env.seed(args.seed)
torch.manual_seed(args.seed)



class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)
        self.rewards = []
        self.logP = []


    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.affine2(x)
        # softmax convert float to probability
        return F.softmax(x, dim=1)  

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state) # run forward 
    m = Categorical(probs)
    action = m.sample()
    policy.logP.append(m.log_prob(action))
    return action.item() 



def finish_episode():
    R = 0
    end = len(policy.rewards) - 1
    loss = []
    returns = []
    for i in range(0, len(policy.rewards)):
        r = policy.rewards[end-i]
        R = R * args.gamma + r
        returns.insert(0, R)
    
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std())

    for i in range(0, len(returns)):
        loss.append(policy.logP[i] * returns[i])
    
    optimizer.zero_grad()
    loss = torch.cat(loss).sum()
    loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.logP[:]





def main():
    running_reward = 10
    for i_episode in count(1):
        
        # roll outs
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # update the model
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

if __name__ == '__main__':
    main()

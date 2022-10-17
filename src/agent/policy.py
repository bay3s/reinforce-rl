import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np


GAMMA = 0.99


class Policy(nn.Module):

  def __init__(self, in_features, out_features):
    super().__init__()

    layers = [
      nn.Linear(in_features=in_features, out_features=64, bias=True),
      nn.ReLU(),
      nn.Linear(64, out_features=out_features, bias=True)
    ]

    self.model = nn.Sequential(*layers)
    self.log_probs = list()
    self.rewards = list()

  def forward(self, state: torch.Tensor):
    return self.model(state)

  def on_policy_reset(self):
    self.log_probs = list()
    self.rewards = list()

  def act(self, state: torch.Tensor) -> int:
    """
    Take an action based on the state of the environment given to the agent.

    :param state: State of the environment for the agent.

    :return: int
    """
    pd_parameters = self.forward(state)

    """
    PyTorch provides parameterizable probability distributions and sampling functions.
    
    https://pytorch.org/docs/stable/distributions.html
    
    "Gradient Estimation Using Stochastic Computation Graphs", Abbeel et al. https://arxiv.org/abs/1506.05254
    
    - In a variety of ML problems, the loss function is defined as an expectation over a collection of random variables.
    - Estimating the gradient of the loss function, using samples lies at the core of gradient-based learning algorithms
      for such problems.
    """
    pd = Categorical(logits=pd_parameters) # probability distribution
    action = pd.sample() # pi(a|s)

    log_prob = pd.log_prob(action) # log probability of pi(a|s)
    self.log_probs.append(log_prob) # store these for training

    return action.item()

  def train(self, optimizer: optim):
    trajectory_length = len(self.rewards) # gradient ascent loop of REINFORCE

    returns = np.empty(trajectory_length, dtype=np.float32)
    future_returns = 0.0

    for t in reversed(range(trajectory_length)):
      future_returns = self.rewards[t] + GAMMA * future_returns
      returns[t] = future_returns

    returns = torch.tensor(returns)
    log_probs = torch.stack(self.log_probs)

    loss = -log_probs * returns
    loss = torch.sum(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


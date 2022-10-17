import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.optim as optim


class DiscretePolicy(nn.Module):

  def __init__(self, in_features:int, out_features:int, learning_rate: float = 0.99):
    super().__init__()

    layers = [
      nn.Linear(in_features=in_features, out_features=64, bias=True),
      nn.ReLU(),
      nn.Linear(64, out_features=out_features, bias=True)
    ]

    self.learning_rate = learning_rate
    self.network = nn.Sequential(*layers)
    self.log_probabilities = list()
    self.rewards = list()

  def forward(self, state: torch.Tensor):
    return self.network(state)

  def act(self, state: torch.Tensor) -> int:
    """
    Take an action based on the state of the environment given to the agent.

    :param state: State of the environment for the agent.

    :return: int
    """
    pd_parameters = self.forward(state)
    pd = Categorical(logits=pd_parameters)

    action = pd.sample() # pi(a|s)

    log_prob = pd.log_prob(action) # log probability of pi(a|s)
    self.log_probabilities.append(log_prob) # store these for training

    return action.item()

  def on_policy_reset(self):
    self.log_probabilities = list()
    self.rewards = list()

  def compute_loss(self) -> torch.Tensor:
    """
    Returns the loss for the current trajectory.

    :return: torch.Tensor
    """
    trajectory_length = len(self.rewards)
    returns = np.empty(trajectory_length, dtype=np.float32)
    future_returns = 0.0

    # compute discounted returns for each timestep
    for t in reversed(range(trajectory_length)):
      future_returns = self.rewards[t] + self.learning_rate * future_returns
      returns[t] = future_returns

    # log probabilities
    log_probabilities = torch.stack(self.log_probabilities)

    # compute loss
    returns = torch.tensor(returns)
    loss = -log_probabilities * returns
    loss = torch.sum(loss)

    return loss

  def train_agent(self, optimizer: optim) -> torch.Tensor:
    """
    Train the policy network of the agent.

    :param optimizer: Optimizer to use for training the policy network.

    :return: float
    """
    # set the network to training mode
    self.train()

    # compute loss
    loss = self.compute_loss()

    # zero grad to prevent gradient accumulation.
    optimizer.zero_grad()

    # backprop over the computation graph.
    loss.backward()
    optimizer.step()

    return loss

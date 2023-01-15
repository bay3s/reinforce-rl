import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Normal

class InvertedPendulumPolicy(nn.Module):

  DEFAULT_HIDDEN_SIZES = [256, 256]

  def __init__(self, in_features:int, hidden_sizes: list = DEFAULT_HIDDEN_SIZES):
    """
    Initialize a policy for a discrete action space.

    :param in_features: Number of input features.
    :param hidden_sizes: Number of neurons in each of the hidden layers.
    """
    super().__init__()

    self.log_probabilities = list()
    self.rewards = list()

    self.linear1 = nn.Linear(in_features, hidden_sizes[0])
    self.linear2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])

    self.mean = nn.Linear(hidden_sizes[1], 1)
    self.log_std = nn.Linear(hidden_sizes[1], 1)
    pass

  def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
    """
    Returns the mean and standard deviation for the distribution to sample from.

    :param x: Current state of the environment.

    :return: [torch.Tensor, torch.Tensor]
    """
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))

    mean = self.mean(x)
    sd = torch.clamp(self.log_std(x), min = -20, max = 2).exp()

    return mean, sd

  def act(self, state: torch.Tensor) -> torch.Tensor:
    """
    Take an action based on the state of the environment given to the agent.

    :param state: State of the environment for the agent.

    :return: int
    """
    # get mean and std
    mean, sd = self.forward(state)

    # create distribution and sample
    normal = Normal(mean, sd)
    action = normal.sample()

    # log probability of policy(a|s)
    log_prob = normal.log_prob(action)
    self.log_probabilities.append(log_prob)

    # squeeze action
    action = torch.tanh(action)

    return action.detach().numpy()

  def reset(self):
    """
    Reset the rewards and log probabilities.

    :return: None
    """
    self.log_probabilities = list()
    self.rewards = list()
    pass

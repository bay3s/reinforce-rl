import numpy as np

import torch
from torch.distributions import Normal
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class ContinuousActionSpaceAgent:

  def __init__(self, policy: nn.Module, discount_factor: float, use_baseline:bool = False):
    """
    Initialize a policy for a discrete action space.

    :param policy: Neural network to be used as the policy
    :param discount_factor: The discount factor to use for discounting rewards.
    :param use_baseline: Whether to use baseline.
    """
    super().__init__()

    self.policy = policy
    self.discount_factor = discount_factor
    self.use_baseline = use_baseline

    self.log_probabilities = list()
    self.rewards = list()
    pass

  def act(self, state: torch.Tensor) -> torch.Tensor:
    """
    Take an action based on the state of the environment given to the agent.

    :param state: State of the environment for the agent.

    :return: int
    """
    # get mean and std
    mean, sd = self.policy.forward(state)

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

  def compute_loss(self) -> torch.Tensor:
    """
    Returns the computed loss for the trajectory.

    :return: torch.Tensor
    """
    trajectory_length = len(self.rewards)
    returns = np.empty(trajectory_length, dtype = np.float32)
    future_ret = 0.

    for t in reversed(range(trajectory_length)):
      future_ret = self.rewards[t] + self.discount_factor * future_ret
      returns[t] = future_ret
      pass

    return -torch.sum(torch.stack(self.log_probabilities) * torch.tensor(returns))

  def tune(self, optimizer: Optimizer):
    """
    Given a policy compute the loss associated with it.

    :param optimizer: The optimizer to be used for computing loss.

    :return: float
    """
    loss = self.compute_loss()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

import numpy as np
import torch
import torch.nn as nn


def compute_loss(policy: nn.Module, optimizer: torch.optim, discount_factor: float):
  """
  Given a policy compute the loss associated with it.

  :param policy: Neural network to be used as the policy for the agent.
  :param optimizer: The optimizer to be used to tune the policy.
  :param discount_factor: The discount rate for the rewards.

  :return: float
  """
  trajectory_length = len(policy.rewards)
  returns = np.empty(trajectory_length, dtype = np.float32)
  future_ret = 0.

  # compute the returns efficiently.
  for t in reversed(range(trajectory_length)):
    future_ret = policy.rewards[t] + discount_factor * future_ret
    returns[t] = future_ret
    pass

  loss = torch.sum(-torch.stack(policy.log_probabilities) * torch.tensor(returns))
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  return loss

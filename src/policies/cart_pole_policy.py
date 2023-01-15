import torch
import torch.nn as nn


class CartPolePolicy(nn.Module):

  def __init__(self, in_features:int, out_features:int):
    """
    Initialize a policy for a discrete action space.

    :param in_features: Number of input features.
    :param out_features: Number of output features.
    """
    super().__init__()

    layers = [
      nn.Linear(in_features=in_features, out_features=64, bias=True),
      nn.ReLU(),
      nn.Linear(64, out_features=out_features, bias=True)
    ]

    self.model = nn.Sequential(*layers)
    pass

  def forward(self, state: torch.Tensor):
    """
    Forward pass through the neural net.

    :param state: The current state of the environment.

    :return: torch.Tensor
    """
    return self.model(state)

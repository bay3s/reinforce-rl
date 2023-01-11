import torch
import torch.nn as nn
from torch.distributions import Categorical


class DiscretePolicy(nn.Module):

  def __init__(self, in_features:int, out_features:int):
    """
    Initialize a policy for a discrete action space.

    :param in_features: Number of input features.
    :param out_features: Number of output features.
    """
    super().__init__()

    self.log_probabilities = list()
    self.rewards = list()

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

  def act(self, state: torch.Tensor) -> int:
    """
    Take an action based on the state of the environment given to the agent.

    :param state: State of the environment for the agent.

    :return: int
    """
    pd_parameters = self.forward(state)
    pd = Categorical(logits=pd_parameters)

    action = pd.sample() # policy(a|s)

    log_prob = pd.log_prob(action) # log probability of policy(a|s)
    self.log_probabilities.append(log_prob) # store these for training

    return action.item()

  def reset(self):
    """
    Reset the rewards and log probabilities.

    :return: None
    """
    self.log_probabilities = list()
    self.rewards = list()
    pass

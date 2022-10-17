import numpy as np
import gym
import torch
import torch.optim as optim

from src.agent.policy import Policy

GAMMA = 0.99


def main():
  env = gym.make('CartPole-v0')

  # input dimensions:
  policy = Policy(env.observation_space.shape[0], env.action_space.n)
  optimizer = optim.Adam(policy.parameters(), lr=0.01)

  for episode in range(300):
    obs, _ = env.reset()
    state = torch.from_numpy(obs)

    # simulate the current episode
    for t in range(200):
      action = policy.act(state)
      obs, reward, terminated, _, _ = env.step(action)
      state = torch.from_numpy(obs)
      policy.rewards.append(reward)

      if terminated:
        break

    loss = policy.train(optimizer)
    total_reward = sum(policy.rewards)

    solved = total_reward > 195.0
    policy.on_policy_reset()

    print(f'Episode {episode}, STEP: {t}, Loss: {loss}, Total Reward: {total_reward}, Solved: {solved}.')

    if solved:
      break


if __name__ == '__main__':
  main()

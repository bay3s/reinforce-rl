import gym
import torch
import torch.optim as optim

from src.agent.continuous_policy import ContinuousPolicy

MAX_EPISODES = 300
MAX_TRAJECTORY = 1000

def main():
  env = gym.make('Pendulum-v1')
  policy = ContinuousPolicy(env.observation_space.shape[0])
  optimizer = optim.Adam(policy.parameters(), lr=0.01)

  for episode in range(MAX_EPISODES):
    obs, _ = env.reset()
    state = torch.from_numpy(obs)

    # simulate the current episode
    for t in range(MAX_TRAJECTORY):
      action = policy.act(state)
      obs, reward, terminated, _, _ = env.step(action)
      state = torch.from_numpy(obs)
      policy.rewards.append(reward)

      if terminated:
        break

    loss = policy.train_agent(optimizer)
    total_reward = sum(policy.rewards)

    solved = total_reward > 195.0
    policy.on_policy_reset()

    print(f'Episode {episode}, Loss: {loss}, Total Reward: {total_reward}, Solved: {solved}.')

    if solved:
      break


if __name__ == '__main__':
  main()

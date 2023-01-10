### REINFORCE

#### Intro
- Invented in 1992 in the paper "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning".
- Key idea is that while learning, actions that resulted in good outcomes should become more probable - the actions are positively reinforced.
- Conversely, actions that resulted in bad outcomes become less probable.
- If learning is successful, over the course of many iterations, the policy should shift to distribution that results in good performance in an environment.
- Action probabilities are changed by following the policy gradient, so REINFORCE is known as a policy gradient algorithm.
- Additionally, REINFORCE is built upon trajectories instead of episodes because maximizing expected return over trajectories (instead of episodes) lets the method search for optimal policies for both episodic and continuing tasks.
	- Trajectories are more flexible than episodes since there are no restrictions in length.
	- A trajectory can correspond to a full episode or just a part of an episode.
- Algorithm has three components:
	- Policy
	- Objective
	- Policy Gradient (method for updating the policy)


#### Policy $\pi_\theta$
- A policy is a function that maps states to action probabilities. A good policy will maximize the cumulative discounted rewards. 
- REINFORCE learns a good policy through function approximation using a neural network.
- The process of learning a good policy corresponds to searching for a well parameterized neural network, so it is important that the policy network is differentiable (for backprop).


#### Objective $J(\pi_\theta)$
- An agent acting in the environment generates a trajectory. 
- The return of a trajectory is defined as a discounted sum of rewards from the beginning till the end of the trajectory.
$$R_t(\tau) = \sum_{t'=t}^T \space \gamma^{t'-t}r'_t$$
- The objective is the expected return over all complete trajectories generated by an agent.
$$J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[R(\tau)\right]$$
- The expected return converges to the true value as more samples are gathered. 
- The objective can be thought of as an abstract hypersurface on which we try to find the maximum point with $\theta$ as the variables.


#### Policy Gradient $\triangledown_\theta \space J(\pi_\theta)$
- The policy gradient serves to maximize the "objective" (expected return over completed trajectories), solving the following problem:

$$\max_\theta \space J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

- We perform gradient ascent on the policy parameters in order to do achieve this:
$$\theta \leftarrow \theta + \alpha \triangledown_\theta  J(\pi_\theta)$$
- The policy gradient $\triangledown_\theta \space J(\pi_\theta)$ is given by:
$$\triangledown_\theta \space J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} R_t(\tau)\triangledown_\theta \log \pi_\theta(a_t|s_t)\right]$$

#### Monte Carlo Sampling
- The REINOFRCE algorithm numerically estimates the policy gradient using [[Monte-Carlo Sampling]] sampling.
- In the context of this algorithm, we see that the more trajectories that are sampled by the agent and averaged, the more likely it is that we get to the actual policy gradient.
- Instead of sampling many trajectories per policy, we can sample just one per policy.
- Thus, policy gradient in REINFORCE is implemented as a Monte Carlo estimate over sampled trajectories.


#### Pseudocode
1. Initialize learning rate $\alpha$
2. Initialize weights $\theta$ of a policy network $\pi_\theta$
3. FOR episode=:0,..., MAX_EPISODES} DO
	1. Sample a trajectory $\tau = s_0, a_0, r_0, ..., s_T, a_t, r_t$
	2. Set $\triangledown_\theta J(\pi_\theta) = 0$
	3. FOR t={0,...,T} DO:
		1. $R_t{\tau} = \sum_{t'=t}^T \gamma^{t'-t}r'_t$
		2. $\triangledown_\theta J(\pi_\theta) = \triangledown_\theta J(\pi_\theta) + R_t(\tau)\triangledown_\theta \space log \space \pi_\theta(a_t|s_t)$
	2. END FOR
	3. $\theta = \theta + \alpha \triangledown_\theta J(\pi_\theta)$
4. END FOR



#### Gotchas
- It is important that a trajectory is discarded after each parameter update - it cannot be reused because REINFORCE is an on-policy algorithm so the parameter update equation depends on the current policy, and after a policy update the trajectory cannot be used to update any subsequent policies (since we would be updating policies that did not produce the trajectory).


#### Limitations
- When using Monte Carlo sampling, the policy gradient estimate may have high variance because the returns can vary significantly from trajectory to trajectory.
- This is due to 3 factors: 
    - actions have some randomness because they are sampled from a probability distribution.
    - starting state may vary per episode.
    - environment transition function can be stochastic.

**Example Improvements:**
- One way to reduce the variance of the estimate is to modify the returns by subtracting a suitable action-independent baseline similar to Actor-Critic algorithm.
- Another one is to use the mean returns over a trajectory to center the return for each trajectory around 0. For each trajectory on average, the best 50% of returns would be encouraged and the others discouraged.


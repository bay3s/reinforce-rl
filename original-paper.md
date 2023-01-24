**Reference**
 - <a href="https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf">Gradient-Following Algorithms for Connectionist RL (Williams, 1992)</a>

**Abstract**
- Presents a general class of associative RL algorithms for neural networks containing stochastic units (non-linearities / distributions).
- Make weight adjustments in a direction that lies along the gradient of expected reinforcement in both immediate reinforcement tasks and limited forms of delayed-reinforcement tasks.
- This is done without explicitly computing the gradient or storing information from which gradient estimates could be computed.

**Intro**

*Basics*
- Associative taks are ones where the learner is required to perform an input-output mapping - an exception to this is that of immediate refinrocement where the payooff is determined by the most recent input-output pair alone.
- Algorithms presented do not explicitly compute the gradient estimate or store information from which gradient estimates may be computed.
- Experiments adopt a connectionist perspective, but should be noted that analysis can be applied to other ways of implementing input-output mappings.
- A connectionist perspective basically means that the learning agent is a feedforward network.
	- The network operates by receiving input and propagating the corresponding activity through the net which is sent to the environment for "evaluation".
	- The evaluation is given in the form of reinforcement signal, a scalar reward r.

*Notation*
- $x^i$ is the input to the network unit i
- $y^i$ is the output of the network unit i
- $w^i$ is the set of all weights on which the functioning of the unit i depends
- $W$ is the weight matrix
- $g^i(\xi, w^i, x^i) = Pr(y_i = \xi | W, x^i)$ so that $g^i$ is the probability mass function determining the value of $y_i$ as a function of the parameters and its input.
- A stochastic semilinear unit is defined as one whose ouput is drawn from a probability distributin whose mass function has a single parameter $p_i$ computed as $p_i = f_i(s_i)$.
	- $f_i$ is a differentiable squashing function (in the context of this paper a sigmoid)
	- $s_i = w^{iT}x^i = \sum_j w_{ij}x_j$
- Logistic function
	- The general expression is given by $f_i(s_i) = \frac{1}{1 + e ^{-s_i}}$

*Expected Reinforcement Performance Criterion*
- For gradient learning algorithms, we need a performance measure to optimize.
- A natural one for immediate-RL problems is the expected value of rewards conditioned on a particular choice of parameters $E(r|W)$.
- Expected values are used because of randomness in:
	- inputs to the network
	- output corresponding to a specific input
	- reward for a particular input-output
- Makes sense to discuss $E(r|W)$ independent of time only in cases where inputs & rewards are time independent - assumption is that inputs & rewards come from stationary distributions.
- Based on the assumptions made, $E(r|W)$ is a well-defined deterministic funciton of W.
- In this formulation, objective of the RL system is to search the space of possible weights which maximizes $E(r|W)$


**REINFORCE Algorithms**

*Immediate Reinforcement*
- Network faces an associative imediate-RL task.
- Weigths are adjusted in the network following receipt of the reward.
- At the end of each trial, parameters $w_{ij}$ in the network is incremented by an amount 

$$\triangle w_{ij} = \alpha_{ij}(r - b_{ij})e_{ij}$$
- where
	- $b_{ij}$ is the baseline
	- $e_{ij} = \partial ln g_i/ \partial w_{ij}$
- REINFORCE stands for "Reward Increment = Non-negative Factor x Offset Reinforcement x Characteristic Eligibility"

*Theorem 1.* Says that for any REINFORCE algorithm the average update vector in weight space lies ina  direction for which this performance measure is increasing.
- The inner product of $E(\triangle W|W)$ and $\triangledown_W E(r|W)$ is non-negative.
- If $\alpha_{ij} > 0$ for all i and j, then this inner product is zero only when $\triangledown_W E(r|W) = 0$
- If $\alpha_{ij}$ is independent of i and j, then $E(\triangle W|W) = \alpha \triangledown_w E(r|W)$
- For each weight $w_{ij}$ the quantity $(r - b_{ij}) \partial ln g_i / \partial w_{ij}$ is an unbiased estimate of $\partial E(r|W) / \partial w_{ij}$

*Epsiodic Reinforcement*
- REINFORCE extended to learning problems having a temporal credit-assignment component.
- Assume a net N is trained on a episode-by-episode basis where each episode consists of k ime steps.
- Single reinforcement r value is provided to the net at the end of each episode.
- At the conclusion of each episode each weight $w_{ij}$ is incremented by:
$$\triangle w_{ij} = \alpha_{ij} (r - b_{ij}) \sum^k_{t = 1} e_{ij}(t)$$
- where $e_{ij}(t)$ represents the notion of characteristic eligibility for $w_{ij}$ evaluated at time t.

A more general formulation of episodic learning involves reinforcement delivered at each time step during the episode. 
- The appropriate performance measure is

$$E(\sum_{t=1}^kr(t) |W)$$

- A statistical gradient-following algorithm for this case is
$$\triangle w_{ij} = \alpha_{ij}(\sum_{t=1}^kr - b_{ij}) \sum_{t=1}^k e_{ij}(t) $$

*Multiparameter Distributions*
- Application of the REINFORCE framework to the development of learning algorithms for units that determine their scalar output stochasticlly from multiparmeter distributions.
- One way to do this is to deterministically compute parameters for the prbability distribution then sample from the distribution.
	- eg. compute the mean and standard deviation deterministically for a normal distribution, then sample.
- Can use different weights (not including feature extraction layers) to control the each of the parameters of the distribution - leads to separation of concerns and may avoid problems such as catastrophic forgetting.

*Example: Gaussian Output*
- Possible outputs is the set of real numbers and the density function g determining the output y is given by
$$g(y, \mu, \sigma) = \frac{1}{(2\pi)^{1/2}\sigma}e^{-(y - \mu)^2 / 2\sigma^2}$$

- Characteristic eligibility of $\mu$ is 

$$\frac{\partial ln \space g}{\partial \mu} = \frac{y - \mu}{\sigma^2}$$

- Characteristic eligibility of $\sigma$ is

$$\frac{\partial ln \space g}{\partial \sigma} = \frac{(y - \mu)^2 - \sigma^2}{\sigma^3}$$

- REINFORCE algorithm in this case has the form

$$\triangle \mu = \alpha_\mu (r - b_\mu) \frac{y - \mu}{\sigma^2}$$

$$\triangle \sigma = \alpha_\sigma (r - b_\sigma) \frac{(y - \mu)^2 - \sigma^2}{\sigma^3}$$


*Networks Using Deterministic Hidden Units*
- Let x denote the vector of network input and y denote the network output vector.
- We can define $g(\xi, W, x) = Pr(y = \xi | W, x)$ the be overall probability mass function describing the input-output behavior of the entire network.
- Output O of the network is vector-valued and - because fo randomness in the ouput and because randomness is independent across untis we have:


$$\begin{aligned}Pr(y = \xi | W, x) &= \prod_{k \in O} Pr(y_k = \xi_k |W, x)  \\
&= \prod_{k \in O} Pr(y_k = \xi_k |w^k, x^k)\end{aligned}$$

- Taking the natural logs we get

$$ln \space g (\xi, W, x) = ln \prod_{k \in O} g_k (\xi_k, w^k, x^x) = \sum_{k \in O} ln \space g_k(\xi_k, w^k, x^k)$$
- So that
$$\frac{\partial ln \space g}{\partial w_{ij}} (\xi, W, x) = \sum_{k \in O} \frac{\partial ln \space g_k}{\partial w_{ij}}(\xi_k, w^k, x^k)$$
- This can clearly be computed via backpropagation - hence compatible with neural nets.
- When one set of variables depends deterministically on a second set of variables, backpropagating unbiased estimates of partial derivatives with respect to the first set of variables gives rise to unbiased estimates of partial derivatives with respect to the second set of variables.


*Backpropagating through random number generators*
- Suppose that it were possible to "backpropagate through a random number generator".
- Consider a stochastic semilinear unit and a function J having some deterministic dependence on the $y_i$ (the output of the unit).
	- $y_i$ is sampled from a random number generator parameterized by $p_i$.
	- Example of this is when the unit is an output and $J = E(r|W)$ with reinforcement depending on whether the output is correct.
- We would like to be able to compute $\partial J / \partial p_i$ based on knowledge of $\partial J / \partial y_i$.
- But - we could not expect there to be a deterministic relationship between these quantities due to the randomness.
- A more reasonable property to look for is $\partial E(J|p_i)/\partial p_i$ but that fails to hold in general as well.
- However, if the output of the random nuber generator can be written as a differentiable function of its parameters the approach descibed for backpropagating through deterministic computation can be applied.

*Example: Backprop thorugh random number generator*
- Consider a normal random number generator (a Gaussian)
- We may write:
$$y = \mu + \sigma z$$
- where z is a standard normal deviate.
- From this
$$\frac{\partial y}{\partial \mu} = 1$$
- and 
$$\frac{\partial y}{\partial \sigma} = z = \frac{y - \mu}{\sigma}$$
- In such cases, we can combine the use of backprop through Gaussian hidden units with REINFORCE in the output units.
- Since backprop preserves the unbiasedness of gradient estimates in general, this form of argument can be applied to yield statistical gradient-following algorithms that make use of backprop in a variety of other situations where a network of continuous-valued stochastic units is used (for instance in supervised learning).


**Remarks**

*Convergence Properties*
- Limitation of analysis in the paper is that it is not predictive of the asymptotic properties of the algorithm.
- William's previous experiments suggested that such algorithms tend to converge at a local optima.
- Some variants ahve examined the incorporation of modifications designed to overcome this behavior (eg. using entropy in the reward signal which seems to have helped improve performance in tasks that especially require hierarchical organization during the search).
- Vanilla Episodic REINFORCE was found to be slow since it performs temporal credit-assignment by essentially spreading credit or blame over all previous timesteps.
- Algorithms tend to converge to suboptimal solutions (inferior action choices) in cases where the action is always in the direction of the best action based on the current parameterization.
- Depending on the choice of baseline, any REINFORCE algorithm is more or less likely to converge to a local maxima with a non-zero probability of convergece to other points that lead to zero variance in network behavior.

*Gaussian Unit Search Behavior*
- The optimization consists of a single real variabl y, and the adaptable parameters $\mu$ and $\sigma$.
- If the value y is sampled which leads to a higher function value than has been obtained in the recent past,
	- $\mu$ moves towards points giving higher function values
	- $\sigma$ decreases if $|y - \mu| < \sigma$
- If the sampled point y gives rise to a lower function value than has been obtained in the past, 
	- $\mu$ moves away from points giving lower function values
	- $\sigma$ increases if $|y - \mu| > \sigma$
- Search is narrowed around $\mu$ if a better point is found suitably close to the mean and vice versa.
	- If $\mu$ sits on top of a local hill, then $\sigma$ would narrow down to allow convergence to the local maxima.
	- If the local maxima is flat on top, $\sigma$ would decrease to the point where sampling worse values is unlikely and then stop changing.
- If reinforcement r is always non-negative and baselines are not used, REINFORCE may cause $\sigma$ to converge to 0 before $\mu$ has moved to any local or global maxima.
- An alternate approach proposed by Gullapalli (1990) involves normalizing the rewards between 0 and 1, then $\sigma$ can be taken to be proportional to $1 - r$ so that σ is controlling the scale of the search.

*Choice of Baselines*
- Limitation of the analysis is that it offers no basis for choosing baselines.
- An adaptive baseline that incorporates something like the reinforcement comparison strategy can greatly enhance convergence speed, and, in some cases can lead to a big difference in qualitative behavior.
- Example,
	- Consider a single Bernoulli semilinear unit with only a bias weight and input with its output y affecting r deterministically.
	- If r is always positive and in the absence of baselines, behavior similar to a biased random walk is exhibited which leads to non-zero probabilty of convergence to inferior solutions.
	- In contrast, reinforcement comparison (mean reward as baseline) leads to values of the baseline lying between the two values of r which leads to motion toward the better output value.
	- But, this behavior can also be obtained via a other baseline choices whose value is between the two values of r.

*Use of other Local Gadient Estimates*
- There are alternative ways to estimate gradients and it's useful to understand how these techniques can be integrated effectively.
- Model-based proposed by Barto, Sutton, Watkins (1990)
	- Correspond to indirect algorithms in the adaptive control field
	- Explicity estimate relevant parameters underlying the system to be controlled
	- Use this learned model to compute control actions

**Conclusion**
- REINFORCE algorithms are useful in their own right and perhaps serve as a sound basis for developing more effective RL algorithms.
- A major advantage of REINFORCE is that it represents a prescription for devising statistical gradient-following algorithms for RL networks that may compute their output in arbitrary ways.
- REINFORCE integrates well with other gradient computation techniques.
- Disadvantages of REINFORCE include lack of general convergence theory applicable to this class of algorithms and susceptibility to convergence to false optima.

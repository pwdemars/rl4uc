# RL4UC: Reinforcement Learning for Unit Commitment

This project contains an RL environment for the unit commitment problem.

### Introduction to the UC problem 

The unit commitment problem is the task of determining the on/off statuses of generators in a power system in order to meet demand at minimum cost. This environment is primarily for experimenting with RL solutions to this problem. In the RL context, we can formulate a Markov Decision Process where: 

- States refer to the grid configuration (on/off statuses of generators), demand and wind forecasts, forecast errors. 
- Actions refer to commitment decisions determining the on/off statuses of generators for the next timesteps. *Note: the action space is limited by the operating constraints of generators.*
- Rewards reflect the (negative) operating cost of the grid, including costs for not meeting demand.
- Transitions represent the realisations of taking actions on the grid, and the realisations of stochastic processes (so far, this is demand and wind generation).

The process of acting on the environment and receiving a reward and state observation can be broken down into the following: 

1. Make a commitment decision: a binary vector of length N, the number of generators. 
2. Roll forward the environment one timestep, sampling a new (usually stochastic) demand D.
3. Solve the economic dispatch problem for the new commitment at demand D to work out the outputs of online generators.
4. Return the negative operating cost as reward and a new state observation.

##### Economic dispatch 

The UC problem only determines the on/off statuses of generators. The task of determining the optimal outputs of the online generators is known as the **economic dispatch problem** and is typically a relatively straightforward convex optimisation problem. 

In this RL environment, the economic dispatch problem is solved by the lambda-iteration method (see Wood et al., Power Generation, Operation and Control for more details). 


## Installation

You can install the development version of this repository by running:

```
git clone https://github.com/pwdemars/rl4uc.git
cd rl4uc
pip install .
```

Or the latest stable release from PyPI: 

```
pip install rl4uc
```

## Example usage:

Below we will try an action on the 5 generator system. An action is a commitment decision for the following time period, defined by a binary numpy array: 1 indicates that we want to turn (or leave) the generator on, 0 indicates turn or leave it off. 

```python 
from rl4uc.environment import make_env
import numpy as np

# Create an environment, 5 generators by default.
env = make_env()

# Reset the environment to a random demand profile.
obs_init = env.reset()

# Define a commitment decision for the next time period.
action = np.array([1,1,0,0,0]) # Turn on generators 0 & 1, turns all others off.

# Take the action, observe the reward.
observation, reward, done = env.step(action)

print("Dispatch: {}".format(env.disp))
print("Finished? {}".format(done))
print("Reward: {:.2f}".format(reward))
```


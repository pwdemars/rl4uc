# RL4UC: Reinforcement Learning for Unit Commitment

This project contains an RL environment for the unit commitment problem.

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

print("Dispatch: {}".format(env.dispatch))
print("Finished? {}".format(done))
print("Reward: {:.2f}".format(reward))
```


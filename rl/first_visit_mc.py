import numpy as np
import random
from collections import defaultdict

# np.random.seed(0)
# random.seed(0)

### TASK ###
"""
You are provided with a list of len(n). There are m steps in the game. At each step m you select a single value from the list.
You are only allowed to select a value within max_step_size places of the index you accessed on the previous turn. You start at a random index between idx_lower and idx_upper.
Your goal is to max a reward. The reward is drawn from a Gaussian centred at the value of the index you selected. 
You do not have access to the underlying values, and must thus build a value function to estimate the value of each index.
"""

def policy(current_idx:int):
    """ dumb policy that simply steps ahead 2 places every time"""
    return  current_idx + 2

def sample(mean:int, sd:float):
    """ Sample from a Gaussian distribution
    mean; in this case the value of the environment at the current index"""
    return np.random.normal(mean, sd)

def generate_episode():
    """ Follow the dump policy to generate an episode and returns Tuple[episode, total];
    episode: a list of tuples of (step, action, reward)
    total: the sum of the rewards for the episode"""
    current_idx = random.randint(start_idx_lower, start_idx_upper)
    episode = []
    total = 0
    for step in range(m):
        action = policy(current_idx)
        action = max(0,min(action, n-1)) # clip the action to the bounds of the environment
        reward = sample(env[action], sd)
        current_idx = action
        episode.append((step, action, reward))
        total += reward

    return episode, total

def avg_untrained_reward(simulations=100):
    """ Average the reward of 100 episodes using the dumb policy"""
    untrainded_returns = []
    for _ in range(simulations):
        episode, total_reward = generate_episode()
        untrainded_returns.append(total_reward)
    return np.mean(untrainded_returns)

class MCOneVisit:
    """ First visit Monte Carlo with a value function for the environment"""

    def __init__(self):
        self.returns = defaultdict(list)
        self.value = np.random.randint(-1,1, size=(m, n))

    def one_iteration(self):
        """Train the value function for one episode using first visit MC."""
        episode, _ = generate_episode()
        G = 0
        visited = set()
        for step, action, reward in reversed(episode):
            G = reward + G
            if (step,action) not in visited: # first visit MC
                visited.add((step,action))
                self.returns[(step,action)].append(G)
                # There exist more efficient ways that dont require taking the mean at each step. 
                self.value[step][action] = np.mean(self.returns[(step,action)]) 
                
    def avg_trained_reward(self, simulations=100):
        """ Average the reward of 100 episodes using the trained value function"""
        trained_returns = []  
        for _ in range(simulations):
            current_idx = random.randint(start_idx_lower, start_idx_upper)
            G = 0
            for step in range(m):
                valid_action_lower = max(0, current_idx - max_step_size)
                valid_action_upper = min(current_idx + max_step_size + 1, n-1)
                valid_actions_slice = self.value[step][valid_action_lower:valid_action_upper]
                # argmax the best action within the slice
                action = np.argmax(valid_actions_slice) + valid_action_lower
                reward = sample(env[action], sd)
                G += reward
                current_idx = action

            trained_returns.append(G)
        return np.mean(trained_returns)


if __name__ == "__main__":

    sd = 1 # standard deviation of the Gaussian for the reward distribution 
    epochs = 100 # number of iterations to train the value function
    n = 100 # number of values in the environment
    m = 7 # number of steps in episode
    max_step_size = 6 # maximum step size
    start_idx_upper = 55
    start_idx_lower = 45

    if not ((max_step_size * m) + start_idx_upper < n) or not (start_idx_lower - (max_step_size * m)  >=0):
        print("Warning: steps size may be clipped by the bounds of the environment. Consider changing max_step_size or n")

    env = np.random.randint(-5, 5, size=n)


    for test in range(10):
        mc = MCOneVisit()
        untrained = avg_untrained_reward()
        for _ in range(epochs):
            mc.one_iteration()
        trained = mc.avg_trained_reward()
        print(f"Trained - Untrained: {trained - untrained:.2f}, for SD {sd} and epochs {epochs}")


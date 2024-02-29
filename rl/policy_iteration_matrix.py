import numpy as np

n = 3  # Dimension of the matrix
gamma = 0.9  # Discount factor for future rewards

# The reward matrix
m = np.array([
    [1, 0, 0],
    [1, -1, 0],
    [1, 1, 1],
])

# Initialize policy randomly: 0 for down, 1 for right
policy = np.random.randint(2, size=(n, n))

# Initialize value function to zeros
value = np.zeros((n, n))

def is_terminal_state(i, j):
    # Define terminal states
    return (i, j) == (n-1, n-1) or m[i, j] == -1

def policy_evaluation(policy, value, theta=0.001):
    while True:
        delta = 0
        for i in range(n):
            for j in range(n):
                if is_terminal_state(i, j):
                    continue
                v = value[i, j]
                action = policy[i, j]
                new_i, new_j = (i + 1, j) if action == 0 else (i, j + 1)
                if new_i >= n or new_j >= n or m[new_i, new_j] == -1:
                    new_value = -10  # Penalize for out-of-bounds or hitting a -1
                else:
                    new_value = m[new_i, new_j] + gamma * value[new_i, new_j]
                value[i, j] = new_value
                delta = max(delta, abs(v - new_value))
        if delta < theta:
            break

def policy_improvement(policy, value):
    policy_stable = True
    for i in range(n):
        for j in range(n):
            if is_terminal_state(i, j):
                continue
            old_action = policy[i, j]
            # Evaluate the expected value of each action
            down_value = -10 if i + 1 >= n or m[i + 1, j] == -1 else m[i + 1, j] + gamma * value[i + 1, j]
            right_value = -10 if j + 1 >= n or m[i, j + 1] == -1 else m[i, j + 1] + gamma * value[i, j + 1]
            # Update policy with the best action
            policy[i, j] = 0 if down_value > right_value else 1
            if old_action != policy[i, j]:
                policy_stable = False
    return policy_stable

def policy_iteration():
    global policy, value
    while True:
        policy_evaluation(policy, value)
        policy_stable = policy_improvement(policy, value)
        if policy_stable:
            break

policy_iteration()

print("Optimized Policy (0: down, 1: right):")
print(policy)
print("Value Function:")
print(value)

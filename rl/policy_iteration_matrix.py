import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(edgeitems=100, linewidth=100000)
np.random.seed(1234567890)


n = 15 # matrix dim
gamma = 1  # No need to discount

def print_policy(policy, value):
    """Convert policy actions to symbols, using a special symbol for unreachable or invalid squares."""
    for i in range(n):
        for j in range(n):
            if np.isinf(value[i, j]):
                policy[i, j] = -1  

    mapping = {0: u'\u2193', 1: u'\u2192', -1: u'X'}  # Down arrow, right arrow, and 'X' for unreachable/invalid
    mapper = np.vectorize(lambda x, y: mapping[x] if not np.isinf(y) else mapping[-1])
    symbol_policy = mapper(policy, value)
    print(symbol_policy)

def plot_value(value):
    # Replace -np.inf with -1 for plotting
    value_plot = np.where(value == -np.inf, -1, value)
    x = np.arange(0, n)
    y = np.arange(0, n)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Y, X, value_plot, cmap='viridis')
    ax.set_zlabel('Value')
    plt.show()

def init_policy_value():
    # 0 for down, 1 for right
    policy = np.random.randint(2, size=(n, n))
    value = np.full((n, n), -np.inf)  # Start with all values as -inf
    m = np.random.choice([-1, 0, 1, 3], size=(n,n),p=[0.15,0.5,0.3, 0.05])

    for i in range(n):
        for j in range(n):
            if m[i, j] != -1:
                value[i, j] = 0
    value[0, 0] = m[0, 0] if m[0, 0] != -1 else -np.inf
    value[n-1, n-1] = m[n-1, n-1] if m[n-1, n-1] != -1 else -np.inf
    return policy, value, m

def is_terminal_state(i, j):
    return (i, j) == (n-1, n-1)

def policy_evaluation(policy, value, theta=0.1):
    while True:
        delta = 0
        for i in range(n):
            for j in range(n):
                if m[i, j] == -1 or is_terminal_state(i, j):
                    continue
                v = value[i, j]
                action = policy[i, j]
                new_i, new_j = (i + 1, j) if action == 0 else (i, j + 1)
                new_value = -np.inf
                if new_i < n and new_j < n and m[new_i, new_j] != -1:
                    new_value = m[new_i, new_j] + gamma * value[new_i, new_j]
                if not np.isinf(new_value):  # Only update if new_value is not -inf
                    value[i, j] = new_value
                    delta = max(delta, abs(v - new_value))
        if delta < theta:
            break

def policy_improvement(policy, value):
    policy_stable = True
    for i in range(n):
        for j in range(n):
            if m[i, j] == -1 or is_terminal_state(i, j):
                continue
            old_action = policy[i, j]
            # Default to -np.inf to ensure comparisons are valid
            down_value = value[i + 1, j] if i + 1 < n and m[i + 1, j] != -1 else -np.inf
            right_value = value[i, j + 1] if j + 1 < n and m[i, j + 1] != -1 else -np.inf
            best_action = 0 if down_value > right_value else 1
            policy[i, j] = best_action
            if old_action != best_action:
                policy_stable = False
    return policy_stable

def policy_iteration():
    global policy, value
    while True:
        policy_evaluation(policy, value)
        if policy_improvement(policy, value):
            break

def reachability_analysis():
    reachable = np.full((n, n), False)
    stack = [(n-1, n-1)]  # Start from the goal
    while stack:
        i, j = stack.pop()
        if not reachable[i, j]:
            reachable[i, j] = True
            # Check for squares that can move to (i, j)
            if i > 0 and policy[i-1, j] == 0 and m[i-1, j] != -1:
                stack.append((i-1, j))
            if j > 0 and policy[i, j-1] == 1 and m[i, j-1] != -1:
                stack.append((i, j-1))
    # Update value function based on reachability
    for i in range(n):
        for j in range(n):
            if not reachable[i, j]:
                value[i, j] = -np.inf


if __name__ == "__main__":
    np.set_printoptions(edgeitems=100, linewidth=100000)
    np.random.seed(1234567890)

    n = 15 # matrix dim
    gamma = 1  # No need to discount

    policy, value, m = init_policy_value()
    policy_iteration()
    reachability_analysis()

    print("Original Matrix")
    print(m)
    print("Optimized Policy:")
    print_policy(policy, value)
    print("Value Function:")
    print(value)
    plot_value(value)
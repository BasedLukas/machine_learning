import numpy as np

from utils.plotting import plot_2d_heatmap, plot_policy, show



def value_update(matrix, value, gamma=1) -> float:
    """Update the value function using the Bellman equation."""
    delta = 0
    m,n = matrix.shape
    for ij in range(m * n):
        i, j = divmod(ij, n)

        old_value = value[i, j]
        best_next_value = -np.inf  
        
        # Check possible actions and update value function
        if i + 1 < m:
            best_next_value = max(best_next_value, matrix[i + 1, j] + gamma * value[i + 1, j])
        if j + 1 < n:
            best_next_value = max(best_next_value, matrix[i, j + 1] + gamma * value[i, j + 1])
        
        if best_next_value == -np.inf:
            best_next_value = 0  # No future rewards
        
        value[i, j] = best_next_value
        delta = max(delta, abs(old_value - best_next_value))
    
    return delta
 
def create_policy(matrix, value, policy, gamma=1):
    """Create a deterministic policy based on the value function."""
    m,n = matrix.shape
    for ij in range(m * n):
        i, j = divmod(ij, n)
        
        # Initialize variables to track the best action
        best_value = -np.inf
        best_action = None  # 0 for down, 1 for right
        
        # Evaluate possible actions
        if i + 1 < m:
            down_value = matrix[i + 1, j] + gamma * value[i + 1, j]
            if down_value > best_value:
                best_value = down_value
                best_action = 0
        if j + 1 < n:
            right_value = matrix[i, j + 1] + gamma * value[i, j + 1]
            if right_value > best_value:
                best_value = right_value
                best_action = 1
        
        # Update policy
        if best_action is not None:
            policy[i, j] = best_action

def value_iteration(matrix, value, policy):
    while True:
        delta = value_update(matrix,value)
        if delta == 0:
            create_policy(matrix, value, policy)
            break



if __name__ == "__main__":

    m,n = 15, 20 # matrix dim
    value = np.random.random(size=(m,n))  
    policy = np.random.randint(0,2, size=(m,n)) # 0 for down, 1 for right
    matrix = np.random.choice([-16, 0, 1, 18], size=(m,n),p=[0.05,0.4,0.5, 0.05])

    value_iteration(matrix, value, policy)

    plot_2d_heatmap(matrix, "Environment")
    plot_policy(policy, "Policy")
    plot_2d_heatmap(value, "Value Function")
    show()


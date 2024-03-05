import numpy as np

from utils.plotting import plot_2d_heatmap, plot_policy, show



def policy_evaluation(matrix, policy, value):
    """Update the value function to reflect the current policy"""
    m,n = matrix.shape
    while True:
        delta = 0
        for ij in range(m*n):
            i, j = divmod(ij, n)
            action = policy[i, j]
            new_i, new_j = (i + 1, j) if action == 0 else (i, j + 1)
            if new_i < m and new_j < n:
                # the updated value for state (i, j) is the reward we get by following the policy on to the next state, plus the value of the next state
                new_value = matrix[new_i, new_j] + value[new_i, new_j]
            else:
                # penalise invalid moves
                new_value = -20

            old_value = value[i, j]
            value[i, j] = new_value
            delta = max(delta, abs(old_value - new_value))            

        if delta == 0:
            # continue until the value function stabilizes for the policy
            break

def policy_improvement(policy, value)-> bool:
    """update the policy to be greedy with respect to the current value function"""
    m,n = policy.shape
    policy_stable = True
    for ij in range(m*n):
        i, j = divmod(ij, n)
        old_action = policy[i, j]
        # enumerate the possible actions, and check thier values
        down_value = value[i + 1, j] if i + 1 < m else -np.inf
        right_value = value[i, j + 1] if j + 1 < n else -np.inf
        best_action = 0 if down_value > right_value else 1
        #update policy to reflect the best action
        policy[i, j] = best_action
        if old_action != best_action:
            policy_stable = False
    return policy_stable

def policy_iteration(matrix, policy, value):
    while True:
        # update the value function to reflect the current policy
        policy_evaluation(matrix, policy, value)
        # Imporve the policy to be greedy with respect to the current value function
        if policy_improvement(policy, value):
            # stop once the policy is stable
            break




if __name__ == "__main__":

    m,n = 15, 20 # matrix dim
    value = np.random.random(size=(m,n))  
    policy = np.random.randint(0,2, size=(m,n)) # 0 for down, 1 for right
    matrix = np.random.choice([-16, 0, 1, 18], size=(m,n),p=[0.05,0.4,0.5, 0.05])

    policy_iteration(matrix, value, policy)

    plot_2d_heatmap(matrix, "Environment")
    plot_policy(policy, "Policy")
    plot_2d_heatmap(value, "Value Function")
    show()


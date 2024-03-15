import numpy as np
import random
import matplotlib.pyplot as plt


def policy(state: str, epoch: int, q:dict):
    """Epsilon-greedy policy based on Q-values."""

    eps = epsilon * (eps_decay ** epoch)
    
    # number of possible actions for the current state
    number_of_possible_actions = len(transitions[state])
    
    if np.random.random() < eps:
        print("Epsilon event, exploring...",eps)
        return np.random.choice([i for i in range(number_of_possible_actions)])

    action_values = [q.get((state, action), 0) for action in range(number_of_possible_actions)]
    max_value = max(action_values)
    best_actions = [action for action, value in enumerate(action_values) if value == max_value]
    best_action = np.random.choice(best_actions)
    
    return best_action


def get_reward(state:str):
    if state == "a":
        raise ValueError("a should not be passsed as param as it's the starting state")
    if state == 'b':
        return 0
    if state == 'terminal':
        return 0
    # state == c
    return np.random.normal(-0.1, 1)


def calculate_td_update(state, action, new_state, reward, alpha):
    """
    Calculates the TD update for a given state-action pair.
    
    :param state: The current state.
    :param action: The action taken in the current state.
    :param new_state: The state reached after taking the action.
    :param reward: The reward received after taking the action.
    :param alpha: The learning rate.
    :return: The value to update the Q-value by.
    """
    # Current Q-value estimation
    current_q = q.get((state, action), 0)
    # Maximum Q-value for the next state
    max_next_q = max_aq(new_state)
    # TD Target
    target = reward + gamma * max_next_q
    # TD Error
    td_error = target - current_q
    # TD Update
    update = alpha * td_error
    update += current_q
    return update


def max_aq(state: str)-> float:
    """
    max a for Q(s,a)
    returns max value for a state if greedy action is taken
    """
    if state == 'terminal':
        return 0 
    q_values = [q.get((state, action), 0) for action in range(len(transitions[state]))]
    return max(q_values)


def simulate(epoch:int):

    state = 'a'
    while state != 'terminal':
        action = policy(state, epoch, q)
        new_state = transitions[state][action]
        reward = get_reward(new_state)

        update = calculate_td_update(state, action, new_state, reward, lr)
        q[(state,action)] = update

        state = new_state

if __name__ == "__main__":
    ### PARAMS ###
    lr = 0.1
    epsilon = 0.1
    eps_decay = 0.995
    number_of_c_states = 10
    gamma = 1
    epochs = 450

    ### MDP ###
    transitions = {
        "a": ["terminal", "b"],
        "b": ["c"+str(i) for i in range(number_of_c_states)]
    }
    for i in range(number_of_c_states):
        transitions[f"c{i}"] = ["terminal"]


    a_to_b = []
    q = {}

    for epoch in range(epochs):
        simulate(epoch)

        # Q(State=a, Action=b)
        a_b = q.get(('a', 1),0)
        a_to_b.append(a_b)
        print(f"a -> b value: {a_b}")



    min_value = min(a_to_b)
    shift = abs(min_value) + 1
    a_to_b = [np.log(i+shift) for i in a_to_b]
    plt.plot(a_to_b)
    plt.xlabel('epochs')
    plt.ylabel("Q value of moving to state B")
    plt.show()
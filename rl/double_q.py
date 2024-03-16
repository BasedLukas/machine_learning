import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple


def argmax_a(state: str, q: Dict) -> int:
    """
    Find the action that maximizes the Q-value in a given state.

    Args:
    - state: The current state.
    - q: The Q-value dictionary.

    Returns:
    - An integer representing the action that maximizes the Q-value.
    """
    if state == 'terminal':
        return 0
    actions = range(len(transitions[state]))
    values = [q.get((state, action), 0) for action in actions]
    max_value = max(values)
    best_actions = [action for action, value in enumerate(values) if value == max_value]
    return np.random.choice(best_actions)


def max_a(state: str, q: Dict) -> float:
    """
    Find the maximum Q-value for any action in a given state.

    Args:
    - state: The current state.
    - q: The Q-value dictionary.

    Returns:
    - The maximum Q-value for the given state.
    """
    if state == 'terminal':
        return 0 
    return max([q.get((state, action), 0) for action in range(len(transitions[state]))])


def policy(
        state: str, 
        epoch: int, 
        q1: Dict, 
        q2: Optional[Dict] = None
        ) -> int:
    """
    Select an action based on the epsilon-greedy policy derived from Q-values.

    Args:
    - state: The current state in the environment.
    - epoch: The current epoch of training, used to decay epsilon.
    - q1: The primary Q-value dictionary.
    - q2: An optional secondary Q-value dictionary for double Q-learning.

    Returns:
    - An integer representing the chosen action.
    """

    eps = epsilon * (eps_decay ** epoch)
    number_of_possible_actions = len(transitions[state])

    #exploration
    if np.random.random() < eps:
        return np.random.choice(range(number_of_possible_actions))

    action1 = argmax_a(state, q1)
    if not q2:
        return action1
    
    action2 = argmax_a(state, q2)
    return np.random.choice([action1, action2])


def get_reward(state: str) -> float:
    """
    Returns the reward for transitioning into a given state.

    Args:
    - state: The state transitioned into.

    Returns:
    - A float representing the reward for that transition.

    Raises:
    - ValueError: If an invalid state is provided.
    """
    if state == "a":
        raise ValueError("a should not be passed as a param as it's the starting state")
    if state == 'b' or state == 'terminal':
        return 0
    if 'c' in state:
        return np.random.normal(-0.1, 1)
    raise ValueError(f"state: {state} not recognized")


def q_update(
        state: str, 
        action: int, 
        new_state: str, 
        reward: float, 
        alpha: float, 
        q: Dict
    ) -> None:
    """
    In-place update of Q-values for Q-learning.

    Args:
        state: The current state.
        action: The action taken in the current state.
        new_state: The state reached after taking the action.
        reward: The reward received after taking the action.
        alpha: The learning rate.
        q: The Q-values dictionary.
    """
    current_q = q.get((state, action), 0)  # Current Q-value estimation
    max_next = max_a(new_state, q)  # Maximum Q-value for the next state
    target = reward + gamma * max_next  # TD Target
    td_error = target - current_q  # TD Error
    update = alpha * td_error  # TD Update
    q[(state, action)] = current_q + update


def double_q_update(
        state: str, 
        action: int, 
        new_state: str, 
        reward: float, 
        alpha: float, 
        q1: Dict, 
        q2: Dict
    ) -> None:
    """
    In-place update of Q-values for Double Q-learning.

    Args:
        state: The current state.
        action: The action taken in the current state.
        new_state: The state reached after taking the action.
        reward: The reward received after taking the action.
        alpha: The learning rate.
        q1: The first Q-values dictionary.
        q2: The second Q-values dictionary.
    """
    qs = [q1, q2]  # List of Q dictionaries
    random.shuffle(qs)  # Randomly shuffle to choose one for updating
    qa, qb = qs  # qa is the Q to update, qb is used for target calculation

    current_q = qa.get((state, action), 0)  # Current Q-value estimation
    best_action = argmax_a(new_state, qa)  # Best action based on qa
    target = reward + gamma * qb.get((new_state, best_action), 0)  # TD Target using qb
    error = target - current_q  # TD Error
    update = alpha * error  # TD Update
    qa[(state, action)] = current_q + update


def simulate(
        epoch: int, 
        q: Dict, 
        q2: Optional[Dict] = None
    ) -> None:
    """
    Simulate an epoch of the agent's interaction with the environment, updating Q-values based on observed transitions.

    Args:
        epoch: The current epoch of the simulation.
        q: The Q-values dictionary for Q-learning or the primary Q-values dictionary for Double Q-learning.
        q2: The secondary Q-values dictionary for Double Q-learning, if applicable.
    """
    double = q2 is not None
    state = 'a'
    while state != 'terminal':
        if double:
            action = policy(state, epoch, q, q2)
        else:
            action = policy(state, epoch, q) 
        new_state = transitions[state][action]
        reward = get_reward(new_state)
        
        if double:
            double_q_update(
                state=state,
                action=action,
                new_state=new_state,
                reward=reward,
                alpha=lr,
                q1=q,
                q2=q2
                )
        else:
            q_update(state, action, new_state, reward, lr, q)

        state = new_state



if __name__ == "__main__":
    ### PARAMS ###
    lr = 0.001
    epsilon = 0.1
    eps_decay = 0.995
    number_of_c_states = 10
    gamma = 1
    epochs = 1000

    ### MDP ###
    transitions = {
        "a": ["terminal", "b"],
        "b": ["c"+str(i) for i in range(number_of_c_states)]
    }
    for i in range(number_of_c_states):
        transitions[f"c{i}"] = ["terminal"]

    # Track the evolution of Q-values for a specific action over epochs
    normal = []  # For standard Q-learning
    double = []  # For Double Q-learning
    
    # Q-values dictionaries: key=(state,action)
    q:  Dict[Tuple[str, int], float] = {}
    q1: Dict[Tuple[str, int], float] = {}
    q2: Dict[Tuple[str, int], float] = {}

    for epoch in range(epochs):
        simulate(epoch, q) # normal
        simulate(epoch, q1, q2) # double

        normal_q_value: float = q.get(('a', 1), 0)
        double_q_value: float = (q1.get(('a', 1), 0) + q2.get(('a', 1), 0)) / 2
        normal.append(normal_q_value)
        double.append(double_q_value)

    plt.plot(normal, label='Standard Q-Learning')
    plt.plot(double, label='Double Q-Learning')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Value of A -> B')
    plt.title('Comparison of Q-Learning and Double Q-Learning')
    plt.show()
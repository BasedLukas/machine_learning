import numpy as np
import random
import matplotlib.pyplot as plt


def policy(state: str, epoch: int, q1:dict, q2={})-> int:
    """
    Epsilon-greedy policy based on Q-values.
    Q2: Optional 2nd Q dict for double Q learning
    Returns action:int
    """
    # default value for q2, to prevent interference when not used
    # when used we want default to be 0
    default = -float('inf') if q2 == {} else 0
    # epsilon decay over time
    eps = epsilon * (eps_decay ** epoch)
    
    # number of possible actions for the current state
    number_of_possible_actions = len(transitions[state])
    
    # exploration
    if np.random.random() < eps:
        print("Epsilon event, exploring...")
        return np.random.choice([i for i in range(number_of_possible_actions)])

    # get values for all possible actions in current state
    action_values1 = [q1.get((state, action), 0) for action in range(number_of_possible_actions)]
    action_values2 = [q2.get((state, action), default) for action in range(number_of_possible_actions)]

    # there may be multiple best actions
    max_value = max(action_values1 + action_values2)
    best_actions = [
        action for action, (v1, v2) in enumerate(zip(action_values1, action_values2))
        if v1 == max_value or v2 == max_value
        ]
    
    # sorandomly select from the best scoring actions
    return np.random.choice(best_actions)
    

def get_reward(state:str)-> float:
    """
    returns reward for transitioning into a state
    state: the state you just transitioned into.
    """
    if state == "a":
        raise ValueError("a should not be passsed as a param as it's the starting state")
    if state == 'b':
        return 0
    if state == 'terminal':
        return 0
    if 'c' in state:
        return np.random.normal(-0.1, 1)
    raise ValueError(f"state: {state} not recognized")


def max_aq(state: str, q:dict)-> float:
    """
    max a for Q(s,a)
    returns max value for a state if greedy action is taken
    """
    if state == 'terminal':
        return 0 
    q_values = [q.get((state, action), 0) for action in range(len(transitions[state]))]
    return max(q_values)


def argmax_a(state:str, q:dict):
    if state == 'terminal':
        return 0
    n_of_actions_for_state = len(transitions[state])
    values = [q.get((state, action), 0) for action in range(n_of_actions_for_state)]
    indices = [i for i, x in enumerate(values) if x == max(values)]
    return np.random.choice(indices)


def td_update(state, action, new_state, reward, alpha, q):
    """
    In place update of Q.
    
    :param state: The current state.
    :param action: The action taken in the current state.
    :param new_state: The state reached after taking the action.
    :param reward: The reward received after taking the action.
    :param alpha: The learning rate.
    :param q: q values dict

    """
    # Current Q-value estimation
    current_q = q.get((state, action), 0)
    # Maximum Q-value for the next state
    max_next = max_aq(new_state, q)
    # TD Target
    target = reward + gamma * max_next
    # TD Error
    td_error = target - current_q
    # TD Update
    update = alpha * td_error
    q[(state, action)] = current_q + update


def double_q_update(state, action, new_state, reward, alpha, q1, q2):
    # randomly select one q to be updated
    qs = [q1,q2]
    random.shuffle(qs)
    qa, qb = qs[0], qs[1]

    current_q = qa.get((state, action), 0)
    best_action = argmax_a(new_state, qa)
    target = reward + gamma * qb.get((new_state, best_action), 0)
    error = target - current_q
    update = alpha * error
    qa[(state, action)] = update + current_q



def simulate(epoch:int, q:dict, q2=None):
    double = isinstance(q2, dict)
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
            td_update(state, action, new_state, reward, lr, q)

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

    normal = []
    double = []
    
    q, q1, q2 = {},{},{}

    for epoch in range(epochs):
        # normal
        simulate(epoch, q)
        #double Q
        simulate(epoch, q1, q2)

        normal.append(q.get(('a',1),0))
        d1 = q1.get(('a',1), 0)
        d2 = q2.get(('a',1), 0)
        double.append((d1+d2)/2)

    plt.plot(normal, label='normal')
    plt.plot(double, label='double')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel("Q value of moving to state B")
    plt.show()
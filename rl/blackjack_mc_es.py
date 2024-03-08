from games.blackjack import Blackjack
from utils.plotting import plot_3d_heatmap, show, plot_2d_heatmap

import numpy as np
from collections import defaultdict


player_policy_ace = np.zeros(shape=(10, 10))
player_policy_no_ace = np.zeros(shape=(10, 10))
q_ace = np.zeros(shape=(10, 10, 2)) # dealer, player, action (0,1)
q_no_ace = np.zeros(shape=(10, 10, 2))
returns = defaultdict(float)
returns_count = defaultdict(float)


def player_policy_factory():
    """
    Using a closure to store the first state, so that we can have an exploring start.
    They don't let me write fun code like this on the job :(
    """
    first_state = [True] #mutable!

    def policy(state)->int:
        """return 0 for stick, 1 for hit."""
        dealer, player, ace = state
        # clip player sum to 21
        player = min(player, 21)
        # on first state, randomly select a player action, so that we have an exploring start.
        if first_state[0]:
            first_state[0] = False
            return np.random.choice([0, 1])
        
        if ace:
            return player_policy_ace[dealer-1, player-12]
        return player_policy_no_ace[dealer-1, player-12]

    return policy


for j in range(1_000_000):
    #policy that randiomly selects first action, thereafter follows the policy
    player_policy = player_policy_factory() 
    game = Blackjack()

    states, actions, result = game.play(player_policy)

    for k,(state,action) in enumerate(zip(states, actions)):
        dealer, player, ace = state
        
        if player > 21: # no need for policy or q update
            continue

        state_action = (state, action)
        # compute the mean return for this state action pair
        old_return = returns[state_action]
        returns_count[state_action] += 1
        new_return = (1/returns_count[state_action]) * (result - old_return)
        current_return = new_return + old_return
        returns[state_action] = current_return

        #update q
        if ace:
            q_ace[dealer-1, player-12, action] = current_return
            best_action = np.argmax(q_ace[dealer-1, player-12])
            player_policy_ace[dealer-1, player-12] = best_action
        else:
            q_no_ace[dealer-1, player-12, action] = current_return
            best_action = np.argmax(q_no_ace[dealer-1, player-12])
            player_policy_no_ace[dealer-1, player-12] = best_action
        
# create v from q
v_ace = np.max(q_ace, axis=2)
v_no_ace = np.max(q_no_ace, axis=2)


plot_2d_heatmap(
    player_policy_no_ace, 
    'policy,no ace', 
    ytick_labels=list(range(1, 11)),
    xtick_labels=list(range(12, 22))
    )

plot_2d_heatmap(
    player_policy_ace, 
    'policy, ace',
    ytick_labels=list(range(1, 11)),
    xtick_labels=list(range(12, 22))
    )

plot_2d_heatmap(
    v_no_ace, 
    'value, no ace', 
    ytick_labels=list(range(1, 11)),
    xtick_labels=list(range(12, 22))
    )   
plot_2d_heatmap(
    v_ace, 
    'value, ace', 
    ytick_labels=list(range(1, 11)),
    xtick_labels=list(range(12, 22))
    )

show()
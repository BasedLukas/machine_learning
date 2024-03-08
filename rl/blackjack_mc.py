from games.blackjack import Blackjack
from utils.plotting import plot_3d_heatmap, show

import numpy as np
from collections import defaultdict


def player_policy(state:tuple)->bool:
    """sample player policy; draw if sum < 19"""
    dealer, current_sum, ace = state
    total = current_sum
    total2 = current_sum + 10 if ace else total 
    if total < 19 and total2 < 19:
        return True
    return False

def simulate(episodes):
    value_ace = np.zeros((10, 10))
    value_no_ace = np.zeros((10, 10))
    value = defaultdict(list)

    for _ in range(episodes):
        game = Blackjack()
        states, actions, result = game.play(player_policy)

        for state in states:
            dealer, player, ace = state
            if player <= 21: # ignore states where player sum exceeds 21 as we know he'll lose
                value[state].append(result)

    for dealer, player, ace in value.keys():
        results_mean = np.mean(value[(dealer, player, ace)])
        if ace:
            value_ace[dealer-1, player-12] = results_mean
        else:
            value_no_ace[dealer-1, player-12] = results_mean


    plot_3d_heatmap(
        value_no_ace, 
        x_values=[1,2,3,4,5,6,7,8,9,10], 
        y_values=[12,13,14,15,16,17,18,19,20,21],
        x_label='Dealer Card',
        y_label='Player Sum',
        title=f"No Ace: {episodes} episodes"
        )
    plot_3d_heatmap(
        value_ace, 
        x_values=[1,2,3,4,5,6,7,8,9,10], 
        y_values=[12,13,14,15,16,17,18,19,20,21],
        x_label='Dealer Card',
        y_label='Player Sum',
        title=f"Ace: {episodes} episodes"
        )


if __name__ == "__main__":
    simulate(10_000)
    simulate(500_000)
    show()
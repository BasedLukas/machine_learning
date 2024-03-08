
from games.blackjack import Blackjack





def test_blackjack():

    def player_policy(state:tuple)->bool:
        """sample player policy; draw if sum < 19"""
        dealer, current_sum, ace = state
        total = current_sum
        total2 = current_sum + 10 if ace else total 
        if total < 19 and total2 < 19:
            return True
        return False

    game = Blackjack()
    states, actions, result = game.play(player_policy)
    assert result in [-1, 0, 1], "Invalid game outcome"
    assert len(states) == len(actions) ,"each state should be followed by a corresponding action but length of states and actions do not match"
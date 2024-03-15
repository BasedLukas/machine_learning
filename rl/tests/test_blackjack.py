
from games.blackjack import Blackjack


def player_policy(state:tuple)->bool:
    """sample player policy; draw if sum < 19"""
    dealer, current_sum, ace = state
    total = current_sum
    total2 = current_sum + 10 if ace else total 
    if total < 19 and total2 < 19:
        return True
    return False

def always_hit(state):
    return True

def test_blackjack():
    game = Blackjack()
    states, actions, result = game.play(player_policy)
    assert result in [-1, 0, 1], "Invalid game outcome"
    assert len(states) == len(actions) ,"each state should be followed by a corresponding action but length of states and actions do not match"

def test_outcomes():
    game = Blackjack()
    game.player = [1,10] #natural
    game.dealer = [2,5]
    s,a,r = game.play(player_policy)
    assert r == 1, 'player should win due to natural'

    game = Blackjack()
    game.player = [1,10] #natural
    game.dealer = [1,10]
    s,a,r = game.play(player_policy)
    assert r == 0, 'should be drawn due to both natural'

    game = Blackjack()
    game.player = [10,10, 5] #loss
    game.dealer = [1, 2]
    s,a,r = game.play(player_policy)
    assert r == -1, 'player should lose'

    game = Blackjack()
    s,a,r = game.play(always_hit)
    assert r == -1, 'player should lose, because hit until bust'

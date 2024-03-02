import numpy as np

from types import SimpleNamespace as sn
from collections import defaultdict


class Blackjack:

    def __init__(self):
        self.player = [self.draw(), self.draw()]
        self.dealer = [self.draw(), self.draw()]
        self._init_player()

    def draw(self) -> int:
        """draw from deck with replacement. 1 == 1 or 11"""
        cards = [1,2,3,4,5,6,7,8,9,10]# j q k
        probs = [1/13]*9 + [4/13]
        result = np.random.choice(cards, p=probs)
        return result

    def _init_player(self):
        """make obvious initial moves"""
        current_sum = sum(self.player)
        ace = 1 in self.player

        if (current_sum < 12 and not ace) or (ace and current_sum + 10 < 12):
            self.player.append(self.draw())
            self._init_player()

    def dealer_policy(self):
        """make dealer moves based on fixed policy"""
        current_sum = sum(self.dealer)
        current_ace_sum = current_sum + 10 if 1 in self.dealer else current_sum

        if current_sum < 17 and current_ace_sum < 17:
            # print("dealer draws a card")
            self.dealer.append(self.draw())
            self.dealer_policy()
        
    def state(self)->tuple:
        dealer_card = self.dealer[0]
        ace = 1 in self.player
        current_sum = sum(self.player)
        return (dealer_card, current_sum, ace)
    
    def play(self, player_policy:callable)-> int:
        # """returns 1 for win, 0 for draw, -1 for loss"""
        # print(f"Game init. player: {self.player} dealer: {self.dealer}")
        # player has natural
        if 1 in self.player and 10 in self.player:
            # print(f"player 1 natural")
            # if both natural draw, else player 1 wins
            if 1 in self.dealer and 10 in self.dealer:
                # print("dealer also natural")
                return 0
            return 1
        
        # player turn
        player_policy(self)
        # dealer turn
        self.dealer_policy()

        # eval winner
        player_sum = sum(self.player)
        dealer_sum = sum(self.dealer)
        player_aces = self.player.count(1)
        dealer_aces = self.dealer.count(1)

        player_off = abs(21 -player_sum)
        dealer_off = abs(21 - dealer_sum)
        # print("Game Over.")
        # print(f"player {self.player} \ndealer {self.dealer}")
        
        if player_aces:
            player_off = min(player_off, abs(21- (player_sum + 10)))
            if player_aces > 1:
                player_off = min(player_off, abs(21- (player_sum + 20)))

        if dealer_aces:
            dealer_off = min(dealer_off, abs(21 - (dealer_sum + 10)))
            if dealer_aces > 1:
                dealer_off = min(dealer_off, abs(21 - (dealer_sum + 20)))

        # print(f"player off {player_off} and dealer off {dealer_off}")
        if player_off < dealer_off:
            return 1
        if dealer_off < player_off:
            return -1
        return 0


def player_policy(game:Blackjack):
    ace = 1 in game.player
    total = sum(game.player)
    total2 = total + 10 if ace else total
    # print("called player policy", game.player)  
    if total < 19 and total2 < 19:
        # print("player elects to draw a card")
        game.player.append(game.draw())
        player_policy(game)
    

wins = 0  
for _ in range(100_000):
    value = defaultdict(float)
    game = Blackjack()
    state = game.state()
    result = game.play(player_policy)
    wins += result
    value[state] += result


print(wins)
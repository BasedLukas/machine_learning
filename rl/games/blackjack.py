import numpy as np
from typing import List, Tuple



class Blackjack:
    """
    Game of blackjack, as defined by Sutton and Barto. 
    Cards are drawn from an infinite deck with replacement.
    Ace is 1 or 11, and is represented as 1 in the game.
    Face cards are valued at 10.
    We initialize the player and the dealer with two cards each.
    The player can see the dealer's first card.
    The player automatically draws a card until the sum of his cards is 12 or more, regardless of policy.
    The state of the game is represented as (dealer_card:int, player_sum:int, ace:bool)
    The dealer follows a fixed policy of drawing until sum is 17 or more.
    Thus the player must follow a policy that takes in the state and returns a boolean indicating whether to draw a card or not.
    The winner is decided based on which player is closest to 21 without going over.
    """

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
        """return current state of the game: (dealer_card:int, player_sum:int, ace:bool)"""
        dealer_card = self.dealer[0]
        ace = 1 in self.player
        current_sum = sum(self.player)
        return (dealer_card, current_sum, ace)
    
    def play(self, player_policy:callable)-> Tuple[List, List, int]:
        """
        player_policy: callable, takes state as input and returns bool indicating whether to draw a card or not
        returns list of states, a list of actions and game outcome
        1 for win, 0 for tie, -1 for loss
        """
        states =[]
        actions = []
        
        # check for natural
        if 1 in self.player and 10 in self.player:
            # if both natural tie, else player 1 wins
            if 1 in self.dealer and 10 in self.dealer:
                return [self.state()],[0], 0
            return [self.state()],[0], 1
        
        # player turn
        while True:
            state = self.state()
            draw_card = player_policy(state)
            actions.append(int(draw_card))
            states.append(state)
            if draw_card:
                self.player.append(self.draw())  
                # break to avoid infinite loop on bad policies
                if sum(self.player) > 21:
                    break   
            else:
                break

        # dealer turn
        self.dealer_policy()

        # eval winner
        player_sum = sum(self.player)
        dealer_sum = sum(self.dealer)
        player_ace = 1 in self.player
        dealer_ace = 1 in self.dealer

        
        #check if either player exceeded 21
        # if player exceeeds, dealer automatically wins before even drawing. Thus the ordering of the checks
        if player_sum > 21:
            return states, actions, -1
        if dealer_sum > 21:
            return states, actions, 1
        
        player_sum += 10 if player_ace and player_sum + 10 <= 21 else 0
        dealer_sum += 10 if dealer_ace and dealer_sum + 10 <= 21 else 0

        if player_sum > dealer_sum:
            return states, actions, 1
        if player_sum == dealer_sum:
            return states, actions, 0
        return states, actions, -1

        


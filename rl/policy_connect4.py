import torch
import numpy as np
import math
from collections import defaultdict

from games.connect_4 import Game, Status, ROWS, COLS

class Policy(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.fc1 = torch.nn.Linear(obs_dim, 64)
        self.fc2 = torch.nn.Linear(64, act_dim)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self, board_state):
        x = self.relu(self.fc1(board_state))
        x = self.fc2(x)
        return self.softmax(x)
    
class BadPolicy(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc1 = torch.nn.Linear(obs_dim, act_dim)
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self, board_state):
        x = self.dropout(board_state)
        x = self.fc1(x)
        return self.softmax(x)
    

lr = 0.001

player1 = Policy(ROWS*COLS, 7)
player2 = BadPolicy(ROWS*COLS, 7)
optimizer1 = torch.optim.Adam(player1.parameters(), lr=lr)
optimizer2 = torch.optim.Adam(player2.parameters(), lr=lr)


def play_game(player1, player2):
    game = Game()
    players = {1: player1, 2: player2}
    actions = defaultdict(list)

    while game.status.name == 'ongoing':
        current_player = players[game.player]
        board_state = torch.tensor(game.board.flatten(), dtype=torch.float32)
        probs = current_player(board_state)
        action = torch.multinomial(probs, 1)
        move = game.move(action.item())
        actions[game.player].append(probs[action])

        if not move:
            game.status = Status.player1_wins if game.player == 2 else Status.player2_wins
            # print('Player', game.player, 'tried to move in column', action, 'but that column is full.')
        else:
            pass
        #     print('Player', game.player, 'moved in column', action)
        #     print(game)
        # print()

    return game.status.name, actions


def update_models(optimizer1, optimizer2, winner, actions):
    """
    Update the models based on the game outcome.
    Rewards: -1 for a loss, 1 for a win, 0.5 for a draw for player2, and -0.5 for a draw for player1.
    """
    # Define the rewards
    if winner == 'player1_wins':
        reward1, reward2 = 1, -1
    elif winner == 'player2_wins':
        reward1, reward2 = -1, 1
    else:  # Draw
        reward1, reward2 = -0.5, 0.5

    optimizer1.zero_grad()
    optimizer2.zero_grad()
    # Update player1
    for action in actions[1]:
        log_prob = torch.log(action)
        loss = -log_prob * reward1
        loss.backward()

    # Update player2
    for action in actions[2]:
        log_prob = torch.log(action)
        loss = -log_prob * reward2
        loss.backward()

    optimizer1.step()
    optimizer2.step()

iters = 1000_000
wins = 0 
draws = 0
losses = 0
       
for i in range(iters):            
    winner, actions = play_game(player1, player2)
    update_models(optimizer1, optimizer2, winner, actions)
    # print('Winner:', winner)
    if winner == 'player1_wins':
        wins += 1
    elif winner == 'draw':
        draws += 1
    else:
        losses += 1

print(f'Player 1 wins {wins} losses {losses} draws {draws}') 


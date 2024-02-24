from __future__ import annotations
import numpy as np
from enum import Enum

COLS = 7
ROWS = 6

class Status(Enum):
    ongoing = 0
    player1_wins = 1
    player2_wins = 2
    draw = 3



class Game:
    def __init__(self) -> None:
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.player = 1
        self.status = Status.ongoing

    def __str__(self) -> str:
        board_str = '\n'.join(' '.join(str(self.board[row, col]) for col in range(COLS)) for row in range(ROWS))
        return f'Player {self.player} to move\nStatus: {self.status.name}\n{board_str}'

    def _update_status(self) -> Status:
        """updates and returns the status of the game."""
        # Check rows and columns for four of a kind
        for axis in [0, 1]:
            for line in np.apply_along_axis(self._four_line, axis, self.board):
                if line != 0:
                    self.status = Status.player1_wins if line == 1 else Status.player2_wins
                    return self.status

        # Check diagonals for four of a kind
        diag_winner = self._four_diagonal(self.board)
        if diag_winner != 0:
            self.status = Status.player1_wins if diag_winner == 1 else Status.player2_wins
            return self.status

        # Check if board is full
        if np.all(self.board != 0):
            self.status = Status.draw
            return self.status

        return self.status

    def move(self, col: int) -> bool:
        """Makes a move for the current player in the given column."""
        if self.status != Status.ongoing:
            raise ValueError('Game is already over.')
        if col < 0 or col >= COLS:
            return False
        if self.board[0, col] != 0:
            return False
        
        # Find the first empty row in the given column
        row = ROWS - 1
        while self.board[row, col] != 0:
            row -= 1

        # Make the move
        self.board[row, col] = self.player

        # set the status and switch players
        self._update_status()
        self.player = 2 if self.player == 1 else 1
        return True

    def play(self):
        """Plays a game of connect 4."""
        while self.status == Status.ongoing:
            print(self)
            col = int(input("Enter a column: "))
            # try to move until a valid move is made
            while not self.move(col):
                col = int(input("Enter a column: "))

        print("Game over!") 
        print(self)

    @staticmethod
    def _four_diagonal(board) -> int:
        for offset in range(-ROWS + 4, COLS - 3):
            diag1 = np.diagonal(board, offset=offset)
            diag_winner = Game._four_line(diag1)
            if diag_winner != 0:
                return diag_winner

            diag2 = np.diagonal(np.fliplr(board), offset=offset)
            diag_winner = Game._four_line(diag2)
            if diag_winner != 0:
                return diag_winner

        return 0

    @staticmethod
    def _four_line(line) -> int:
        for i in range(len(line) - 3):
            if line[i] != 0 and np.all(line[i] == line[i+1:i+4]):
                return line[i]
        return 0
    

if __name__ == '__main__':
    # to programatically play a game
    game = Game()
    game.move(0)
    game.move(0)
    game.move(1)
    print(game)

    game = Game()
    game.play()



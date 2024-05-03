#two_player_env.py

import numpy as np
import random

class ConnectFour:
    def __init__(self, rows=6, columns=7, win_length=4):
        self.rows = rows
        self.columns = columns
        self.win_length = win_length
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.current_player = 1  # for now, gets updated on reset
        self.done = False
        self.winner = None

    def get_player_symbol(self, current_player):
        """
        Returns the symbol for the current player.
        Player 1 is always 'X' and Player 2 is always 'O'.
        """
        if current_player == 1:
            return 'X'
        else:
            return 'O'

    def reset(self):
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.current_player = random.choice([1, 2])  # Randomly choose the starting player
        self.done = False
        self.winner = None
        return self.board.flatten(), self.current_player

    def step(self, action):
        """ Place a piece in the chosen column. """
        if self.done:
            raise ValueError("Game is over.")
        if self.board[0, action] != 0:
            raise ValueError("Column is full.")

        # Find the lowest empty spot in the column
        row = max(np.where(self.board[:, action] == 0)[0])
        self.board[row, action] = self.current_player

        # Check game status
        if self.check_win(player=self.current_player):
            self.done = True
            self.winner = self.current_player
            reward = 1.5  # Win
        elif np.all(self.board != 0):
            self.done = True
            self.winner = None  # Draw
            reward = 0.3  # Draw is less rewarding than a win
        elif self.winner != self.current_player:
            reward = -1.0
        else:
            reward = 0  # Game continues, no intermediate rewards

        # Prepare for next step
        self.current_player = 3 - self.current_player  # Switch player between 1 and 2

        return self.board.flatten(), reward, self.done, self.current_player
    
    

    def check_win(self, player):
        # Horizontal, vertical, and diagonal checks
        for c in range(self.columns - self.win_length + 1):
            for r in range(self.rows - self.win_length + 1):
                if np.all(self.board[r:r+self.win_length, c] == player):
                    return True
                if np.all(self.board[r, c:c+self.win_length] == player):
                    return True
                if np.all(np.diag(self.board[r:r+self.win_length, c:c+self.win_length]) == player):
                    return True
                if np.all(np.diag(np.fliplr(self.board[r:r+self.win_length, c:c+self.win_length])) == player):
                    return True

        # Additional checks for vertical and horizontal without slicing
        for r in range(self.rows):
            for c in range(self.columns - self.win_length + 1):
                if np.all(self.board[r, c:c+self.win_length] == player):
                    return True

        for c in range(self.columns):
            for r in range(self.rows - self.win_length + 1):
                if np.all(self.board[r:r+self.win_length, c] == player):
                    return True

        return False
    
    def get_valid_actions(self):
        return [col for col in range(self.columns) if self.board[0, col] == 0]


    def render(self):
        symbols = {0: " . ", 1: " X ", 2: " O "}
        # Create a copy of the board for display purposes
        display_board = self.board.copy()
        
        # Function to highlight winning pieces
        def highlight_winning_pieces():
            # Check all possible directions for a win
            directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # horizontal, vertical, diagonal, anti-diagonal
            for r in range(self.rows):
                for c in range(self.columns):
                    current_player = self.board[r][c]
                    if current_player != 0:
                        for dr, dc in directions:
                            # Check if there are four in a row starting from (r, c)
                            if all(0 <= r + i * dr < self.rows and 0 <= c + i * dc < self.columns and
                                self.board[r + i * dr][c + i * dc] == current_player for i in range(self.win_length)):
                                # Highlight all pieces in this sequence
                                for i in range(self.win_length):
                                    rr, cc = r + i * dr, c + i * dc
                                    display_board[rr][cc] = -current_player  # Use negative to denote winning pieces
                                return

        if self.winner is not None:
            highlight_winning_pieces()

        # Update the symbols dictionary to include color for winning pieces
        symbols[-1] = "\033[94m X \033[0m"  # Blue "X"
        symbols[-2] = "\033[94m O \033[0m"  # Blue "O"

        print("  " + "   ".join(map(str, range(self.columns))))
        for row in display_board:
            print(" |" + "|".join(symbols[cell] for cell in row) + "|")
        print()

        return self.winner
        

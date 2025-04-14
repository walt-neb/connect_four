# filename: two_player_env.py
"""
Connect Four Environment for Two AI Agents.

This module implements the game logic for Connect Four, designed to be used
in a reinforcement learning setting. It now uses the render function that
highlights the winning line.
"""

import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

class TwoPlayerConnectFourEnv():
    """
    A Connect Four environment for two players (agents).
    """
    def __init__(self, rows=6, columns=7, win_length=4, writer: SummaryWriter = None):
        self.rows = rows
        self.columns = columns
        self.win_length = win_length
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        self.total_steps = 0
        self.writer = writer
        self.reset()

    def get_player_symbol(self, player_id):
        return 'X' if player_id == 1 else 'O'

    def reset(self):
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.current_player = random.choice([1, 2])
        self.done = False
        self.winner = None
        return self.board, self.current_player

    def step(self, action):
        """
        Executes a player's action (dropping a piece in a column).
        (Removed the internal debug print)
        """
        if self.done:
            print("Warning: step() called after game was done.")
            return self.board, 0.0, self.done, self.current_player

        if not self.is_valid_action(action):
            raise ValueError(f"Illegal move attempted: Column {action}")

        row = np.max(np.where(self.board[:, action] == 0))
        placing_player = self.current_player
        self.board[row, action] = placing_player

        # Check win/draw conditions
        if self.check_win(placing_player):
            reward = 1.0
            self.done = True
            self.winner = placing_player
        elif np.all(self.board[0, :] != 0):
            reward = 0.0
            self.done = True
            self.winner = None
        else:
            reward = 0.0
            self.done = False
            self.winner = None

        # Switch player for next turn
        self.current_player = 3 - placing_player # Determine next player
        self.total_steps += 1

        # Return state, reward for action, done flag, and NEXT player
        return self.board, reward, self.done, self.current_player

    def is_valid_action(self, action):
        return 0 <= action < self.columns and self.board[0, action] == 0

    def get_valid_actions(self):
        return [col for col in range(self.columns) if self.is_valid_action(col)]

    def check_win(self, player):
        # Check horizontal
        for c in range(self.columns - self.win_length + 1):
            for r in range(self.rows):
                if np.all(self.board[r, c:c + self.win_length] == player): return True
        # Check vertical
        for c in range(self.columns):
            for r in range(self.rows - self.win_length + 1):
                if np.all(self.board[r:r + self.win_length, c] == player): return True
        # Check positive diagonal
        for c in range(self.columns - self.win_length + 1):
            for r in range(self.rows - self.win_length + 1):
                if np.all(np.diag(self.board[r:r + self.win_length, c:c + self.win_length]) == player): return True
        # Check negative diagonal
        for c in range(self.columns - self.win_length + 1):
            for r in range(self.win_length - 1, self.rows):
                 subgrid = self.board[r - self.win_length + 1 : r + 1, c : c + self.win_length]
                 if np.all(np.diag(np.fliplr(subgrid)) == player): return True
        return False

    def render(self):
        """
        Prints a text representation of the board, highlighting winning pieces.
        """
        symbols = {0: " . ", 1: " X ", 2: " O "}
        display_board = self.board.copy()
        winning_coords = self._find_winning_line()

        if winning_coords:
            for r, c in winning_coords:
                display_board[r, c] *= -1 # Mark winning pieces
            symbols[-1] = "\033[94m X \033[0m"  # Blue "X"
            symbols[-2] = "\033[94m O \033[0m"  # Blue "O"

        print("  " + "   ".join(map(str, range(self.columns))))
        print("+" + "---+" * self.columns)
        for r in range(self.rows):
             print("|" + "|".join(symbols.get(cell, " ? ") for cell in display_board[r]) + "|")
             print("+" + "---+" * self.columns)
        print()
        return self.winner

    def _find_winning_line(self):
        """Internal helper to find coordinates of the winning line."""
        if self.winner is None: return None
        player = self.winner
        # Check horizontal
        for r in range(self.rows):
            for c in range(self.columns - self.win_length + 1):
                if np.all(self.board[r, c:c + self.win_length] == player): return [(r, c+i) for i in range(self.win_length)]
        # Check vertical
        for c in range(self.columns):
            for r in range(self.rows - self.win_length + 1):
                if np.all(self.board[r:r + self.win_length, c] == player): return [(r+i, c) for i in range(self.win_length)]
        # Check positive diagonal
        for c in range(self.columns - self.win_length + 1):
            for r in range(self.rows - self.win_length + 1):
                if np.all(np.diag(self.board[r:r + self.win_length, c:c + self.win_length]) == player): return [(r+i, c+i) for i in range(self.win_length)]
        # Check negative diagonal
        for c in range(self.columns - self.win_length + 1):
            for r in range(self.win_length - 1, self.rows):
                 subgrid = self.board[r - self.win_length + 1 : r + 1, c : c + self.win_length]
                 if np.all(np.diag(np.fliplr(subgrid)) == player): return [(r-i, c+i) for i in range(self.win_length)]
        return None
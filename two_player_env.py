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

        # print('board:', self.board)
        # print(f'row: {row}, column: {action}, self.board[row, action]: {self.board[row, action]}')
        # print(f'current_player: {self.current_player}')

        last_move = (row, action)


        reward = self.reward_for_blocking(self.board, self.current_player, last_move)
        if reward > 0:
            print(f"Player {self.current_player} gets blocking reward: {reward:.2f}")

        # Check game status
        if self.check_win(player=self.current_player):
            self.done = True
            self.winner = self.current_player
            reward += 2.5  # Win
        elif np.all(self.board != 0):
            self.done = True
            self.winner = None  # Draw
            reward = +0.3  # Draw is less rewarding than a win

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
    
    def reward_for_blocking(self, board, player, last_move):
        '''
        Reward the player for blocking the opponent's three-in-a-row.
        :param board: 2D list representing the Connect Four board.
        :param player: The current player (1 for X, 2 for O).
        :param last_move: Tuple (row, column) where the player just placed their piece.
        :return: float, reward value.
        '''
        opponent = 2 if player == 1 else 1
        rows = len(board)
        cols = len(board[0])
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Horizontal, Vertical, Diagonal Down-right, Diagonal Down-left
        reward = 0

        r, c = last_move

        for dr, dc in directions:
            for sign in [1, -1]:  # Check both directions from the piece
                sequence = 0
                # Check up to three pieces away from the last move in the current direction
                for i in range(1, 4):
                    nr, nc = r + dr * i * sign, c + dc * i * sign
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if board[nr][nc] == opponent:
                            sequence += 1
                        else:
                            break  # Stop if a non-opponent piece is encountered
                    else:
                        break  # Stop if out of bounds

                # Check if the sequence was exactly three and is blocked by the player's last move
                if sequence == 3:
                    # Check the space before the sequence (in the opposite direction)
                    prev_r, prev_c = r - dr * sign, c - dc * sign
                    if 0 <= prev_r < rows and 0 <= prev_c < cols:
                        if board[prev_r][prev_c] == 0:
                            reward += 1  # Reward for blocking
                            break  # No need to check further in this direction

        return reward
    

        '''
    def reward_for_blocking(self, board, player):
        """
        This function calculates the reward for blocking an opponent's three in a row.
        :param board: 2D list representing the Connect Four board.
        :param player: integer, 1 if the function is calculating for player 1, or 2 for player 2.
        :return: float, reward value.
        """
        opponent = 2 if player == 1 else 1
        rows = len(board)
        cols = len(board[0])
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Horizontal, Vertical, Diagonal Down-right, Diagonal Down-left
        reward = 0

        for r in range(rows):
            for c in range(cols):
                if board[r][c] == 0:  # Only consider empty cells for potential blocking
                    for dr, dc in directions:
                        count = 0
                        for i in range(1, 4):  # Check next three positions in the direction
                            nr, nc = r + dr * i, c + dc * i
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if board[nr][nc] == opponent:
                                    count += 1
                                elif board[nr][nc] == player:
                                    count = 0  # Reset if player's piece is found
                                    break
                            else:
                                break  # Out of bounds
                        if count == 3:
                            # Check for empty space on either side of the three in a row
                            before_r, before_c = r - dr, c - dc
                            after_r, after_c = r + dr * 4, c + dc * 4
                            if (0 <= before_r < rows and 0 <= before_c < cols and board[before_r][before_c] == 0) or \
                            (0 <= after_r < rows and 0 <= after_c < cols and board[after_r][after_c] == 0):
                                reward += 1  # Reward for blocking an opponent's three-in-a-row

        return reward
'''

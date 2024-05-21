

#two_player_env.py

import numpy as np
import random

from torch.utils.tensorboard import SummaryWriter



class TwoPlayerConnectFourEnv():
    def __init__(self, rows=6, columns=7, win_length=4, writer=None, sequence_length=7):
        self.rows = rows
        self.columns = columns
        self.win_length = win_length
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.current_player = 1  # for now, gets updated on reset
        self.done = False
        self.winner = None
        self.total_steps = 0 
        self.writer = writer  # Store the SummaryWriter instance
        self.reward_shaping = False
        self.debug_mode = False
        self.sequence_length = sequence_length

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
    
    def enable_reward_shaping(self, enable=True):
        self.reward_shaping = enable

    def enable_debug_mode(self, enable=False):
        self.debug_mode = enable

    def step(self, action):
        if self.done:
            raise ValueError("Game is over.")
        if self.board[0, action] != 0:
            board_state = self.board.copy()
            self.render(board_state)
            print(f'player {self.current_player}:\taction {action}')
            print(f'Valid actions: {self.get_valid_actions()}')
            print(f'Player symbol: {self.get_player_symbol(self.current_player)}')
            raise ValueError("Column is full.")

        row = max(np.where(self.board[:, action] == 0)[0])
        self.board[row, action] = self.current_player
        last_move = (row, action)

        # Initial reward calculation
        reward = 0
        r_win = 0
        r_block = 0
        r_opp = 0
        r_futil = 0
        r_force = 0
        r_adv = 0


        # Win check
        if self.check_win(player=self.current_player):
            r_win = 7 # Reward for winning is larger
            #print(f'\tplayer {self.current_player}:\twinning\t\t{reward:.2f}')
            self.done = True
            self.winner = self.current_player
            if self.writer is not None:
                self.writer.add_scalar(f'Env/Win_player_{self.current_player}', 1, self.total_steps)
        elif np.all(self.board != 0):
            self.done = True
            self.winner = None
            if self.writer is not None:
                self.writer.add_scalar('Env/Draw', 1, self.total_steps)
        
        # Reward shapping : fix this or delete it... 
        if self.reward_shaping:
            r_block = self.reward_for_blocking(self.board, self.current_player, last_move)
            r_opp   = self.reward_for_opportunities(self.board, self.current_player, last_move)
            r_futil = self.reward_for_futile_moves(self.board, action)            
            #r_force = self.reward_for_forcing(self.board, self.current_player, last_move)
            #r_adv   = self.reward_for_advanced_patterns(self.board, self.current_player, last_move)
        
        reward = r_win + r_block + r_opp + r_futil + r_force

        if self.debug_mode: #Print the details of the reward
            print(f'\n---- player {self.current_player}:\taction {action}---')
            board_state = self.board.copy()
            self.render(board_state)
            if self.reward_shaping:
                print(f'reward for winning: {r_win:.2f}')
                print(f'reward for blocking: {r_block:.2f}')
                print(f'reward for opportunities: {r_opp:.2f}')
                print(f'reward for futile moves: {r_futil:.2f}')
                #print(f'reward for forcing: {r_force:.2f}')
                #print(f'reward for advanced patterns: {r_adv:.2f}')
            print(f'Total reward: {reward:.2f}')
        
        self.current_player = 3 - self.current_player
        self.total_steps += 1

        next_state = self._get_next_state() 
        return self.board.flatten(), next_state, reward, self.done, self.current_player


    def _get_next_state(self):
        next_state = self.board.flatten()
        return next_state


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
        valid_col = [col for col in range(self.columns) if self.board[0, col] == 0]
        if len(valid_col) != 7 and self.debug_mode:  # Check if all columns are considered when debug mode is enabled
            print("Debugging get_valid_actions:")
            board_state = self.board.copy()
            self.render(board_state)
            print("Valid columns identified: ", valid_col)
        return valid_col


    def render(self, board_state):
        symbols = {0: " . ", 1: " X ", 2: " O "}
        # Create a copy of the board for display purposes
        if board_state is None:
            board_state = self.board
            
        display_board = board_state
        
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
    
    def reward_for_opportunities(self, board, player, move):
        reward = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # vertical, horizontal, diagonal, anti-diagonal
        for d in directions:
            count = 1  # start with the current piece
            for step in [1, -1]:
                for i in range(1, 4):
                    r = move[0] + step * i * d[0]
                    c = move[1] + step * i * d[1]
                    if 0 <= r < board.shape[0] and 0 <= c < board.shape[1] and board[r, c] == player:
                        count += 1
                    else:
                        break
            if count == 2:
                reward += 0.1  # small reward for creating a line of two
            elif count == 3:
                reward += 0.5  # larger reward for creating a line of three

        return reward

    def reward_for_forcing(self, board, player, move):
        opponent = 3 - player
        forced_moves = 0
        # Check around the last move
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # vertical, horizontal, diagonal, anti-diagonal
        for d in directions:
            potential_threats = 0
            for step in [1, -1]:
                for i in range(1, 4):
                    r = move[0] + step * i * d[0]
                    c = move[1] + step * i * d[1]
                    if 0 <= r < board.shape[0] and 0 <= c < board.shape[1]:
                        if board[r, c] == opponent:
                            break
                        if board[r, c] == 0:
                            potential_threats += 1
                            break
            if potential_threats == 1:
                forced_moves += 1

        return 0.2 * forced_moves if forced_moves > 0 else 0

    def reward_for_futile_moves(self, board, action):
        # Check if the column is nearly full (i.e., only the top cell is empty).
        if board[1, action] != 0:
            # Assume a default penalty for a nearly full column
            penalty = -0.2

            # Check if this move creates a direct opportunity to win in the next turn
            # which would make it not a futile move.
            # We temporarily simulate placing one more piece in the same column.
            if board[0, action] == 0:  # Just to be safe, we check if the top is still open
                board[0, action] = self.current_player  # Temporarily place piece
                if self.check_win(player=self.current_player):  # Check for a win condition
                    penalty = 0  # No penalty if it can lead to a win next turn
                board[0, action] = 0  # Reset the top cell

            return penalty
        return 0
    



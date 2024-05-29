
#c4_env_tr.py

import numpy as np
import random

class ConnectFourEnv:
    def __init__(self, rows=6, columns=7):
        self.rows = rows
        self.columns = columns
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.current_player = random.choice([1, 2])
        self.done = False
        self.winner = None
        self.win_length = 4
        self.reward_shaping = False
        self.debug_mode = False
        self.total_steps = 0
        self.writer = None

    def reset(self):
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.current_player = random.choice([1, 2])
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

        next_state = self.board.flatten()
        #return self.board.flatten(), 
        return next_state, reward, self.done, self.current_player


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

        if board_state.shape != (self.rows, self.columns):
            #reshape the board_state
            board_state = np.array(board_state).reshape(self.rows, self.columns)
            
        display_board = board_state

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
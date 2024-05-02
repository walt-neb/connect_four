import numpy as np

class ConnectFour:
    def __init__(self, rows=6, columns=7, win_length=4):
        self.rows = rows
        self.columns = columns
        self.win_length = win_length
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.done = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.done = False
        self.winner = None
        return self.board.flatten()

    def step(self, action):
        """ Place a piece in the chosen column. """
        if self.done:
            raise ValueError("Game is over.")
        if self.board[0, action] != 0:
            raise ValueError("Column is full.")
        
        # Find the lowest empty spot in the column
        row = max(np.where(self.board[:, action] == 0)[0])
        self.board[row, action] = 1  # Assuming player 1 is always the agent for simplicity

        # Check game status
        if self.check_win(player=1):
            self.done = True
            self.winner = 1
            reward = 1  # Win
        elif np.all(self.board != 0):
            self.done = True
            reward = 0  # Draw
        else:
            reward = 0  # Game continues

        return self.board.flatten(), reward, self.done

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

    def render(self):
        print("Current board:")
        for r in range(self.rows):
            print(' '.join([str(x) for x in self.board[r]]))
        print()

# Example usage
env = ConnectFour()
env.reset()
env.render()

# filename: test_reward_functions.py

import unittest
import numpy as np
from two_player_env import TwoPlayerConnectFourEnv  # Assuming your env class is in this module

class TestRewardFunctions(unittest.TestCase):
    def setUp(self):
        # This method will be called before each test function
        self.env = TwoPlayerConnectFourEnv()
        self.env.current_player = 1  # Set the current player

    def test_reward_for_opportunities_direct_win(self):
        # Set up a specific board state
        self.env.board = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 1, 2, 0],
            [0, 0, 0, 2, 1, 1, 0],
            [0, 0, 1, 1, 2, 1, 2]
        ])
        # Assume the player is making a move that creates a direct winning opportunity
        reward = self.env.reward_for_opportunities(self.env.board, 1, (2, 3))
        print(f"Test for opportunities with direct win: Reward = {reward}")
        # Check if the reward is correctly assigned
        self.assertEqual(reward, 0.5)  # Example expected value

    def test_reward_for_opportunities_no_opportunity(self):
        # Setup similar to above
        reward = self.env.reward_for_opportunities(self.env.board, 1, (5, 0))
        print(f"Test for opportunities with no opportunity: Reward = {reward}")
        # No strategic advantage should give no additional reward
        self.assertEqual(reward, 0)

class TestAdvancedPatternRecognition(unittest.TestCase):
    def setUp(self):
        self.env = TwoPlayerConnectFourEnv()  # assuming this initializes an empty board
        self.env.current_player = 1

    def test_advanced_pattern_recognition(self):
        # Setup the board manually to a state that should trigger the advanced pattern reward
        self.env.board = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [2, 1, 2, 0, 0, 0, 0],
            [2, 1, 1, 2, 0, 0, 0]
        ])  # This setup assumes '1' is the player, and this setup should have a clear advanced pattern
        reward = self.env.reward_for_advanced_patterns(self.env.board, 1, (2, 0))
        print(f"Test for advanced pattern recognition: Reward = {reward}")
        # Check the output
        self.assertNotEqual(reward, 0, "Advanced pattern reward should not be zero")


if __name__ == '__main__':
    unittest.main()

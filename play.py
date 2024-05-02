

# filename: play.py

from sys import argv
import torch
import numpy as np
from dqn_agent import DQNAgent  
from two_player_env import ConnectFour 

def main():
    if len(argv) != 2:
        print("Usage: python play.py <trained_agent_path>")
        return
    agent_path = argv[1]

    # Initialize the game environment
    env = ConnectFour()
    
    # Parameters
    input_dim = 6 * 7  # Board size
    output_dim = 7     # One output for each column

    # Initialize and load the trained agent
    agent = DQNAgent(input_dim, output_dim)
    #agent.load_state_dict(torch.load('agent1_weights.pth'))

    agent.load_state_dict(torch.load(agent_path))
    print("Loaded weights for agent.")

    agent.eval()  # Set the network to evaluation mode

    # Start the game
    state, current_player = env.reset()
    done = False
    
    while not done:
        env.render()  # Display the current board

        if current_player == 1:  # Assuming human is player 1
            action = get_human_action(env)
        else:  # AI's turn
            action = agent.select_action(np.array(state), env.get_valid_actions(), 0)

        # Update environment with selected action
        state, reward, done, current_player = env.step(action)

        if done:
            env.render()  # Show the final board
            if reward == 1:  # Assuming reward of 1 for a win
                print(f"Player {env.winner} wins!")
            else:
                print("It's a draw!")

def get_human_action(env):
    valid_actions = env.get_valid_actions()
    action = -1
    while action not in valid_actions:
        try:
            action = int(input("Enter your column choice (0-6): "))
            if action not in valid_actions:
                print("Invalid column or column full. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 6.")
    return action

if __name__ == '__main__':
    main()

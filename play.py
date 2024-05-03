

# filename: play.py

from sys import argv
import torch
import numpy as np
from dqn_agent import DQNAgent  
from two_player_env import ConnectFour 

def get_hidden_layer_dimensions(agent_path):
    
    #Extracts hidden layer dimensions from a PyTorch checkpoint weights file.
    
    # Load the checkpoint weights
    state_dict = torch.load(agent_path)

    # Extract hidden layer dimensions (from bias parameters)
    hidden_layer_dims = []
    for key, param in state_dict.items():
        if 'bias' in key:  # Focus on bias parameters
            hidden_layer_dims.append(param.shape[0])  

    hidden_layer_dims.pop()  # Remove the output layer dimension

    # Format the output string
    formatted_dims = f"[{','.join(str(dim) for dim in hidden_layer_dims)}]"
    return formatted_dims



def main():
    if len(argv) != 2:
        print("Usage: python play.py <trained_agent_weights>")
        return
    agent_path = argv[1]

    # Initialize the game environment
    env = ConnectFour()
    
    # Parameters
    input_dim = 6 * 7  # Board size
    output_dim = 7     # One output for each column


    layer_dims =  get_hidden_layer_dimensions(agent_path)
    print(layer_dims)
    layer_dims = layer_dims.replace(',', ' ')
    layer_dims_string = layer_dims.strip('[]')
    layer_dims_string = [int(dim) for dim in layer_dims_string.split()]
    print(layer_dims_string)

    # Initialize and load the trained agent
    agent = DQNAgent(input_dim, output_dim, layer_dims_string)

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
            if reward == 1.5:  # Assuming reward of 1 for a win
                print(f"Player {env.winner} wins!")
            elif reward == -1:
                print(f"Human wins!")
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

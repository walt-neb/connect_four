

# filename: play_human.py

from sys import argv
import torch
import numpy as np
from ddqn_agent import DQNAgent
from two_player_env import TwoPlayerConnectFourEnv

def get_hidden_layer_dimensions(agent_path):
    # Load the checkpoint weights
    state_dict = torch.load(agent_path, map_location=torch.device('cpu'))

    # Extract hidden layer dimensions (from bias parameters)
    hidden_layer_dims = []
    for key, param in state_dict.items():
        if 'bias' in key:  # Focus on bias parameters
            hidden_layer_dims.append(param.shape[0])

    hidden_layer_dims.pop()  # Remove the output layer dimension
    return hidden_layer_dims

def load_agent(agent_path, input_dim, output_dim):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    layer_dims = get_hidden_layer_dimensions(agent_path)
    agent = DQNAgent(input_dim, output_dim, layer_dims)
    agent.load_state_dict(torch.load(agent_path, map_location=device))
    agent.eval()  # Set the network to evaluation mode
    agent.to(device)  # Ensure the model is on the correct device
    return agent

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

def main():
    if len(argv) not in [2, 3]:
        print("Usage: python play_human.py <agent1_weights>")
        return

    agent1_path = argv[1]
    num_games = int(argv[2]) if len(argv) == 3 else 1

    # Initialize the game environment
    env = TwoPlayerConnectFourEnv(writer=None)

    # Parameters
    input_dim = 6 * 7  # Board size
    output_dim = 7     # One output for each column

    # Load agents
    agent1 = load_agent(agent1_path, input_dim, output_dim)

    # Game stats
    wins = {1: 0, 2: 0, 'draws': 0}
    steps_per_game = []

    for game in range(num_games):
        state, current_player = env.reset()
        done = False
        step_count = 0
        print(f'--- Game {game + 1} --- ')
        while not done:
            _ = env.render()
            step_count += 1
            if current_player == 1:
                action = get_human_action(env)
            else:  # AI's turn
                action = agent1.select_action(np.array(state), env.get_valid_actions(), 0) 

            state, _, done, current_player = env.step(action)
        
        _ = env.render()
        # Print if human or AI wins
        if env.winner:
            if env.winner == 1:
                print('Human wins!')
            else:
                print('AI wins!')
        else:
            print('Draw!')
        
        steps_per_game.append(step_count)
        if env.winner:
            wins[env.winner] += 1
        else:
            wins['draws'] += 1

    # Print results
    print(f"Results after {num_games} games:")
    print(f"Human wins: {wins[1]}")
    print(f"Agent wins: {wins[2]}")
    print(f"Draws: {wins['draws']}")
    print(f"Average number of steps per game: {np.mean(steps_per_game):.2f}")

if __name__ == '__main__':
    main()
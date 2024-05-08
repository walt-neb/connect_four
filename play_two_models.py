

# filename: play_two_models.py

from sys import argv
import torch
import numpy as np
from dqn_agent import DQNAgent
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
    layer_dims = get_hidden_layer_dimensions(agent_path)
    agent = DQNAgent(input_dim, output_dim, layer_dims)
    agent.load_state_dict(torch.load(agent_path))
    agent.eval()  # Set the network to evaluation mode
    return agent

def main():
    if len(argv) not in [3, 4]:
        print("Usage: python play.py <agent1_weights> <agent2_weights> [num_games]")
        return

    agent1_path = argv[1]
    agent2_path = argv[2]
    num_games = int(argv[3]) if len(argv) == 4 else 1

    # Initialize the game environment
    env = TwoPlayerConnectFourEnv(writer=None)

    # Parameters
    input_dim = 6 * 7  # Board size
    output_dim = 7     # One output for each column

    # Load agents
    agent1 = load_agent(agent1_path, input_dim, output_dim)
    agent2 = load_agent(agent2_path, input_dim, output_dim)

    # Game stats
    wins = {1: 0, 2: 0, 'draws': 0}
    steps_per_game = []

    for game in range(num_games):
        state, current_player = env.reset()
        done = False
        step_count = 0

        while not done:
            if num_games == 1:              
                _ = env.render()
            step_count += 1
            action = agent1.select_action(np.array(state), env.get_valid_actions(), 0) if current_player == 1 else agent2.select_action(np.array(state), env.get_valid_actions(), 0)
            state, _, done, current_player = env.step(action)

        steps_per_game.append(step_count)
        if env.winner:
            wins[env.winner] += 1
        else:
            wins['draws'] += 1

    # Print results
    print(f"Results after {num_games} games:")
    print(f"{agent1_path} wins: {wins[1]}")
    print(f"{agent2_path} wins: {wins[2]}")
    print(f"Draws: {wins['draws']}")
    print(f"Average number of steps per game: {np.mean(steps_per_game):.2f}")

if __name__ == '__main__':
    main()


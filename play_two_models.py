# filename: play_two_models.py
"""
Script to play two trained CNN DDQN Agents against each other.

Loads two separate agent models, automatically determining their architecture
by looking for associated .hyp files in the './hyp/' directory based on
the model filename. Reports win/loss/draw statistics.
"""

import sys
import torch
import numpy as np
import ast
import os
import re # Import regular expressions

# --- Local Imports ---
from ddqn_agent_cnn import CNNDDQNAgent
from two_player_env import TwoPlayerConnectFourEnv

# --- Helper Functions ---

def normalize_state(board_2d, current_player):
    """Normalizes board state (player=1, opp=-1, empty=0)."""
    normalized_board = np.zeros_like(board_2d, dtype=np.float32)
    opponent_player = 3 - current_player
    normalized_board[board_2d == current_player] = 1.0
    normalized_board[board_2d == opponent_player] = -1.0
    return normalized_board.flatten()

def load_hyperparams(hyp_file):
    """Loads hyperparameters from a text file (needed for architecture)."""
    params = {}
    try:
        with open(hyp_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line: continue
                if "=" in line:
                    var_name, var_value = line.split("=", 1)
                    var_name = var_name.strip(); var_value = var_value.strip()
                    try: params[var_name] = ast.literal_eval(var_value)
                    except: params[var_name] = var_value
    except FileNotFoundError:
        print(f"Config file not found: {hyp_file}")
        raise # Re-raise error
    return params

def load_cnn_agent(agent_path, device):
    """
    Loads a CNNDDQNAgent, loading architecture from associated .hyp file
    located in the './hyp/' directory.
    """
    print(f"Attempting to load agent: {agent_path}")
    weights_filename = os.path.basename(agent_path)

    # --- Determine the base hyperparameter filename ---
    match = re.match(r"^(?:m1_|m2_)?(.*?)(?:_ep\d+|_final)?\.pth$", weights_filename)
    if match:
        hyp_base_name = match.group(1)
    else: # Fallback parsing
        hyp_base_name = weights_filename.replace('.pth', '')
        if hyp_base_name.startswith('m1_'): hyp_base_name = hyp_base_name[3:]
        if hyp_base_name.startswith('m2_'): hyp_base_name = hyp_base_name[3:]
        if '_final' in hyp_base_name: hyp_base_name = hyp_base_name.replace('_final', '')
        last_underscore_idx = hyp_base_name.rfind('_ep')
        if last_underscore_idx != -1 and hyp_base_name[last_underscore_idx+3:].isdigit():
            hyp_base_name = hyp_base_name[:last_underscore_idx]

    # --- Construct the CORRECT path to the .hyp file ---
    config_path = os.path.join('hyps', hyp_base_name + '.hyp') # Look in ./hyp/

    print(f"Derived hyperparameter base name: '{hyp_base_name}'")
    print(f"Looking for config file at: '{config_path}'")

    # --- Load architecture from the identified .hyp file ---
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        print("Ensure a .hyp file matching the base name of your model exists in the 'hyp/' directory.")
        sys.exit(1)

    try:
        params = load_hyperparams(config_path)
        # Determine if loading Agent 1 or Agent 2 architecture
        if 'm1_' in weights_filename:
            cnn_params = params['cnn_a1']
            fc_dims = params['fc_a1']
            print("Inferred architecture as Agent 1's from config.")
        elif 'm2_' in weights_filename:
            cnn_params = params['cnn_a2']
            fc_dims = params['fc_a2']
            print("Inferred architecture as Agent 2's from config.")
        else:
            # Default to agent 1 if no m1_/m2_ prefix found
            print("Warning: Could not determine if agent is A1 or A2 from filename, assuming A1 architecture.")
            cnn_params = params['cnn_a1']
            fc_dims = params['fc_a1']

    except KeyError as e:
        print(f"Error: Missing expected architecture key ({e}) in config file '{config_path}'.")
        print("Ensure the .hyp file contains 'cnn_a1', 'fc_a1', 'cnn_a2', 'fc_a2'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or parsing architecture from config file '{config_path}': {e}")
        sys.exit(1)

    print(f"Using Agent Architecture - CNN: {cnn_params}, FC Hidden Dims: {fc_dims}")

    # --- Initialize and load weights ---
    input_dim = (1, 6, 7); output_dim = 7
    input_channels, input_height, input_width = input_dim
    agent = CNNDDQNAgent(input_channels, input_height, input_width, output_dim, cnn_params, fc_dims)
    try:
        agent.load_state_dict(torch.load(agent_path, map_location=device, weight_only=True))
        agent.eval()
        agent.to(device)
        print(f"Agent loaded successfully from: {agent_path}")
        return agent
    except FileNotFoundError:
        print(f"Error: Agent weights file not found at {agent_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading agent weights from {agent_path}: {e}")
        print("Ensure the architecture defined in the config file matches the saved model.")
        sys.exit(1)

# --- Main Function ---
def main():
    if len(sys.argv) not in [3, 4]:
        print("\nUsage: python play_two_models.py <agent1_weights> <agent2_weights> [num_games]")
        print("Example: python play_two_models.py ./wts/m1_final.pth ./wts/m2_final.pth 100")
        print("\nNote: Assumes corresponding .hyp files exist in './hyp/' directory.")
        sys.exit(1)

    agent1_path = sys.argv[1]
    agent2_path = sys.argv[2]
    num_games = int(sys.argv[3]) if len(sys.argv) == 4 else 1

    # --- Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TwoPlayerConnectFourEnv()

    # Load agents
    agent1 = load_cnn_agent(agent1_path, device)
    agent2 = load_cnn_agent(agent2_path, device)

    # Game stats
    wins = {1: 0, 2: 0}; draws = 0; total_steps = 0

    print(f"\n--- Starting {num_games} Game(s) Between: ---")
    print(f"  Agent 1 (X): {os.path.basename(agent1_path)}")
    print(f"  Agent 2 (O): {os.path.basename(agent2_path)}")
    print("-" * 40)

    start_play_time = time.time() # Optional: time the games

    for game in range(num_games):
        state_2d, current_player = env.reset()
        done = False
        step_count = 0

        # Progress indicator
        if num_games > 20 and game > 0 and game % (num_games // 20) == 0:
             elapsed = time.time() - start_play_time
             print(f"  Progress: {game}/{num_games} ({game/num_games:.0%}) | Time Elapsed: {elapsed:.1f}s")

        while not done:
            if num_games == 1: # Render only for a single game run
                env.render()
                print(f"Player {current_player}'s turn...")

            normalized_state = normalize_state(state_2d, current_player)
            valid_actions = env.get_valid_actions()

            if current_player == 1:
                action = agent1.select_action(normalized_state, valid_actions, 0) # Epsilon = 0
            else:
                action = agent2.select_action(normalized_state, valid_actions, 0) # Epsilon = 0

            next_state_2d, reward, done, next_player = env.step(action)
            state_2d = next_state_2d
            current_player = next_player
            step_count += 1

        # End of Game
        total_steps += step_count
        if env.winner == 1: wins[1] += 1
        elif env.winner == 2: wins[2] += 1
        else: draws += 1

        if num_games == 1:
             print("\n--- Game Over ---")
             env.render()
             if env.winner: print(f"Agent {env.winner} Wins!")
             else: print("It's a Draw!")
             print(f"Game ended in {step_count} steps.")

    end_play_time = time.time() # Optional

    # Final Results
    print("\n--- Final Results ---")
    print(f"Games Played: {num_games}")
    a1_win_rate = wins[1] / num_games if num_games > 0 else 0
    a2_win_rate = wins[2] / num_games if num_games > 0 else 0
    draw_rate = draws / num_games if num_games > 0 else 0
    print(f"Agent 1 ({os.path.basename(agent1_path)}) Wins: {wins[1]} ({a1_win_rate:.1%})")
    print(f"Agent 2 ({os.path.basename(agent2_path)}) Wins: {wins[2]} ({a2_win_rate:.1%})")
    print(f"Draws: {draws} ({draw_rate:.1%})")
    if num_games > 0:
        print(f"Average steps per game: {total_steps / num_games:.2f}")
        print(f"Total playing time: {end_play_time - start_play_time:.2f}s")

if __name__ == '__main__':
    import time # Add import for timing
    main()

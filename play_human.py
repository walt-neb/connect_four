# filename: play_human.py
"""
Script to play Connect Four against a trained CNN DDQN Agent.

Allows a human player (always Player 1, 'X') to play against a loaded
AI agent (always Player 2, 'O'). Now correctly loads architecture
parameters from the associated .hyp file in the './hyp/' directory.
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
    located in the './hyp/' directory. Assumes agent uses 'agent 1' architecture
    from the hyp file unless 'm2_' prefix is found in filename.
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
    config_path = os.path.join('hyp', hyp_base_name + '.hyp') # Look in ./hyp/

    print(f"Derived hyperparameter base name: '{hyp_base_name}'")
    print(f"Looking for config file at: '{config_path}'")

    # --- Load architecture from the identified .hyp file ---
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        print("Ensure a .hyp file matching the base name of your model exists in the 'hyp/' directory.")
        sys.exit(1)

    try:
        params = load_hyperparams(config_path)
        # For human play, AI is usually Agent 2 (O), but training saves m1/m2.
        # Let's default to loading m1's architecture unless the file clearly indicates m2.
        if 'm2_' in weights_filename:
             print("Loading architecture as Agent 2's from config.")
             cnn_params = params['cnn_a2']
             fc_dims = params['fc_a2']
        else:
             print("Assuming architecture is Agent 1's from config (or no m1_/m2_ prefix found).")
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


def get_human_action(env):
    """ Prompts the human player for a valid action. """
    valid_actions = env.get_valid_actions()
    while True:
        try:
            action_str = input(f"Enter your column choice {valid_actions}: ")
            action = int(action_str)
            if action in valid_actions: return action
            else: print("Invalid column or column full.")
        except ValueError: print("Invalid input. Please enter a number.")
        except EOFError: print("\nInput interrupted."); sys.exit(0)

# --- Main Function ---
def main():
    if len(sys.argv) not in [2, 3]:
        print("\nUsage: python play_human.py <agent_weights_path> [num_games]")
        print("Example: python play_human.py ./wts/m1_h4_cnn_final.pth 5")
        print("\nNote: Assumes associated .hyp file exists in './hyp/' directory.")
        sys.exit(1)

    agent_weights_path = sys.argv[1]
    num_games = int(sys.argv[2]) if len(sys.argv) == 3 else 1

    # --- Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TwoPlayerConnectFourEnv()

    # Load the AI agent (will be Player 2)
    ai_agent = load_cnn_agent(agent_weights_path, device)

    # Game stats
    human_wins = 0; ai_wins = 0; draws = 0; total_steps = 0

    print("\n--- Starting Connect Four Game(s) ---")
    print("You are Player 1 ('X'), AI is Player 2 ('O')")

    for game in range(num_games):
        state_2d, current_player = env.reset()
        done = False
        step_count = 0
        print(f"\n--- Game {game + 1}/{num_games} --- ")
        print(f"{'Human (X)' if current_player == 1 else 'AI (O)'} starts.")

        while not done:
            env.render()
            step_count += 1

            if current_player == 1: # Human's turn
                print("Your turn (Player 1 - X).")
                action = get_human_action(env)
            else: # AI's turn (Player 2 - O)
                print("AI's turn (Player 2 - O)...")
                normalized_state = normalize_state(state_2d, current_player) # Normalize for AI (P2)
                valid_actions = env.get_valid_actions()
                action = ai_agent.select_action(normalized_state, valid_actions, 0) # Epsilon = 0
                print(f"AI chose column: {action}")

            next_state_2d, reward, done, next_player = env.step(action)
            state_2d = next_state_2d
            current_player = next_player

        # End of Game
        total_steps += step_count
        print("\n--- Game Over ---")
        env.render()
        if env.winner == 1: print("Congratulations, Human wins!"); human_wins += 1
        elif env.winner == 2: print("AI wins!"); ai_wins += 1
        else: print("It's a Draw!"); draws += 1
        print(f"Game ended in {step_count} steps.")

    # Final Results
    print("\n--- Final Results ---"); print(f"Games Played: {num_games}")
    print(f"Human (P1) Wins: {human_wins} ({human_wins/num_games:.1%})")
    print(f"AI (P2) Wins: {ai_wins} ({ai_wins/num_games:.1%})")
    print(f"Draws: {draws} ({draws/num_games:.1%})")
    if num_games > 0: print(f"Average steps per game: {total_steps / num_games:.2f}")

if __name__ == '__main__':
    main()

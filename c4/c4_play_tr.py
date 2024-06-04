
#filename = 'c4_play_tr.py'
# This script allows you to play Connect Four against a trained Transformer model.

import torch.optim as optim
import numpy as np
import datetime
import os
import sys

from c4_env_tr import ConnectFourEnv
from c4_replay_buf import ReplayBuffer
from c4_transformer import TransformerAgent
import torch
import torch.nn as nn
import time

from torch.utils.tensorboard import SummaryWriter

def load_hyperparams(hyp_file):
    params = {}
    with open(hyp_file, 'r') as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                # Parse line
                var_name, var_value = line.split("=")
                var_name = var_name.strip()  # remove leading/trailing white spaces
                var_value = var_value.strip()

                # Attempt to convert variable value to int, float, or leave as string
                try:
                    var_value = int(var_value)
                except ValueError:
                    try:
                        var_value = float(var_value)
                    except ValueError:
                        # If it's neither int nor float, keep as string
                        pass

                # Add the variable to the params dictionary
                params[var_name] = var_value
    return params


def main():
    start_time = datetime.datetime.now()
    if len(sys.argv) < 2:
        print("Usage: python c4_play_tr.py [model.pth]]")
        return
    
    hyp_file = sys.argv[1]
    hyp_file_root = hyp_file.rstrip('.pth')
    hyp_file_root = os.path.basename(hyp_file_root)
    print(f'hyp_file_root: {hyp_file_root}')

    par = load_hyperparams('./hyps/'+hyp_file_root+'.hyp')

    input_dim = par['input_dim']
    embed_dim = par['embed_dim']
    n_heads = par['n_heads']
    ff_dim = par['ff_dim']
    n_layers = par['n_layers']
    output_dim = par['output_dim']
    dropout = par['dropout']


    # Initialize components
    env = ConnectFourEnv()
    agent = TransformerAgent(input_dim, embed_dim, n_heads, ff_dim, n_layers, output_dim, dropout)

    # Load the model's state dictionary
    model_file = f'./wts/{hyp_file_root}.pth'
    agent.load_state_dict(torch.load(model_file))
    agent.eval()  # Set the model to evaluation mode


    # play_against_human 
    human_player = int(input("Choose your player (1 or 2): "))
    while human_player not in [1, 2]:
        human_player = int(input("Invalid choice. Choose your player (1 or 2): "))
    
    agent_player = 3 - human_player  # The agent will be the other player

    state, player = env.reset()
    done = False

    while not done:
        env.render(state) 
        if player == agent_player:
            # Agent's turn
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = agent(state_tensor)
            valid_actions = env.get_valid_actions()
            mask = torch.ones(output_dim) * float('-inf')
            mask[valid_actions] = 0
            masked_q_values = q_values + mask
            action = torch.argmax(masked_q_values).item()
            print(f"Agent's move: {action}")
        else:
            # Human's turn
            valid_actions = env.get_valid_actions()
            action = int(input(f"Your move (valid actions: {valid_actions}): "))
            while action not in valid_actions:
                action = int(input(f"Invalid move. Your move (valid actions: {valid_actions}): "))

        # Perform the action and get the next state and reward
        state, reward, done, next_player = env.step(action)

        # Render the state
        env.render(state)

        if done:
            if reward == 1:
                if player == agent_player:
                    print("Agent wins!")
                else:
                    print("Human wins!")
            else:
                print("It's a draw!")
        else:
            player = next_player  # Switch player


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")


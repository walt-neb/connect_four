


#filename:  train_c4.py

import curses
import datetime
import time
import ast
import os
import sys
import pickle
import numpy as np
from ddqn_cnn3d import CNN3D 
from two_player_env import TwoPlayerConnectFourEnv
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
from collections import namedtuple

from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import torch
import pickle
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

dbmode=1
def str_to_bool(s):
    result = s.lower() == 'true'
    return result

def ensure_list_of_tuples(data):
    if isinstance(data, tuple):
        return [data]
    return data


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

def print_parameters(params):
    if not params:
        print("The parameters dictionary is empty.")
        return
    param_str = ""
    print("*** Training Parameters: ")
    for key, value in params.items():
        param_str += (f"\t\t{key} :\t{value}\n")
    return param_str

def load_all_models(a1, a2, o1, o2, rp1, rp2, hyp_root, ckpt_num):
    # Load models
    wts_dir = "./wts/"
    base_name = f"{wts_dir}chkpt_{hyp_root}_{ckpt_num}"
    a1.load_state_dict(torch.load(f"{base_name}_a1.pth"))
    a1.load_state_dict(torch.load(f"{base_name}_a2.pth"))
    o1.load_state_dict(torch.load(f"{base_name}_a1_optimizer.pth"))
    o2.load_state_dict(torch.load(f"{base_name}_a2_optimizer.pth"))
    
    # Load replay buffer
    with open(f"{base_name}_a1_replay_buffer.pkl", "rb") as f:
        rp1 = pickle.load(f)
    with open(f"{base_name}_a2_replay_buffer.pkl", "rb") as f:
        rp2 = pickle.load(f)

    print(f"All models and replay buffer loaded from files with base filename: {base_name}")
    return a1, a2, o1, o2, rp1, rp2


def save_all_models(agent, optimizer, name, replay_buffer, base_filename):
    #Saves both agents' models and replay buffer to files.
    # Save models (a single model for each agent)
    filename = f"{base_filename}_{name}.pth"
    print(f"Saving model to: {filename}")  # Print the full file path
    torch.save(agent.state_dict(), filename)
    torch.save(optimizer.state_dict(), f"{base_filename}_{name}_optimizer.pth") 
    # Save replay buffer
    with open(f"{base_filename}_{name}_replay_buffer.pkl", "wb") as f:
        pickle.dump(replay_buffer, f)
    print(f"Agent {name} model and replay buffer saved to files with base filename: {base_filename}")

def load_agents_and_buffer(cnn_a1, cnn_a2, fc_a1, fc_a2, seq_len=7, rp_size=10000, agent1_wts=None, agent2_wts=None, rp_buffer_file=None):
    input_channels = 1   # Assuming a single-channel input for the Connect Four board
    input_height = 6     # Connect Four board height
    input_width = 7      # Connect Four board width
    output_dim = 7       # One output for each column

    print(f'A1 Convolutional layers:   {cnn_a1}')
    print(f'A1 Fully connected layers: {fc_a1}')    
    print(f'A2 Convolutional layers:   {cnn_a2}')   
    print(f'A2 Fully connected layers: {fc_a2}')
    print(f'Sequence length:           {seq_len}')

    # Initialize agents with the specified convolutional and fully connected layers
    agent1 = CNN3D(input_channels, seq_len, input_height, input_width, output_dim, cnn_a1, fc_a1)
    agent2 = CNN3D(input_channels, seq_len, input_height, input_width, output_dim, cnn_a2, fc_a2)

    replay_buffer1 = deque(maxlen=rp_size)  # Adjust size as needed
    replay_buffer2 = deque(maxlen=rp_size)  # Adjust size as needed

    # Load weights if paths are provided
    if agent1_wts and os.path.exists(agent1_wts):    
        agent1.load_state_dict(torch.load(agent1_wts))
        agent1.eval()
        print(f"Loaded weights for Agent 1 from {agent1_wts}.")
    else:
        print(f"Starting Agent 1 from scratch or file not found: {agent1_wts}")

    if agent2_wts and os.path.exists(agent2_wts):
        agent2.load_state_dict(torch.load(agent2_wts))
        agent2.eval()
        print(f"Loaded weights for Agent 2 from {agent2_wts}.")
    else:
        print(f"Starting Agent 2 from scratch or file not found: {agent2_wts}")

    # Load replay buffer if a file is provided
    if rp_buffer_file and os.path.exists(rp_buffer_file):
        with open(rp_buffer_file+'_a1', 'rb') as f:
            replay_buffer1 = pickle.load(f)
            assert isinstance(replay_buffer1, deque), "Replay buffer is not a deque!"
        with open(rp_buffer_file+'_a2', 'rb') as f:
            replay_buffer2 = pickle.load(f)
            assert isinstance(replay_buffer2, deque), "Replay buffer is not a deque!"

        print(f"Loaded replay buffers from {rp_buffer_file}.")
    else:
        print(f"Starting new replay buffers or file not found: {rp_buffer_file}")

    return agent1, agent2, replay_buffer1, replay_buffer2


def soft_update(target_model, source_model, tau=0.005):
    """
    Softly update the target model's weights using the weights from the source model.
    tau is a small coefficient.
    """
    for target_param, param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)        

# Define the Transition namedtuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

def train(agent, agent_tgt, optimizer, replay_buffer, batch_size, gamma, tau):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Ensure enough samples for a batch
    if len(replay_buffer) < batch_size:
        return None  # Not enough samples to train

    states, actions, rewards, next_states, dones, masks = replay_buffer.sample(batch_size)

    states = torch.tensor(states, device=device, dtype=torch.float32)
    actions = torch.tensor(actions, device=device, dtype=torch.long)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
    next_states = torch.tensor(next_states, device=device, dtype=torch.float32)
    dones = torch.tensor(dones, device=device, dtype=torch.bool)
    masks = torch.tensor(masks, device=device, dtype=torch.float32)

    # Current Q values
    current_q_values = agent(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    
    # Next Q values
    next_q_values = agent_tgt(next_states).max(1)[0]
    next_q_values[dones] = 0.0  # Zero out the next Q values for terminal states

    # Compute the target Q values using the Bellman equation
    target_q_values = rewards + gamma * next_q_values

    # Calculate the MSE loss
    loss = F.mse_loss(current_q_values, target_q_values)

    # Gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Soft update of the target network
    soft_update(agent_tgt, agent, tau)

    return loss.item()


def main():

    start_time = datetime.datetime.now()
    print(f'torch.cuda.is_available()={torch.cuda.is_available()}')

    if len(sys.argv) < 2:
        print("Usage: python train_c4.py <hyperparameters_file> [agent1_weights] [agent2_weights] [replay_buffer]")
        return
    
    hyp_file = sys.argv[1]
    hyp_file_root = hyp_file.rstrip('.hyp')
    hyp_file_root = os.path.basename(hyp_file_root)
    print(f'hyp_file_root: {hyp_file_root}')

    params = load_hyperparams(hyp_file)
    print(print_parameters(params))
    writer = SummaryWriter(f'runs/{hyp_file_root}_connect_four_experiment')

    try:
        cnn_a1 = ast.literal_eval(params["cnn_a1"])
        cnn_a2 = ast.literal_eval(params["cnn_a2"])
        cnn_a1 = ensure_list_of_tuples(cnn_a1)
        cnn_a2 = ensure_list_of_tuples(cnn_a2)
        fc_a1 = ast.literal_eval(params["fc_a1"])
        fc_a2 = ast.literal_eval(params["fc_a2"])
    except Exception as e:
        print(e)
        sys.exit(1)
    
    render_games = ast.literal_eval(params["render_game_at"])
    end_episode = params["end_episode"]
    batch_size = params["batch_size"]
    buffer_capacity = params["max_replay_buffer_size"]
    a1_lr = params["agent1_learning_rate"]
    a2_lr = params["agent2_learning_rate"]
    gamma = params["gamma"]
    max_timesteps = params["sequence_length"]
    start_episode = params["start_episode"]
    env_debug_mode = params["env_debug_mode"]
    env_reward_shaping = params["enable_reward_shaping"]
    a1_epsilon_start = params["a1_epsilon_start"]
    a1_epsilon_end = params["a1_epsilon_end"]
    a2_epsilon_start = params["a2_epsilon_start"]
    a2_epsilon_end = params["a2_epsilon_end"]
    tau = params["tau"]

    episode_framecount = 0
    agent_1_score = 0
    agent_2_score = 0
    draw_score = 0
    agent_1_starts = 0
    agent_2_starts = 0
    agent_1_reward = 0
    agent_2_reward = 0
    update_frequency = 100
    ave_steps_per_game = 10
    padding_value = 0  # Define the padding value
    sequence_length = max_timesteps

    # Move agents to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Replay buffers
    replay_buffer1 = ReplayBuffer(buffer_capacity, max_timesteps)
    replay_buffer2 = ReplayBuffer(buffer_capacity, max_timesteps)

   # Environment
    env = TwoPlayerConnectFourEnv(writer=writer)

    replay_buffer1.set_env(env)
    replay_buffer2.set_env(env)

    # Load agents and replay buffer
    if len(sys.argv) == 5:  # Expected: script name, agent1 weights, agent2 weights, buffer
        agent1, agent2, replay_buffer1, replay_buffer2 = load_agents_and_buffer(cnn_a1, cnn_a2, fc_a1, fc_a2, sequence_length, buffer_capacity, sys.argv[2], sys.argv[3], sys.argv[4])        
    elif len(sys.argv) == 4:  # Only agents, no buffer
        agent1, agent2, replay_buffer1, replay_buffer2 = load_agents_and_buffer(cnn_a1, cnn_a2, fc_a1, fc_a2, sequence_length, buffer_capacity, sys.argv[2], sys.argv[3])
    else:
        agent1, agent2, replay_buffer1, replay_buffer2 = load_agents_and_buffer(cnn_a1, cnn_a2, fc_a1, fc_a2, sequence_length, buffer_capacity)  
 

    agent1 = agent1.to(device)
    agent2 = agent2.to(device)

    
    # Create target networks for policy stabilization
    agent1_tgt = deepcopy(agent1)  # Create target networks
    agent2_tgt = deepcopy(agent2)
    agent1_tgt = agent1_tgt.to(device)
    agent2_tgt = agent2_tgt.to(device)
    print(f'agent1.device: {agent1.device}')
    print(f'agent2.device: {agent2.device}')
    print(f'agent1_tgt.device: {agent1_tgt.device}')
    print(f'agent2_tgt.device: {agent2_tgt.device}')

    # Optimizers
    optimizer1 = optim.Adam(agent1.parameters(), lr=a1_lr)
    optimizer2 = optim.Adam(agent2.parameters(), lr=a2_lr)



    # Load models and replay buffers if checkpoint file is provided
    if len(sys.argv) == 3:
        agent1, agent2, optimizer1, optimizer2, replay_buffer1, replay_buffer2 = (
        load_all_models(agent1, agent2, 
                        optimizer1, optimizer2, 
                        replay_buffer1, replay_buffer2, 
                        hyp_file_root, sys.argv[2]
        ))

    # Set environment parameters
    env.enable_reward_shaping(str_to_bool(env_reward_shaping))
    env.enable_debug_mode(str_to_bool(env_debug_mode))

    # Initialization at the start of an episode
    state, current_player = env.reset()  # Reset the environment and get the initial state
    states1 = deque(maxlen=max_timesteps)  # Initialize state sequence for player 1
    states2 = deque(maxlen=max_timesteps)  # Initialize state sequence for player 2
    # No need to pre-fill; start empty and fill with actual gameplay data
    states1.append(state)  # Append the initial state
    states2.append(state)  # Append the initial state
    actions1, rewards1, next_states1, dones1, masks1 = [], [], [], [], []
    actions2, rewards2, next_states2, dones2, masks2 = [], [], [], [], []

    # Assume pre-allocated tensors: states, actions, rewards, next_states, dones, masks
    for episode in range(start_episode, end_episode):
        state, current_player = env.reset()
        done = False
        timestep = 0
        episode_framecount += 1
        epsilon1 = agent1.get_epsilon(episode_framecount, end_episode, a1_epsilon_start, a1_epsilon_end)
        epsilon2 = agent2.get_epsilon(episode_framecount, end_episode, a2_epsilon_start, a2_epsilon_end)

        while not done:
            valid_actions = env.get_valid_actions()

            print(f'prior to select_action: state.shape: {state.shape}')
            if current_player == 1:
                action = agent1.select_action(state, valid_actions, epsilon1)
                next_state, reward, done, _ = env.step(action)

                replay_buffer1.push(state, action, reward, next_state, done)
                if len(replay_buffer1) >= batch_size:
                    train(agent1, agent1_tgt, optimizer1, replay_buffer1, batch_size, gamma, sequence_length, env, tau)
            else:
                action = agent2.select_action(state, valid_actions, epsilon2)
                next_state, reward, done, _ = env.step(action)
                replay_buffer2.push(state, action, reward, next_state, done)
                if len(replay_buffer2) >= batch_size:
                    train(agent2, agent2_tgt, optimizer2, replay_buffer2, batch_size, gamma, sequence_length, env, tau)


            state = next_state
            timestep += 1


            # Update target network if necessary
            if timestep % params['update_target_network_every'] == 0:
                soft_update(agent1_tgt, agent1, tau)
                soft_update(agent2_tgt, agent2, tau)

            print(f'Episode {episode+1} completed')    


        # Reset the batch_index for the next episode
        batch_index = (batch_index + 1) % batch_size
        if batch_index == 0:

            # When preparing a batch for training
            if len(states1) < max_timesteps:
                padded_states1 = list(states1) + [padding_value] * (max_timesteps - len(states1))
                padded_states2 = list(states2) + [padding_value] * (max_timesteps - len(states2))
            else:
                padded_states1 = list(states1)
                padded_states2 = list(states2)
            # Push the sequences to the replay buffer after the episode is complete
            replay_buffer1.push(padded_states1, actions1, rewards1, next_states1, dones1, masks1)
            replay_buffer2.push(padded_states2, actions2, rewards2, next_states2, dones2, masks2)

            # Train the agents
            agent1.train(states1, actions1, rewards1, next_states1, dones1, masks1)
            agent2.train(states2, actions2, rewards2, next_states2, dones2, masks2)



            # Update target networks
            if episode % update_frequency == 0:
                soft_update(agent1_tgt, agent1, tau)
                soft_update(agent2_tgt, agent2, tau)


            # Log the training status
            if episode % params["console_status_interval"] == 0:
                print(f'Episode {episode} of {end_episode}')
                print(f'Agent 1 epsilon: {epsilon1}')
                print(f'Agent 2 epsilon: {epsilon2}')
                print(f'Agent 1 loss: {agent1.loss}')
                print(f'Agent 2 loss: {agent2.loss}')

            if episode % params["tensorboard_status_interval"] == 0:
                writer.add_scalar('Agent 1/Epsilon', epsilon1, episode)
                writer.add_scalar('Agent 2/Epsilon', epsilon2, episode)
                writer.add_scalar('Agent 1/Loss', agent1.loss, episode)
                writer.add_scalar('Agent 2/Loss', agent2.loss, episode)



            if done and (episode % params["ckpt_interval"] == 0):   
                save_all_models(agent1, optimizer1, "a1", replay_buffer1, f"./wts/chkpt_{hyp_file_root}_{episode}")
                save_all_models(agent2, optimizer2, "a2", replay_buffer2, f"./wts/chkpt_{hyp_file_root}_{episode}")
                

            if check_e_keypress():
                print('Keyboard Keypress -e- detected, exiting training loop')
                save_all_models(agent1, optimizer1, "a1", replay_buffer1, f"./wts/chkpt_{hyp_file_root}_{episode}")
                save_all_models(agent2, optimizer2, "a2", replay_buffer2, f"./wts/chkpt_{hyp_file_root}_{episode}")
                break

    writer.close()

    print(f'\nTraining ended on episode count: {episode}')
                       
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time                       

    test_results_string = '\n\n----------- Training Results ----------------\n' 
    test_results_string += f'Started training at: \t{start_time.strftime("%Y-%m-%d  %H:%M:%S")}\n'
    test_results_string += f'Ended training at: \t{end_time.strftime("%Y-%m-%d  %H:%M:%S")}\n'
    test_results_string += f'Total training time:  {str(elapsed_time)}\n'
    test_results_string += f'start_episode: {start_episode}\n'
    test_results_string += f'end_episode: {end_episode}\n'
    test_results_string += f'Episode count: {episode}\n'
    test_results_string += f'agent1 end epsilon: {epsilon1}\n'
    test_results_string += f'agent2 end epsilon: {epsilon2}\n'
    test_results_string += f'Draws: {draw_score}\n'
    if agent_2_starts > 0:
        test_results_string += f'agent_1_starts / agent_2_starts {agent_1_starts / agent_2_starts}\n'
        test_results_string += f'agent_1_reward / agent_2_reward {agent_1_reward / agent_2_reward}\n'
    test_results_string += f'Ave steps per game: {ave_steps_per_game:.2f}\n'

    test_results_string += f'Input Parameters:\n'
    test_results_string += print_parameters(params)

    if episode != end_episode-1:
        test_results_string += f'\n**Exited training loop early at episode {episode}'
        test_results_string += f'\nstart_episode = {episode}\n'

    print(test_results_string)
    save_test_results_with_hyps(hyp_file,test_results_string)

       
def save_test_results_with_hyps(hyp_file, test_results_string):
    try:
        with open(hyp_file, 'r+') as file:
            #append to the end of the file
            file.seek(0, os.SEEK_END)
            file.write(f'{test_results_string}\n')  # Write the "new results"
    except Exception as e:
        print(e)



def check_e_keypress():
    stdscr = curses.initscr()
    curses.cbreak()
    stdscr.nodelay(1)  # set getch() non-blocking
    key = stdscr.getch()
    curses.endwin()
    if key != ord('e'):  
        return False
    else:
        return True
            

if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")


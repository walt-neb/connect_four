


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

def train(agent, agent_tgt, optimizer, replay_buffer, batch_size, gamma, sequence_length, env, tau=0.005):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Ensure enough samples for a batch
    num_complete_sequences = sum(1 for seq in replay_buffer.buffer if len(seq) == sequence_length)
    if num_complete_sequences < batch_size:
        return torch.tensor(0.0, device=device)
    
    # Ensure the replay buffer is a list or convert it temporarily for sampling
    if isinstance(replay_buffer, (set, dict)):
        replay_buffer = list(replay_buffer)  # Convert to list if not already

    # Sample a batch of sequences from the replay buffer
    transitions = replay_buffer.sample(batch_size)


    #Assuming transitions is a list of sequences and each sequence is a list of tuples (state, action, reward, next_state, done)
    for i, seq in enumerate(transitions):
        print(f"\n\ntrain() - Rendering sequence {i+1}/{len(transitions)}")
        for j, transition in enumerate(seq):
            state = transition[0]  # This is the flattened state array
            # Reshape the flattened state back into a 6x7 board
            reshaped_state = state.reshape((6, 7))
            print(f"Board State: {j+1}/{len(seq)}")
            env.render(reshaped_state)  
            print(f"Reward: {transition[2]}, Done: {transition[4]}")
            reshaped_next_state = transition[3].reshape((6, 7))
            print(f"Next State:")
            env.render(reshaped_next_state)
            print("\n")



    # Extract and pad sequences
    state_seqs = []
    next_state_seqs = []
    actions = []
    rewards = []
    dones = []

    for seq in transitions:
        state_seqs.append(np.array([s[0] for s in seq]))
        next_state_seqs.append(np.array([s[3] for s in seq]))
        actions.append(seq[-1][1])  # Last action
        rewards.append(seq[-1][2])  # Last reward
        dones.append(seq[-1][4])  # Last done flag
        
    # Pad sequences to have the same length
    state_seqs = [
        np.pad(s, ((0, max(0, sequence_length - len(s))), (0, 0)), mode='constant', constant_values=0)
        for s in state_seqs ]
    next_state_seqs = [
        np.pad(s, ((0, max(0, sequence_length - len(s))), (0, 0)), mode='constant', constant_values=0) 
        for s in next_state_seqs ]

    # Convert to tensors
    state_seqs = torch.tensor(state_seqs, dtype=torch.float32).view(batch_size, 1, sequence_length, 6, 7).to(device)
    next_state_seqs = torch.tensor(next_state_seqs, dtype=torch.float32).view(batch_size, 1, sequence_length, 6, 7).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.bool).to(device)

    # Current Q values using the main network (modified for sequences)

    agent_output = agent(state_seqs)
    #print("Output dimensions from agent:", agent_output.shape)
    #[batch_size, number_of_actions] (2D tensor)
    curr_q_values = agent_output.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Select best actions in next state using the main network (modified for sequences)
    # Add a channel dimension and ensure the correct order
    next_state_seqs = next_state_seqs.unsqueeze(2)  # Adds a channel dimension
    next_state_seqs = next_state_seqs.permute(0, 2, 1, 3, 4)  # Reorder to [batch, channel, depth, height, width]
    next_actions = agent(next_state_seqs)[:, -1, :].argmax(1, keepdim=True)  # Take actions of last state

    # Evaluate these next actions using the target network (modified for sequences)
    next_q_values = agent_tgt(next_state_seqs)[:, -1, :].gather(1, next_actions).squeeze(1)  # Take Q-values of last state
    next_q_values[dones] = 0  # No next state if the game is done
   

    # Compute the target Q values using the Bellman equation
    target_q_values = rewards + (gamma * next_q_values)

    # Calculate the MSE loss
    loss = nn.MSELoss()(curr_q_values, target_q_values)

    # Gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Soft update of the target network
    soft_update(agent_tgt, agent, tau=0.005)

    return loss

def pad_sequence(sequence, seq_len, pad_value=-1):
    padding_size = seq_len - len(sequence)
    if padding_size > 0:
        # Pad with arrays filled with the padding value
        padded_sequence = sequence + [np.full((6, 7), pad_value) for _ in range(padding_size)]
    else:
        padded_sequence = sequence
    return padded_sequence

# Example masking in loss calculation
def masked_loss(output, target, mask):
    masked_output = torch.masked_select(output, mask)
    masked_target = torch.masked_select(target, mask)
    return torch.nn.functional.mse_loss(masked_output, masked_target)

# Assuming `mask` is a tensor of the same shape as `output` and `target`,
# containing True for valid data points and False for padded data points.

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
        fc_a1 = ast.literal_eval(params["fc_a1"])
        fc_a2 = ast.literal_eval(params["fc_a2"])
        render_games = ast.literal_eval(params["render_game_at"])
    except Exception as e:
        print(e)
        sys.exit(1)

    cnn_a1 = ensure_list_of_tuples(cnn_a1)
    cnn_a2 = ensure_list_of_tuples(cnn_a2)
  


    max_rp_size = params["max_replay_buffer_size"]
    seq_len = params["sequence_length"]
    end_episode = params["end_episode"]
    batch_size = params["batch_size"]
    buffer_capacity = params["max_replay_buffer_size"]
    a1_lr = params["agent1_learning_rate"]
    a2_lr = params["agent2_learning_rate"]

    # Check command line arguments
    if len(sys.argv) == 5:  # Expected: script name, agent1 weights, agent2 weights, buffer
        agent1, agent2, replay_buffer = load_agents_and_buffer(cnn_a1, cnn_a2, fc_a1, fc_a2, seq_len, max_rp_size, sys.argv[2], sys.argv[3], sys.argv[4])        
    elif len(sys.argv) == 4:  # Only agents, no buffer
        agent1, agent2, replay_buffer = load_agents_and_buffer(cnn_a1, cnn_a2, fc_a1, fc_a2, seq_len, max_rp_size, sys.argv[2], sys.argv[3])
    else:
        agent1, agent2, replay_buffer = load_agents_and_buffer(cnn_a1, cnn_a2, fc_a1, fc_a2, seq_len, max_rp_size)  
    
    assert isinstance(replay_buffer, deque), "Replay buffer is not a deque!"

    # Move agents to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent1 = agent1.to(device)
    agent2 = agent2.to(device)
    
    # Create target networks for policy stabilization
    agent1_tgt = agent1
    agent2_tgt = agent2
    agent1_tgt = agent1_tgt.to(device)
    agent2_tgt = agent2_tgt.to(device)
    print(f'agent1.device: {agent1.device}')
    print(f'agent2.device: {agent2.device}')
    print(f'agent1_tgt.device: {agent1_tgt.device}')
    print(f'agent2_tgt.device: {agent2_tgt.device}')


    # Optimizers
    optimizer1 = optim.Adam(agent1.parameters(), lr=a1_lr)
    optimizer2 = optim.Adam(agent2.parameters(), lr=a2_lr)


    # Environment
    env = TwoPlayerConnectFourEnv(writer=writer)

    # Main training loop
    replay_buffer1 = ReplayBuffer(buffer_capacity, seq_len)
    replay_buffer2 = ReplayBuffer(buffer_capacity, seq_len)
    replay_buffer1.set_env(env)
    replay_buffer2.set_env(env)


    # Load models and replay buffers if checkpoint file is provided
    if len(sys.argv) == 3:
        agent1, agent2, optimizer1, optimizer2, replay_buffer1, replay_buffer2 = (
        load_all_models(agent1, agent2, 
                        optimizer1, optimizer2, 
                        replay_buffer1, replay_buffer2, 
                        hyp_file_root, sys.argv[2]
        ))

    episode_framecount = 0
    agent_1_score = 0
    agent_2_score = 0
    draw_score = 0
    agent_1_starts = 0
    agent_2_starts = 0
    agent_1_reward = 0
    agent_2_reward = 0
    update_frequency = 100
    gamma = params["gamma"]
    steps_per_game = 0
    ave_steps_per_game = 10

    run_number = 1

    if len(sys.argv) == 3:
        start_episode = int(sys.argv[2])
    else:
        start_episode = params["start_episode"]

    env.enable_reward_shaping(str_to_bool(params["enable_reward_shaping"]))
    env.enable_debug_mode(str_to_bool(params["env_debug_mode"]))

    sequence_length = params["sequence_length"]

    state, current_player = env.reset()       # Reset the environment and get the initial state
    state_seq = deque(maxlen=sequence_length)  # Initialize state sequence
    for _ in range(sequence_length):
        state_seq.append(state)  # Fill with initial state for first sequence

    print("Initial state_seq length:", len(state_seq))

    for episode in range(start_episode, end_episode):
        state, current_player = env.reset()

        #current_player = 1 # Start with agent 1

        done = False
        total_loss1 = 0
        total_loss2 = 0
        num_steps1 = 0
        num_steps2 = 0
        episode_framecount += 1
        if current_player == 1:
            agent_1_starts += 1
        else:
            agent_2_starts += 1
        if agent_2_starts > 0:
            #writer.add_scalar(f'agent_1_starts/agent_2_starts={agent_1_starts/agent_2_starts:.3f}', episode)
            pass

        state_seq1 = deque(maxlen=sequence_length)  # Initialize the state sequence for the episode
        state_seq2 = deque(maxlen=sequence_length)  # Initialize the state sequence for the episode



        while not done:
            logthis = False
            valid_actions = env.get_valid_actions()
            epsilon1 = agent1.get_epsilon(episode_framecount, end_episode, params["a1_epsilon_start"], params["a1_epsilon_end"])
            epsilon2 = agent2.get_epsilon(episode_framecount, end_episode, params["a2_epsilon_start"], params["a2_epsilon_end"])
            
            # Determine which agent is playing
            if current_player == 1:
                action1 = agent1.select_action(state_seq1, valid_actions, epsilon1) 
                board, next_state1, reward1, done, next_player = env.step(action1)
                num_steps1 += 1

                if not done:
                    # Construct next_state_seq (assuming sequence_length = 7)
                    next_state_seq1 = deque(state_seq1)  # Create a new deque without maxlen
                    next_state_seq1.append(next_state1)


                # Push sequences to replay buffers
                replay_buffer1.push(list(state_seq1), action1, reward1, list(next_state_seq1), done) 
                #display status of replay buffer
                
                print(f'\nmain() - Pushed to replay buffer 1 for episode={episode}, game moves={num_steps1 + num_steps2}:')
                print(f'replay1.get_len()={replay_buffer1.get_len()}')
                #print(f'replay1.buffer[0]={replay_buffer1.buffer[0]}')
                print(f'current_state1')
                env.render(replay_buffer1.buffer[0][0][0]) #state
                print(f'next_state1')
                env.render(replay_buffer1.buffer[0][3][0]) #next state

                # Update the state sequence (append and remove the oldest state)
                state_seq1 = deque(state_seq1)  # Create a new deque without maxlen                
                state_seq1.append(next_state1)

                loss1 = train(agent1, agent1_tgt, optimizer1, replay_buffer1, batch_size, gamma, seq_len, env, tau=0.005)

                # Update target network using soft updates
                if episode % update_frequency == 0:
                    soft_update(agent1_tgt, agent1, .05)
                
                if loss1 is not None:
                    total_loss1 += loss1.item()
                    
            else:
                action2 = agent2.select_action(state_seq2, valid_actions, epsilon2) 
                board, next_state2, reward2, done, next_player = env.step(action2)
                num_steps2 += 1

                if not done:
                    # Construct next_state_seq (assuming sequence_length = 7)
                    next_state_seq2 = deque(state_seq2)  # Create a new deque without maxlen
                    next_state_seq2.append(next_state2)
          

                # Push sequences to replay buffers
                replay_buffer2.push(list(state_seq2), action2, reward2, list(next_state_seq2), done) 
                #display status of replay buffer
                print(f'\nmain() - Pushed to replay buffer 2 for episode={episode}, game moves={num_steps1 + num_steps2}:')
                print(f'replay2.get_len()={replay_buffer2.get_len()}')
                print(f'current_state2')
                env.render(replay_buffer2.buffer[0][0][0]) #state
                print(f'next_state2')
                env.render(replay_buffer2.buffer[0][3][0]) #next state


                # Update the state sequence (append and remove the oldest state)
                state_seq2 = deque(state_seq2)  # Create a new deque without maxlen
                state_seq2.append(next_state2)

                loss2 = train(agent2, agent2_tgt, optimizer2, replay_buffer2, batch_size, gamma, seq_len, env, tau=0.005)

                # Update target network using soft updates
                if episode % update_frequency == 0:
                    soft_update(agent2_tgt, agent2, .05)

                if loss2 is not None:
                    total_loss2 += loss2.item()
                   

            if episode in render_games: # or steps_per_game > 40: 
                #logthis = True
                print(f"main() ----Episode {episode} Step {num_steps1 + num_steps2}")                
                winner = env.render(None) 
                #print(f"Agent {current_player} playing ({env.get_player_symbol(current_player)}) takes action {action} receives reward of {reward:.2f}")   
                #input("Press key to continue...")

            if not done:
                state1 = next_state1
                state2 = next_state2
                current_player = next_player
            #print('.', end='')

        # On episode completion
        if done:
            replay_buffer1.push(list(state_seq), action1, reward1, [], done)  # Push the final state with an empty next_state
            replay_buffer2.push(list(state_seq), action2, reward2, [], done)  # Push the final state with an empty next_state
            state_seq1.clear()  # Clear for the next game
            state_seq2.clear()  # Clear for the next game
            print(f"Episode finished. Sequence reset.")

        # track the scores
        if env.winner == 1:
            agent_1_score += 1
            agent_1_reward += reward
        elif env.winner == 2:
            agent_2_score += 1
            agent_2_reward += reward
        elif env.winner == None:
            draw_score += 1

        steps_per_game = (num_steps1 + num_steps2)
        if ave_steps_per_game == 0:
            ave_steps_per_game = steps_per_game
        else:
            ave_steps_per_game = 0.97*ave_steps_per_game + 0.03*steps_per_game
        print(f'\t{steps_per_game:02d} moves in {episode} of {end_episode}. Ave moves: {ave_steps_per_game:.3f} - Press (e) for early exit')

        if done and (episode % params["console_status_interval"] == 0 or logthis==True):  # Render end of the game for specified episodes
            print(f'----Episode {episode} of {end_episode}--------')
            winner = env.render(None)
            print(f"Episode {episode} Step {num_steps1 + num_steps2}")
            print(f"Agent {current_player} ({env.get_player_symbol(current_player)}) action: {action}")
            if winner is not None:
                print(f"Agent {winner} wins")  
            print(f'Agent 1: {agent_1_score}')
            print(f'Agent 2: {agent_2_score}, Draws: {draw_score}')
            print(f'Agent 1 epsilon: {epsilon1}')
            print(f'Agent 2 epsilon: {epsilon2}')
            if num_steps1 > 0 and num_steps2 > 0:
                print(f'Agent 1 loss: {total_loss1 / num_steps1}')
                print(f'Agent 2 loss: {total_loss2 / num_steps2}')
            if agent_2_reward > 0:
                print(f'A1 reward / A2 reward: {agent_1_reward / agent_2_reward:.3f}')
            if episode > 0:
                print(f'Win Rates -> Agent 1: {agent_1_score/episode:.4f}')
                print(f'             Agent 2: {agent_2_score/episode:.4f}')

        if done and (episode % params["tensorboard_status_interval"] == 0 or logthis==True): 
            # Correct the scalar tags to be consistent and not include dynamic values
            writer.add_scalar('Agent 1/Score', agent_1_score, episode)
            writer.add_scalar('Agent 2/Score', agent_2_score, episode)
            writer.add_scalar('Agent 1/Epsilon', epsilon1, episode)
            writer.add_scalar('Agent 2/Epsilon', epsilon2, episode)

            if num_steps1 > 0 and num_steps2 > 0:
                writer.add_scalar('Agent 1/Loss', total_loss1 / num_steps1, episode)
                writer.add_scalar('Agent 2/Loss', total_loss2 / num_steps2, episode)

            writer.add_scalar('Comp/StepsPerGame', steps_per_game, episode)

            if False and agent_2_starts > 0:
                writer.add_scalar('Comp/Ratio Agent_1_over_Agent_2 as Player1', agent_1_starts / agent_2_starts, episode)

            writer.add_scalar('Comp/Draws', draw_score, episode)

            if agent_2_reward > 0:
                writer.add_scalar('Comp/Ratio Agent_1_to_Agent_2 Reward', agent_1_reward / agent_2_reward, episode)

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
    if num_steps1 > 0 and num_steps2 > 0:
        test_results_string += f'total_loss1 / num_steps1: {total_loss1 / num_steps1}\n'
        test_results_string += f'total_loss2 / num_steps2: {total_loss2 / num_steps2}\n'
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


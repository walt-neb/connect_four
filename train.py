

#filename:  train.py
import cProfile
import pstats
import time

import os
import sys
import pickle
import numpy as np
import dqn_agent
from dqn_agent import DQNAgent 
from two_player_env import ConnectFour
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import math

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch
import pickle
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

def print_parameters(params):
    if not params:
        print("The parameters dictionary is empty.")
        return

    print("*** Training Parameters: ")
    for key, value in params.items():
        print(f"\t\t{key} = {value}")


def save_checkpoint(model, optimizer, replay_buffer, episode, checkpoint_path):
    # Save model state
    model_state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'episode': episode  # Include the episode number
    }
    torch.save(model_state, checkpoint_path + '_model.ckpt')

    # Save replay buffer and episode number using pickle
    buffer_state = {
        'replay_buffer': replay_buffer,
        'episode': episode
    }
    with open(checkpoint_path + '_buffer.pkl', 'wb') as f:
        pickle.dump(buffer_state, f)

def load_checkpoint(checkpoint_path, model, optimizer, device):
    # Load model state
    model_checkpoint = torch.load(checkpoint_path + '_model.ckpt', map_location=device)
    model.load_state_dict(model_checkpoint['state_dict'])
    optimizer.load_state_dict(model_checkpoint['optimizer'])
    episode = model_checkpoint['episode']

    # Load replay buffer
    with open(checkpoint_path + '_buffer.pkl', 'rb') as f:
        buffer_state = pickle.load(f)
    replay_buffer = buffer_state['replay_buffer']
    assert episode == buffer_state['episode'], "Episode mismatch between model and buffer state."

    return model, optimizer, replay_buffer, episode

def load_agents_and_buffer(a1_layer_dims, a2_layer_dims, agent1_wts=None, agent2_wts=None, rp_buffer_file=None):
    # Initialize agents
    input_dim = 6 * 7  # Board size
    output_dim = 7     # One output for each column
    print(f'a1_layer_dims: {a1_layer_dims}')
    print(f'a2_layer_dims: {a2_layer_dims}')
    agent1 = DQNAgent(input_dim, output_dim, a1_layer_dims)
    agent2 = DQNAgent(input_dim, output_dim, a2_layer_dims)


    replay_buffer = deque(maxlen=10000)  # Adjust size as needed

    # Load weights if paths are provided
    if agent1_wts and os.path.exists(agent1_wts):
        '''
        checkpoint = torch.load(agent1_wts)
        for key, tensor in checkpoint.items():
            print(f"{agent1_wts}: {key}: {tensor.size()}")  
        '''          
        agent1.load_state_dict(torch.load(agent1_wts))
        agent1.eval()
        print(f"Loaded weights for Agent 1 from {agent1_wts}.")
    else:
        print(f"Starting Agent 1 from scratch or file not found: {agent1_wts}")

    if agent2_wts and os.path.exists(agent2_wts):
        '''
        checkpoint = torch.load(agent2_wts)
        for key, tensor in checkpoint.items():
            print(f"{agent2_wts}: {key}: {tensor.size()}")  
        '''
        agent2.load_state_dict(torch.load(agent2_wts))
        agent2.eval()
        print(f"Loaded weights for Agent 2 from {agent2_wts}.")
    else:
        print(f"Starting Agent 2 from scratch or file not found: {agent2_wts}")

    if rp_buffer_file and os.path.exists(rp_buffer_file):
        with open(rp_buffer_file, 'rb') as f:
            replay_buffer = pickle.load(f)
        print(f"Loaded replay buffer from {rp_buffer_file}.")
    else:
        print(f"Starting new replay buffer or file not found: {rp_buffer_file}")

    return agent1, agent2, replay_buffer


def soft_update(target_model, source_model, tau=0.005):
    """
    Softly update the target model's weights using the weights from the source model.
    tau is a small coefficient.
    """
    for target_param, param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)        

def train(agent, agent_tgt, optimizer, replay_buffer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return torch.tensor(0.0)  # Not enough samples to train so return 0 loss

    # Double DQN training
    transitions = replay_buffer.sample(batch_size)
    batch = list(zip(*transitions))

    states = torch.tensor(np.array(batch[0]), dtype=torch.float32)
    actions = torch.tensor(batch[1], dtype=torch.long)
    rewards = torch.tensor(batch[2], dtype=torch.float32)
    next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32)
    dones = torch.tensor(batch[4], dtype=torch.bool)

    # Current Q values using the main network
    curr_q_values = agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Select best actions in next states using the main network
    next_actions = agent(next_states).argmax(1, keepdim=True)

    # Evaluate these next actions using the target network
    next_q_values = agent_tgt(next_states).gather(1, next_actions).squeeze(1)
    next_q_values[dones] = 0  # No next state if the game is done

    # Compute the target Q values using the Bellman equation
    target_q_values = rewards + (gamma * next_q_values)

    # Calculate the MSE loss
    loss = nn.MSELoss()(curr_q_values, target_q_values)

    # Gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

dbmode=1

def main():

    print(f'torch.cuda.is_available()={torch.cuda.is_available()}')

    if len(sys.argv) < 2:
        print("Usage: python train.py <hyperparameters_file> <agent1_weights> <agent2_weights> <replay_buffer>")
        return
    hyp_file = sys.argv[1]
    hyp_file_root = hyp_file.rstrip('.hyp')

    writer = SummaryWriter(f'runs/{hyp_file_root}_connect_four_experiment')
    params = load_hyperparams(hyp_file)
    print_parameters(params)

    try:
        agent_1_layer_dims_string = params["agent_1_layer_dims"].strip('[]')
        agent_1_layer_dims_string = [int(dim) for dim in agent_1_layer_dims_string.split()]
        agent_2_layer_dims_string = params["agent_2_layer_dims"].strip('[]')
        agent_2_layer_dims_string = [int(dim) for dim in agent_2_layer_dims_string.split()]
        render_games = params["render_game_at"].strip('[]')
        render_games = [int(game) for game in render_games.split()]
    except Exception as e:
        print(e)
    print(f'agent_1_layer_dims_string: {agent_1_layer_dims_string}')
    print(f'agent_2_layer_dims_string: {agent_2_layer_dims_string}')
    print(f'render_games: {render_games}')

    # Check command line arguments
    if len(sys.argv) == 5:  # Expected: script name, agent1 weights, agent2 weights, buffer
        agent1, agent2, replay_buffer = load_agents_and_buffer(agent_1_layer_dims_string, agent_2_layer_dims_string, sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 4:  # Only agents, no buffer
        agent1, agent2, replay_buffer = load_agents_and_buffer(agent_1_layer_dims_string, agent_2_layer_dims_string, sys.argv[2], sys.argv[3])
    else:
        agent1, agent2, replay_buffer = load_agents_and_buffer(agent_1_layer_dims_string, agent_2_layer_dims_string)  
    
    # Create target networks for policy stabilization
    agent1_tgt = agent1
    agent2_tgt = agent2

    # Move agents to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #agent1.to(device)
    #agent2.to(device)
    # Optimizers
    optimizer1 = optim.Adam(agent1.parameters(), lr=params["agent1_learning_rate"])
    optimizer2 = optim.Adam(agent2.parameters(), lr=params["agent1_learning_rate"])


    # Environment
    env = ConnectFour()

    # Main training loop
    end_episode = params["end_episode"]
    batch_size = params["batch_size"]
    buffer_capacity = params["max_replay_buffer_size"]

    replay_buffer1 = ReplayBuffer(buffer_capacity)
    replay_buffer2 = ReplayBuffer(buffer_capacity)

    episode_framecount = 0

    agent_1_score = 0
    agent_2_score = 0
    draw_score = 0
    agent_1_starts = 0
    agent_2_starts = 0
    agent_1_reward = 0
    agent_2_reward = 0
    update_frequency = 1000
    gamma = params["gamma"]

    start_episode = 0

    #agent1, optimizer1, replay_buffer, start_episode = load_checkpoint('path_to_checkpoint', agent1, optimizer1, device)

    for episode in range(start_episode, end_episode):
        state, active_agent = env.reset()

        active_agent = 1 # Start with agent 1

        done = False
        total_loss1 = 0
        total_loss2 = 0
        num_steps1 = 0
        num_steps2 = 0
        episode_framecount += 1
        if active_agent == 1:
            agent_1_starts += 1
        else:
            agent_2_starts += 1
        if agent_2_starts > 0:
            #writer.add_scalar(f'agent_1_starts/agent_2_starts={agent_1_starts/agent_2_starts:.3f}', episode)
            pass

        while not done:
            logthis = False
            valid_actions = env.get_valid_actions()
            epsilon1 = agent1.get_epsilon(episode_framecount, end_episode, params["a1_epsilon_start"], params["a1_epsilon_end"])
            epsilon2 = agent2.get_epsilon(episode_framecount, end_episode, params["a2_epsilon_start"], params["a2_epsilon_end"])
            
            
            # Determine which agent is playing
            if active_agent == 1:
                action = agent1.select_action(state, valid_actions, epsilon1)
                next_state, reward, done, next_player = env.step(action)
                replay_buffer1.push(state, action, reward, next_state, done)
                loss1 = train(agent1, agent1_tgt, optimizer1, replay_buffer1, batch_size, gamma)
                # Update target network using soft updates
                if episode % update_frequency == 0:
                    soft_update(agent1_tgt, agent1, .05)
                
                if loss1 is not None:
                    total_loss1 += loss1.item()
                    num_steps1 += 1
            else:
                action = agent2.select_action(state, valid_actions, epsilon2)
                next_state, reward, done, next_player = env.step(action)
                replay_buffer2.push(state, action, reward, next_state, done)
                loss2 = train(agent2, agent2_tgt, optimizer2, replay_buffer2, batch_size, gamma)
                if loss2 is not None:
                    total_loss2 += loss2.item()
                    num_steps2 += 1

            if episode in render_games: 
                logthis = True
                print(f"----Episode {episode} Step {num_steps1 + num_steps2}")                
                winner = env.render() 
                print(f"Agent {active_agent} playing ({env.get_player_symbol(active_agent)}) takes action {action} receives reward of {reward}")   
                #input("Press key to continue...")

            if not done:
                state = next_state
                active_agent = next_player
            print('.', end='')

        # track the scores
        if env.winner == 1:
            agent_1_score += 1
            agent_1_reward += reward
        elif env.winner == 2:
            agent_2_score += 1
            agent_2_reward += reward
        elif env.winner == 0:
            draw_score += 1

        print(f'Episode {episode} of {end_episode} completed.')

        if done and (episode % params["console_status_interval"] == 0 or logthis==True):  # Render end of the game for specified episodes
            print(f'----Episode {episode} of {end_episode}--------')
            winner = env.render()
            print(f"Episode {episode}, Step {num_steps1 + num_steps2}")
            print(f"Agent {active_agent} ({env.get_player_symbol(active_agent)}) action: {action}")
            if winner is not None:
                print(f"Agent {winner} wins")  
            print(f'Agent 1: {agent_1_score}')
            print(f'Agent 2: {agent_2_score}, Draws: {draw_score}')
            print(f'Agent 1 epsilon: {epsilon1}')
            print(f'Agent 2 epsilon: {epsilon2}')
            if num_steps1 > 0 and num_steps2 > 0:
                print(f'Agent 1 loss: {total_loss1 / num_steps1}')
                print(f'Agent 2 loss: {total_loss2 / num_steps2}')
            #print(f'Agent 1 replay buffer size: {len(replay_buffer1)}')
            #print(f'Agent 2 replay buffer size: {len(replay_buffer2)}')
            if agent_2_starts > 0:
                print(f'A1 over A2 as Player1: {agent_1_starts/agent_2_starts:.3f}')
            print(f'A1 reward - A2 reward: {agent_1_reward - agent_2_reward:.3f}')
            if agent_2_reward > 0:
                print(f'A1 reward // A2 reward: {agent_1_reward / agent_2_reward:.3f}')

            if episode > 0:
                print(f'Win Rates -> Agent 1: {agent_1_score/episode:.4f}')
                print(f'             Agent 2: {agent_2_score/episode:.4f}')

            agent1_filename = f'agent1_{hyp_file_root}.wts'
            agent2_filename = f'agent2_{hyp_file_root}.wts'

        if done and (episode % params["tensorboard_status_interval"] == 0 or logthis==True): 
            writer.add_scalar(f'Agent 1: {agent_1_score}')
            writer.add_scalar(f'Agent 2: {agent_2_score}, Draws: {draw_score}', episode)
            writer.add_scalar(f'Agent 1 epsilon: {epsilon1}', episode)
            writer.add_scalar(f'Agent 2 epsilon: {epsilon2}', episode)
            if num_steps1 > 0 and num_steps2 > 0:
                writer.add_scalar(f'Agent 1 loss: {total_loss1 / num_steps1}', episode)
                writer.add_scalar(f'Agent 2 loss: {total_loss2 / num_steps2}', episode)
            writer.add_scalar(f'Agent 2 replay buffer size: {len(replay_buffer2)}', episode)
            if agent_2_reward > 0:
                writer.add_scalar(f'A1/A2 reward: {agent_1_reward / agent_2_reward:.3f}', episode)

            if episode > 0:
                writer.add_scalar(f'Win Rates -> Agent 1: {agent_1_score/episode:.4f}', episode)
                writer.add_scalar(f'             Agent 2: {agent_2_score/episode:.4f}', episode)
            if num_steps1 > 0:
                average_loss1 = total_loss1 / num_steps1
                writer.add_scalar('Agent 1/Ave_Loss', average_loss1, episode)
                writer.add_scalar('Agent 1/Scores', agent_1_score, episode)
            if agent_2_starts > 0:
                writer.add_scalar('Comp/Ratios/Agent_1_over_Agent_2_as_Player1', agent_1_starts / agent_2_starts, episode)
            writer.add_scalar('Comp/Differences/Agent_1_reward_minus_Agent_2_reward', agent_1_reward - agent_2_reward, episode)
            if agent_2_reward > 0:
                writer.add_scalar('Comp/Ratios/Agent_1_reward_divided_by_Agent_2_reward', agent_1_reward / agent_2_reward, episode)
            if episode > 0:
                writer.add_scalar('Comp/Win Rates/Agent_1', agent_1_score / episode, episode)
                writer.add_scalar('Comp/Win Rates/Agent_2', agent_2_score / episode, episode)

            if done and (episode % params["ckpt_interval"] == 0):   
                agent1_filename = f'agent1_{hyp_file_root}.wts'
                agent2_filename = f'agent2_{hyp_file_root}.wts'
                print(f'\nSaving model weights to {agent1_filename}')
                torch.save(agent1.state_dict(), agent1_filename)
                torch.save(agent2.state_dict(), agent2_filename)

                # redundent save_checkpoint(agent1, optimizer1, replay_buffer1, episode, f'{hyp_file_root}_1.ckpt')
                # we need to switch over to this method, but retain the separate model weight files for play.py
                save_checkpoint(agent1, optimizer1, replay_buffer, episode, f'{hyp_file_root}_1.ckpt')
                save_checkpoint(agent2, optimizer2, replay_buffer, episode, f'{hyp_file_root}_2.ckpt')


    writer.close()


    print(f'\nTraining is done! The agents have played {end_episode} episodes.')
    agent1_filename = f'agent1_{hyp_file_root}.wts'
    agent2_filename = f'agent2_{hyp_file_root}.wts'
    replay_buffer_filename = f'replay_buffer_{hyp_file_root}.pkl'
    torch.save(agent1.state_dict(), agent1_filename)
    torch.save(agent2.state_dict(), agent2_filename)
    with open(replay_buffer_filename, 'wb') as f:
        pickle.dump(replay_buffer, f)

 
    # Print the sizes of the saved models
    checkpoint_a1 = torch.load(agent1_filename)
    for key, tensor in checkpoint_a1.items():
        print(f"{agent1_filename}: {key}: {tensor.size()}")   
    checkpoint_a2 = torch.load(agent2_filename)
    for key, tensor in checkpoint_a2.items():
        print(f"{agent2_filename}: {key}: {tensor.size()}")    

                       


    print(f'models saved for both agents, agent1 to {agent1_filename}, agent2 to {agent2_filename}')
    print(f'replay buffer saved to {replay_buffer_filename}')
       

if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    #cProfile.run('main()', 'train_stats')
    #p = pstats.Stats('train_stats')
    #p.sort_stats('cumtime').print_stats()





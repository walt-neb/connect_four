


#filename:  train_c4.py

import curses
import datetime
import time

import os
import sys
import pickle
import numpy as np
#import ddqn_agent_cnn
from ddqn_agent_cnn import CNNDDQNAgent 
from two_player_env import TwoPlayerConnectFourEnv
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
from collections import namedtuple



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
    param_str = ""
    print("*** Training Parameters: ")
    for key, value in params.items():
        param_str += (f"\t\t{key} :\t{value}\n")
    return param_str


def save_checkpoint(model, optimizer, replay_buffer, episode, checkpoint_path):
    print(f"Type before action: {type(replay_buffer)}")
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
    assert isinstance(replay_buffer, deque), "Replay buffer is not a deque!"
    assert episode == buffer_state['episode'], "Episode mismatch between model and buffer state."

    return model, optimizer, replay_buffer, episode

def load_agents_and_buffer(cnn_a1, cnn_a2, fc_a1, fc_a2, agent1_wts=None, agent2_wts=None, rp_buffer_file=None):
    input_channels = 1   # Assuming a single-channel input for the Connect Four board
    input_height = 6     # Connect Four board height
    input_width = 7      # Connect Four board width
    output_dim = 7       # One output for each column

    print(f'A1 Convolutional layers: {cnn_a1}')
    print(f'A2 Convolutional layers: {cnn_a2}')
    print(f'A1 Fully connected layers: {fc_a1}')
    print(f'A2 Fully connected layers: {fc_a2}')

    # Initialize agents with the specified convolutional and fully connected layers
    agent1 = CNNDDQNAgent(input_channels, input_height, input_width, output_dim, cnn_a1, fc_a1)
    agent2 = CNNDDQNAgent(input_channels, input_height, input_width, output_dim, cnn_a2, fc_a2)

    replay_buffer = deque(maxlen=10000)  # Adjust size as needed

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
        with open(rp_buffer_file, 'rb') as f:
            replay_buffer = pickle.load(f)
            assert isinstance(replay_buffer, deque), "Replay buffer is not a deque!"

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

# Define the Transition namedtuple
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

def train(agent, agent_tgt, optimizer, replay_buffer, batch_size, gamma):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if len(replay_buffer) < batch_size:
        return torch.tensor(0.0, device=device)  # Not enough samples to train so return 0 loss

    # Ensure the replay buffer is a list or convert it temporarily for sampling
    if isinstance(replay_buffer, (set, dict)):
        replay_buffer = list(replay_buffer)  # Convert to list if not already

    # Sample a batch from the replay buffer
    transitions = replay_buffer.sample(batch_size)    
    #print("Transitions sample:", transitions)
    if transitions and isinstance(transitions[0], tuple):
        batch = [Transition(*t) for t in transitions]  # Convert list of tuples to list of Transition namedtuples
    else:
        batch = transitions  # Already Transition namedtuples
    #print("Batch sample:", batch[0].state, batch[0].action, batch[0].reward, batch[0].next_state, batch[0].done)

    # Unpack the transitions
    states, actions, rewards, next_states, dones = zip(*transitions)
    
    states = torch.tensor([t.state for t in batch], dtype=torch.float32).view(batch_size, 1, 6, 7).to(device)
    next_states = torch.tensor([t.next_state for t in batch], dtype=torch.float32).view(batch_size, 1, 6, 7).to(device)
    actions = torch.tensor([t.action for t in batch], dtype=torch.int64).to(device)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32).to(device)
    dones = torch.tensor([t.done for t in batch], dtype=torch.bool).to(device)


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
def str_to_bool(s):
    result = s.lower() == 'true'
    return result

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
        cnn_a1 = list(eval(params["cnn_a1"].strip('[]')))
        cnn_a2 = list(eval(params["cnn_a2"].strip('[]')))  
        fc_a1 = list(eval(params["fc_a1"].strip('[]')))
        fc_a2 = list(eval(params["fc_a2"].strip('[]')))                    
        render_games = eval(params["render_game_at"].strip('[]'))
    except Exception as e:
        print(e)
        sys.exit(1)


    # Check command line arguments
    if len(sys.argv) == 5:  # Expected: script name, agent1 weights, agent2 weights, buffer
        agent1, agent2, replay_buffer = load_agents_and_buffer(cnn_a1, cnn_a2, fc_a1, fc_a2, sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 4:  # Only agents, no buffer
        agent1, agent2, replay_buffer = load_agents_and_buffer(cnn_a1, cnn_a2, fc_a1, fc_a2, sys.argv[2], sys.argv[3])
    else:
        agent1, agent2, replay_buffer = load_agents_and_buffer(cnn_a1, cnn_a2, fc_a1, fc_a2)  
    
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
    optimizer1 = optim.Adam(agent1.parameters(), lr=params["agent1_learning_rate"])
    optimizer2 = optim.Adam(agent2.parameters(), lr=params["agent2_learning_rate"])


    # Environment
    env = TwoPlayerConnectFourEnv(writer=writer)

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
    steps_per_game = 0
    ave_steps_per_game = 10

    start_episode = params["start_episode"]
    env.enable_reward_shaping(str_to_bool(params["enable_reward_shaping"]))
    env.enable_debug_mode(str_to_bool(params["env_debug_mode"]))

    #agent1, optimizer1, replay_buffer, start_episode = load_checkpoint('path_to_checkpoint', agent1, optimizer1, device)

    for episode in range(start_episode, end_episode):
        state, active_agent = env.reset()

        #active_agent = 1 # Start with agent 1

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

        print(f'{hyp_file_root} ', end='')
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
                #assert isinstance(replay_buffer1, deque), "Replay buffer is not a deque!"

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
                #assert isinstance(replay_buffer2, deque), "Replay buffer is not a deque!"

                loss2 = train(agent2, agent2_tgt, optimizer2, replay_buffer2, batch_size, gamma)
                if loss2 is not None:
                    total_loss2 += loss2.item()
                    num_steps2 += 1

            if episode in render_games or steps_per_game > 40: 
                #logthis = True
                print(f"----Episode {episode} Step {num_steps1 + num_steps2}")                
                winner = env.render() 
                #print(f"Agent {active_agent} playing ({env.get_player_symbol(active_agent)}) takes action {action} receives reward of {reward:.2f}")   
                #input("Press key to continue...")

            if not done:
                state = next_state
                active_agent = next_player
            #print('.', end='')

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
            winner = env.render()
            print(f"Episode {episode} Step {num_steps1 + num_steps2}")
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

            if False and episode > 0:
                writer.add_scalar('Agent 1/Win Rate', agent_1_score/episode, episode)
                writer.add_scalar('Agent 2/Win Rate', agent_2_score/episode, episode)
                writer.add_scalar('Agent 2/Replay Buffer Size', len(replay_buffer2), episode)
                writer.add_scalar('Comp/A1 reward/episode', agent_1_reward / episode, episode)
                writer.add_scalar('Comp/A2 reward/episode', agent_2_reward / episode, episode)
                writer.add_scalar('Comp/A1/A2 rpe', (agent_1_reward/episode)/(agent_2_reward/episode), episode)


            writer.add_scalar('Comp/StepsPerGame', steps_per_game, episode)

            if False and agent_2_starts > 0:
                writer.add_scalar('Comp/Ratio Agent_1_over_Agent_2 as Player1', agent_1_starts / agent_2_starts, episode)

            writer.add_scalar('Comp/Draws', draw_score, episode)

            if agent_2_reward > 0:
                writer.add_scalar('Comp/Ratio Agent_1_to_Agent_2 Reward', agent_1_reward / agent_2_reward, episode)

            if done and (episode % params["ckpt_interval"] == 0):   
                agent1_filename = f'./wts/model1_{hyp_file_root}.wts'
                agent2_filename = f'./wts/model2_{hyp_file_root}.wts'
                replay_buffer_filename = f'./wts/replay_buffer_{hyp_file_root}.pkl'
                torch.save(agent1.state_dict(), agent1_filename)
                torch.save(agent2.state_dict(), agent2_filename)
                with open(replay_buffer_filename, 'wb') as f:
                    pickle.dump(replay_buffer, f)

                # redundent save_checkpoint(agent1, optimizer1, replay_buffer1, episode, f'{hyp_file_root}_1.ckpt')
                # we need to switch over to this method, but retain the separate model weight files for play.py
                #save_checkpoint(agent1, optimizer1, replay_buffer, episode, f'{hyp_file_root}_1.ckpt')
                #save_checkpoint(agent2, optimizer2, replay_buffer, episode, f'{hyp_file_root}_2.ckpt')

            if check_e_keypress():# or draw_score > 10:
                if draw_score > 10:
                    print('Draws > 10, exiting training loop')  
                else:
                    print('Keyboard Keypress -e- detected, exiting training loop')
                break

    writer.close()


    print(f'\nTraining ended on episode count: {episode}')
    agent1_filename = f'./wts/model1_{hyp_file_root}.wts'
    agent2_filename = f'./wts/model2_{hyp_file_root}.wts'
    replay_buffer_filename = f'./wts/replay_buffer_{hyp_file_root}.pkl'
    torch.save(agent1.state_dict(), agent1_filename)
    torch.save(agent2.state_dict(), agent2_filename)
    with open(replay_buffer_filename, 'wb') as f:
        pickle.dump(replay_buffer, f)
        assert isinstance(replay_buffer, deque), "Replay buffer is not a deque!"


 
    # Print the sizes of the saved models
    # checkpoint_a1 = torch.load(agent1_filename)
    # for key, tensor in checkpoint_a1.items():
    #     print(f"{agent1_filename}: {key}: {tensor.size()}")   
    # checkpoint_a2 = torch.load(agent2_filename)
    # for key, tensor in checkpoint_a2.items():
    #     print(f"{agent2_filename}: {key}: {tensor.size()}")    

                       
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
    test_results_string += f'models saved for both agents:\n{agent1_filename}\n{agent2_filename}\n'
    test_results_string += f'replay buffer saved to\n{replay_buffer_filename}\n'
    if episode != end_episode:
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
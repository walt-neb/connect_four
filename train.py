

#filename:  train.py

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


writer = SummaryWriter('runs/connect_four_experiment')

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

def load_agents_and_buffer(agent1_path=None, agent2_path=None, buffer_path=None, a1_layer_dims=[64, 64], a2_layer_dims=[64, 64]):
    # Initialize agents
    input_dim = 6 * 7  # Board size
    output_dim = 7     # One output for each column
    agent1 = DQNAgent(input_dim, output_dim, a1_layer_dims)
    agent2 = DQNAgent(input_dim, output_dim, a2_layer_dims)
    replay_buffer = deque(maxlen=10000)  # Adjust size as needed

    # Load weights if paths are provided
    if agent1_path and agent2_path:
        agent1.load_state_dict(torch.load(agent1_path))
        agent2.load_state_dict(torch.load(agent2_path))
        agent1.eval()
        agent2.eval()
        print("Loaded weights for both agents.")
    else:
        print("Starting agents from scratch.")

    # Load replay buffer if path is provided
    if buffer_path:
        with open(buffer_path, 'rb') as f:
            replay_buffer = pickle.load(f)
        print("Loaded replay buffer.")

    return agent1, agent2, replay_buffer


def train(agent, optimizer, replay_buffer, batch_size):
    if len(replay_buffer) < batch_size:
        return torch.tensor(0.0) # Not enough samples to train so return 0 loss

    transitions = replay_buffer.sample(batch_size)
    batch = list(zip(*transitions))

    states = torch.tensor(np.array(batch[0]), dtype=torch.float32)
    actions = torch.tensor(batch[1], dtype=torch.long)
    rewards = torch.tensor(batch[2], dtype=torch.float32)
    next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32)
    dones = torch.tensor(batch[4], dtype=torch.bool)

    # Current Q values
    curr_q_values = agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    # Next Q values
    next_q_values = agent(next_states).max(1)[0]
    next_q_values[dones] = 0  # No next state if the game is done

    # Compute the target Q values
    target_q_values = rewards + (0.99 * next_q_values)
    # Loss
    loss = nn.MSELoss()(curr_q_values, target_q_values)

    # Gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

dbmode=1

def main():

    hyp_file = 'hyps_1.txt'
    params = load_hyperparams(hyp_file)
    print_parameters(params)
    try:
        agent_1_layer_dims_string = params["agent_1_layer_dims"].strip('[]')
        agent_1_layer_dims_string = [int(dim) for dim in agent_1_layer_dims_string.split()]
        agent_2_layer_dims_string = params["agent_2_layer_dims"].strip('[]')
        agent_2_layer_dims_string = [int(dim) for dim in agent_2_layer_dims_string.split()]
    except Exception as e:
        print(e)
    print(f'agent_1_layer_dims_string: {agent_1_layer_dims_string}')
    print(f'agent_2_layer_dims_string: {agent_2_layer_dims_string}')

    # Check command line arguments
    if len(sys.argv) == 4:  # Expected: script name, agent1 weights, agent2 weights, buffer
        agent1, agent2, replay_buffer = load_agents_and_buffer(sys.argv[1], sys.argv[2], sys.argv[3], agent_1_layer_dims_string, agent_2_layer_dims_string)
    elif len(sys.argv) == 3:  # Only agents, no buffer
        agent1, agent2, replay_buffer = load_agents_and_buffer(sys.argv[1], sys.argv[2], agent_2_layer_dims_string)
    else:
        agent1, agent2, replay_buffer = load_agents_and_buffer()




    # Optimizers
    optimizer1 = optim.Adam(agent1.parameters(), lr=params["agent1_learning_rate"])
    optimizer2 = optim.Adam(agent2.parameters(), lr=params["agent1_learning_rate"])

    # Environment
    env = ConnectFour()

    # Main training loop
    num_episodes = params["end_episode"]
    batch_size = params["batch_size"]
    buffer_capacity = params["max_replay_buffer_size"]

    replay_buffer1 = ReplayBuffer(buffer_capacity)
    replay_buffer2 = ReplayBuffer(buffer_capacity)

    epsilon_step_count = 0

    agent_1_score = 0
    agent_2_score = 0
    draw_score = 0
    agent_1_starts = 0
    agent_2_starts = 0


    for episode in range(num_episodes):
        state, active_agent = env.reset()
        done = False
        total_loss1 = 0
        total_loss2 = 0
        num_steps1 = 0
        num_steps2 = 0
        epsilon_step_count += 1
        if active_agent == 1:
            agent_1_starts += 1
        else:
            agent_2_starts += 1
        if agent_2_starts > 0:
            writer.add_scalar(f'agent_1_starts/agent_2_starts={agent_1_starts/agent_2_starts:.3f}', episode)

        while not done:
            logthis = False
            valid_actions = env.get_valid_actions()
            epsilon1 = agent1.get_epsilon(epsilon_step_count, num_episodes, 1.5, 0.1)
            epsilon2 = agent2.get_epsilon(epsilon_step_count, num_episodes, 1.2, 0.1)
            # Determine which agent is playing
            if active_agent == 1:
                action = agent1.select_action(state, valid_actions, epsilon1)
                next_state, reward, done, next_player = env.step(action)
                replay_buffer1.push(state, action, reward, next_state, done)
                loss1 = train(agent1, optimizer1, replay_buffer1, batch_size)
                if loss1 is not None:
                    total_loss1 += loss1.item()
                    num_steps1 += 1
            else:
                action = agent2.select_action(state, valid_actions, epsilon2)
                next_state, reward, done, next_player = env.step(action)
                replay_buffer2.push(state, action, reward, next_state, done)
                loss2 = train(agent2, optimizer2, replay_buffer2, batch_size)
                if loss2 is not None:
                    total_loss2 += loss2.item()
                    num_steps2 += 1

            if episode in params["render_episodes"]:
                logthis = True
                print(f"Episode {episode}, Step {num_steps1 + num_steps2}")
                print(f"Agent {active_agent} ({env.get_player_symbol(active_agent)}) action: {action}")
                winner = env.render()    
                if winner is not None:
                    print(f"Winner: Agent {winner}")            
                print(f"----------------------")

            if not done:
                state = next_state
                active_agent = next_player

        # track the scores
        if env.winner == 1:
            agent_1_score += 1
        elif env.winner == 2:
            agent_2_score += 1
        elif env.winner == 0:
            draw_score += 1

        if done and (episode % 100 == 0 or logthis==True):  # Render end of the game for specified episodes
            print(f"Episode {episode}, Step {num_steps1 + num_steps2}")
            print(f"Agent {active_agent} ({env.get_player_symbol(active_agent)}) action: {action}")
            winner = env.render()
            if winner is not None:
                print(f"Winner: Agent {winner}")  
            print(f'Episode {episode} of {num_episodes}')
            print(f'Agent 1: {agent_1_score}')
            print(f'Agent 2: {agent_2_score}, Draws: {draw_score}')
            print(f'Agent 1 epsilon: {epsilon1}, Agent 2 epsilon: {epsilon2}')
            if num_steps1 > 0 and num_steps2 > 0:
                print(f'Agent 1 loss: {total_loss1 / num_steps1}, Agent 2 loss: {total_loss2 / num_steps2}')
            print(f'Agent 1 replay buffer size: {len(replay_buffer1)}, Agent 2 replay buffer size: {len(replay_buffer2)}')

            print(f"Episode {episode}: Agent 1 starts {agent_1_starts}, Agent 2 starts {agent_2_starts}")
            print(f"Current Epsilons -> Agent 1: {epsilon1}, Agent 2: {epsilon2}")
            if episode > 0:
                print(f"Win Rates -> Agent 1: {agent_1_score/episode:.4f}, Agent 2: {agent_2_score/episode:.4f}")

            # log the average loss to TensorBoard
            if num_steps1 > 0:
                average_loss1 = total_loss1 / num_steps1
                writer.add_scalar('Loss/Average1', average_loss1, episode)
                writer.add_scalar('Scores/Agent 1', agent_1_score, episode)
                print(f"Episode {episode}: Average Loss1 = {average_loss1:.4f}")
            if num_steps2 > 0:
                average_loss2 = total_loss2 / num_steps2
                writer.add_scalar('Loss/Average2', average_loss2, episode)
                writer.add_scalar('Scores/Agent 2', agent_2_score, episode)
                print(f"Episode {episode}: Average Loss2 = {average_loss2:.4f}")      

            writer.add_scalar('ReplayBuffer/Size_P1', len(replay_buffer1), episode)
            writer.add_scalar('ReplayBuffer/Size_P2', len(replay_buffer2), episode)
            #writer.add_scalar('Loss/AverageLoss1', average_loss1, episode)
            #writer.add_scalar('Loss/AverageLoss2', average_loss2, episode)
            writer.add_scalar('Epsilon/Agent 1', epsilon1, episode)
            writer.add_scalar('Epsilon/Agent 2', epsilon2, episode)                          

        if done and episode % 1000 == 0:
            print(f'saving models and replay buffer at episode {episode}')
            torch.save(agent1.state_dict(), 'agent1_weights.pth')
            torch.save(agent2.state_dict(), 'agent2_weights.pth')
            with open('replay_buffer.pkl', 'wb') as f:
                pickle.dump(replay_buffer, f)


    writer.close()


    print(f'Training is done! The agents have played {num_episodes} episodes.')
    torch.save(agent1.state_dict(), 'agent1_weights.pth')
    torch.save(agent2.state_dict(), 'agent2_weights.pth')
    with open('replay_buffer.pkl', 'wb') as f:
        pickle.dump(replay_buffer, f)

    print(f'models saved for both agents, to agent1_weights.pth and agent2_weights.pth')
    print(f'replay buffer saved to replay_buffer.pkl')
       


if __name__ == '__main__':
    main()    




'''
def train_self_play(agent, env, num_episodes):
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Store transition in the replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            # Move to the next state
            state = next_state

            # Training step
            if len(replay_buffer) > batch_size:
                transitions = replay_buffer.sample(batch_size)
                loss = compute_loss(agent, transitions)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode}: Loss = {loss.item()}")

    # Save the trained model
    torch.save(agent.state_dict(), 'trained_model.pth')

'''
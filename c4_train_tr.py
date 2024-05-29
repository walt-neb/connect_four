import torch.optim as optim
import numpy as np
import curses
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

def print_parameters(params):
    if not params:
        print("The parameters dictionary is empty.")
        return
    param_str = ""
    print("*** Training Parameters: ")
    for key, value in params.items():
        param_str += (f"\t\t{key} :\t{value}\n")
    return param_str

def main():

    # Action selection with epsilon-greedy policy
    def select_action(state, epsilon):
        if np.random.rand() <= epsilon:
            # Explore: select a random action
            valid_actions = env.get_valid_actions()
            action = np.random.choice(valid_actions)
        else:
            # Exploit: select the action with the highest Q-value
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = agent(state_tensor)
            valid_actions = env.get_valid_actions()
            mask = torch.ones(output_dim) * float('-inf')
            mask[valid_actions] = 0
            masked_q_values = q_values + mask
            action = torch.argmax(masked_q_values).item()
        return action

    start_time = datetime.datetime.now()
    print(f'torch.cuda.is_available()={torch.cuda.is_available()}')

    if len(sys.argv) < 2:
        print("Usage: python train_c4.py <hyperparameters_file>")
        return
    
    hyp_file = sys.argv[1]
    hyp_file_root = hyp_file.rstrip('.hyp')
    hyp_file_root = os.path.basename(hyp_file_root)
    print(f'hyp_file_root: {hyp_file_root}')

    writer = SummaryWriter(f'runs/{hyp_file_root}_tr')

    par = load_hyperparams(hyp_file)
    print(print_parameters(par))

    input_dim = par['input_dim']
    embed_dim = par['embed_dim']
    n_heads = par['n_heads']
    ff_dim = par['ff_dim']
    n_layers = par['n_layers']
    output_dim = par['output_dim']
    dropout = par['dropout']
    capacity = par['capacity']
    batch_size = par['batch_size']
    learning_rate = par['learning_rate']
    gamma = par['gamma']
    epsilon = par['epsilon']
    epsilon_decay = par['epsilon_decay']
    epsilon_min = par['epsilon_min']
    num_epochs = par['num_epochs']


    # Initialize components
    env = ConnectFourEnv()
    agent = TransformerAgent(input_dim, embed_dim, n_heads, ff_dim, n_layers, output_dim, dropout)
    if len(sys.argv) == 3:
        model_file = f'./wts/{hyp_file_root}.pth'
        print(f'Continuing training on model {model_file}')
        agent.load_state_dict(torch.load(model_file))
    else:
        print('No model file provided, initializing new model')
    replay_buffer = ReplayBuffer(capacity)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    loss = torch.tensor(0.0)


    # Training loop with epsilon decay
    for epoch in range(num_epochs):
        state, player = env.reset()
        done = False
        loss = torch.tensor(0.0)  # Initialize loss to zero or another default value

        while not done:
            action = select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state.flatten(), done)

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

                state_batch = torch.FloatTensor(state_batch)
                action_batch = torch.LongTensor(action_batch)
                reward_batch = torch.FloatTensor(reward_batch)
                next_state_batch = torch.FloatTensor(next_state_batch)
                done_batch = torch.FloatTensor(done_batch)

                q_values = agent(state_batch)
                next_q_values = agent(next_state_batch).detach()
                max_next_q_values = torch.max(next_q_values, dim=1)[0]
                target_q_values = reward_batch + gamma * max_next_q_values.unsqueeze(1) * (1 - done_batch)
                q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
                loss = criterion(q_value, target_q_values.squeeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state.flatten()
        
        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Log metrics to TensorBoard
        writer.add_scalar('Loss', loss.item(), epoch)
        writer.add_scalar('Epsilon', epsilon, epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Epsilon: {epsilon}")
        if check_e_keypress():
            print('Keyboard Keypress -e- detected, exiting training loop')
            #save_all_models(agent2, optimizer2, "a2", replay_buffer2, f"./wts/chkpt_{hyp_file_root}_{episode}")
            break
    writer.close()

    print(f'\nTraining ended on epoch count: {epoch}')
    # Save the model's state dictionary
    model_file = f'./wts/{hyp_file_root}.pth'
    torch.save(agent.state_dict(), model_file)
                       
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time                       

    test_results_string = '\n\n----------- Training Results ----------------\n' 
    test_results_string += f'Started training at: \t{start_time.strftime("%Y-%m-%d  %H:%M:%S")}\n'
    test_results_string += f'Ended training at: \t{end_time.strftime("%Y-%m-%d  %H:%M:%S")}\n'
    test_results_string += f'Total training time:  {str(elapsed_time)}\n'
    test_results_string += f'epoch count: {epoch}\n'
    test_results_string += f'end epsilon: {epsilon}\n'

    test_results_string += f'Input Parameters:\n'
    test_results_string += print_parameters(par)


    if epoch != num_epochs-1:
        test_results_string += f'\n**Exited training loop early at episode {epoch}'
        test_results_string += f'\nstart_episode = {epoch}\n'

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




if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
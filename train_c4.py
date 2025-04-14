# filename: train_c4.py
"""
Training Script for Connect Four using Double DQN (DDQN) with CNN.

Saves comprehensive checkpoints (episode, models, optimizers) for resuming.
Saves replay buffer separately.
Removes redundant saving of individual model weights during/after training.
Use 'disassemble_checkpoint.py' to extract individual weights for playing.
"""
import datetime
import time
import ast
import os
import sys
import pickle
import numpy as np
import math
import argparse
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

# --- Local Imports ---
from ddqn_agent_cnn import CNNDDQNAgent
from two_player_env import TwoPlayerConnectFourEnv
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

# --- Utility Functions (load_hyperparams, print_parameters, normalize_state, train_step) ---
# (These remain the same)
def load_hyperparams(hyp_file):
    params = {}
    print(f"Loading hyperparameters from: {hyp_file}")
    with open(hyp_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line: continue
            if "=" in line:
                var_name, var_value = line.split("=", 1); var_name = var_name.strip(); var_value = var_value.strip()
                try: params[var_name] = ast.literal_eval(var_value)
                except: params[var_name] = var_value
    print("Hyperparameters loaded."); return params

def print_parameters(params):
    if not params: return "No parameters found."
    param_str = "*** Training Parameters: ***\n";
    for key, value in params.items(): param_str += (f"\t{key:<25} : {value}\n")
    return param_str

def normalize_state(board_2d, current_player):
    normalized_board = np.zeros_like(board_2d, dtype=np.float32); opponent_player = 3 - current_player
    normalized_board[board_2d == current_player] = 1.0; normalized_board[board_2d == opponent_player] = -1.0
    return normalized_board.flatten()

def train_step(agent_policy, agent_target, optimizer, replay_buffer, batch_size, gamma, device):
    if len(replay_buffer) < batch_size: return torch.tensor(0.0, device=device)
    transitions = replay_buffer.sample(batch_size); states, actions, rewards, next_states, dones = zip(*transitions)
    states_tensor = torch.tensor(np.array(states), dtype=torch.float32).view(batch_size, 1, 6, 7).to(device)
    next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).view(batch_size, 1, 6, 7).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(-1).to(device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)
    dones_tensor = torch.tensor(dones, dtype=torch.bool).unsqueeze(-1).to(device)
    with torch.no_grad():
        next_state_actions = agent_policy(next_states_tensor).argmax(dim=1, keepdim=True)
        next_state_q_values = agent_target(next_states_tensor).gather(1, next_state_actions)
        next_state_q_values[dones_tensor] = 0.0; target_q_values = rewards_tensor + (gamma * next_state_q_values)
        # Clip targets to a reasonable range, e.g. based on potential discounted rewards
        # Max possible reward is +1 per step. Max discounted return roughly 1/(1-gamma)
        # Min possible reward is 0 (or -1 if using -1 for loss).
        # Let's clip somewhat generously, e.g., -20 to +20
        target_clip_min = -20.0
        target_clip_max = 20.0
        target_q_values = torch.clamp(target_q_values, target_clip_min, target_clip_max)
    
    current_q_values = agent_policy(states_tensor).gather(1, actions_tensor)
    loss = nn.SmoothL1Loss()(current_q_values, target_q_values) # Huber loss
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(agent_policy.parameters(), max_norm=1.0); optimizer.step()
    return loss

# --- Main Execution Block ---
def main():
    # --- Argument Parser Setup (Same as before) ---
    parser = argparse.ArgumentParser(description='Train Connect Four DDQN Agents.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('hyp_file', type=str, help='Path to hyperparameter file')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint file (.pth) to resume training.')
    parser.add_argument('--load_agent1_weights', type=str, default=None, help='Path to initial weights for agent 1 (only if not resuming).')
    parser.add_argument('--load_agent2_weights', type=str, default=None, help='Path to initial weights for agent 2 (only if not resuming).')
    parser.add_argument('--load_buffer', type=str, default=None, help='Path to replay buffer (.pkl) to load initially or with checkpoint if separate.')
    if len(sys.argv) == 1: parser.print_help(sys.stderr); sys.exit(1)
    args = parser.parse_args()

    # --- Initial Setup (Same as before) ---
    start_time_dt = datetime.datetime.now(); print(f"Starting script at: {start_time_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f'Using PyTorch device: {device}')

    # --- Load Hyperparameters (Same as before) ---
    hyp_file = args.hyp_file
    if not os.path.exists(hyp_file): print(f"Error: Hyp file not found: {hyp_file}"); sys.exit(1)
    params = load_hyperparams(hyp_file); hyp_file_root = os.path.basename(hyp_file).replace('.hyp', '')
    print(print_parameters(params))

    # Extract key parameters (Same as before)
    gamma_lr = params['gamma_lr']; scheduler_step_size = params['scheduler_step_size']; target_clip_min = params['target_clip_min']; target_clip_max = params['target_clip_max']
    end_episode = params['end_episode']; short_log_interval = params.get('short_log_interval', 50); console_status_interval = params.get('console_status_interval', 1000)
    tensorboard_status_interval = params.get('tensorboard_status_interval', 100); ckpt_interval = params.get('ckpt_interval', 2000); render_game_at = params.get('render_game_at', [])
    target_update_freq = params.get('target_update_frequency', 250); agent1_lr = params['agent1_learning_rate']; agent2_lr = params['agent2_learning_rate']
    a1_epsilon_start = params.get('a1_epsilon_start', 1.0); a1_epsilon_end = params.get('a1_epsilon_end', 0.01); a2_epsilon_start = params.get('a2_epsilon_start', 1.0); a2_epsilon_end = params.get('a2_epsilon_end', 0.01)
    batch_size = params['batch_size']; gamma = params['gamma']; buffer_capacity = params['max_replay_buffer_size']
    cnn_a1_params = params['cnn_a1']; cnn_a2_params = params['cnn_a2']; fc_a1_dims = params['fc_a1']; fc_a2_dims = params['fc_a2']

    # --- Setup TensorBoard & Log Dir (Same as before) ---
    log_dir = f'runs/{hyp_file_root}_{start_time_dt.strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir); print(f"TensorBoard logs: {log_dir}")
    writer.add_text('Hyperparameters', print_parameters(params).replace('\t','&nbsp;&nbsp;&nbsp;&nbsp;').replace('\n','<br/>'))
    try: shutil.copy2(hyp_file, os.path.join(log_dir, os.path.basename(hyp_file))); print(f"Copied hyperparameter file to log directory.")
    except Exception as e: print(f"Warning: Could not copy hyperparameter file: {e}")

    # --- Initialize Agents, Optimizers, Env, Buffer (Same as before) ---
    input_channels = 1; input_height = 6; input_width = 7; output_dim = 7
    print("\n--- Building Agent 1 POLICY Network ---")
    agent1 = CNNDDQNAgent(input_channels, input_height, input_width, output_dim, cnn_a1_params, fc_a1_dims).to(device)
    print("\n--- Building Agent 2 POLICY Network ---")
    agent2 = CNNDDQNAgent(input_channels, input_height, input_width, output_dim, cnn_a2_params, fc_a2_dims).to(device)
    print("\n--- Building Agent 1 TARGET Network ---")
    agent1_tgt = CNNDDQNAgent(input_channels, input_height, input_width, output_dim, cnn_a1_params, fc_a1_dims).to(device); agent1_tgt.eval()
    print("\n--- Building Agent 2 TARGET Network ---")
    agent2_tgt = CNNDDQNAgent(input_channels, input_height, input_width, output_dim, cnn_a2_params, fc_a2_dims).to(device); agent2_tgt.eval()
    print("\n--- Building Optimizers ---")
    optimizer1 = optim.Adam(agent1.parameters(), lr=agent1_lr); optimizer2 = optim.Adam(agent2.parameters(), lr=agent2_lr)
    env = TwoPlayerConnectFourEnv(writer=writer); replay_buffer = ReplayBuffer(buffer_capacity); start_episode = 0
    # Decay LR by factor gamma_lr every scheduler_step_size episodes
    gamma_lr = 0.99 # Decay factor (e.g., 1% decay) - tune this
    scheduler_step_size = 1000 # How often to decay (e.g., every 1000 episodes) - tune this

    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=scheduler_step_size, gamma=gamma_lr)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=scheduler_step_size, gamma=gamma_lr)

    # --- Loading Logic (Same as before - handles resume OR initial weights/buffer) ---
    if args.resume_from_checkpoint:
        if os.path.exists(args.resume_from_checkpoint):
            print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
            try:
                checkpoint = torch.load(args.resume_from_checkpoint, map_location=device, weights_only=False)
                start_episode = checkpoint.get('episode', 0)
                agent1.load_state_dict(checkpoint['agent1_state_dict']); agent2.load_state_dict(checkpoint['agent2_state_dict'])
                optimizer1.load_state_dict(checkpoint['optimizer1_state_dict']); optimizer2.load_state_dict(checkpoint['optimizer2_state_dict'])
                if 'replay_buffer_deque' in checkpoint:
                     replay_buffer.buffer = checkpoint['replay_buffer_deque']; print(f"Loaded replay buffer ({len(replay_buffer)}) from checkpoint.")
                elif args.load_buffer and os.path.exists(args.load_buffer):
                     print(f"Trying separate buffer: {args.load_buffer}")
                     with open(args.load_buffer, 'rb') as f: replay_buffer.buffer = pickle.load(f)
                     print(f"Loaded replay buffer ({len(replay_buffer)}) from {args.load_buffer}.")
                else: print("No replay buffer found/specified for resume.")
                print(f"Resuming from episode {start_episode}")
            except Exception as e: print(f"ERROR loading checkpoint: {e}. Starting scratch."); start_episode = 0
        else: print(f"Warn: Checkpoint file not found: {args.resume_from_checkpoint}. Starting scratch."); start_episode = params.get('start_episode', 0)
    else:
        print("No checkpoint specified. Starting new run or loading initial components.")
        start_episode = params.get('start_episode', 0)
        if args.load_agent1_weights and os.path.exists(args.load_agent1_weights):
            try: agent1.load_state_dict(torch.load(args.load_agent1_weights, map_location=device, weights_only=True)); print(f"Loaded initial W1: {args.load_agent1_weights}")
            except Exception as e: print(f"Warn: Load initial W1 failed: {e}")
        if args.load_agent2_weights and os.path.exists(args.load_agent2_weights):
            try: agent2.load_state_dict(torch.load(args.load_agent2_weights, map_location=device, weights_only=True)); print(f"Loaded initial W2: {args.load_agent2_weights}")
            except Exception as e: print(f"Warn: Load initial W2 failed: {e}")
        if args.load_buffer and os.path.exists(args.load_buffer):
             try:
                 with open(args.load_buffer, 'rb') as f: replay_buffer.buffer = pickle.load(f)
                 print(f"Loaded initial buffer ({len(replay_buffer)}) from: {args.load_buffer}")
             except Exception as e: print(f"Error loading initial buffer: {e}")

    # Sync targets after loading
    agent1_tgt.load_state_dict(agent1.state_dict()); agent2_tgt.load_state_dict(agent2.state_dict())

    # --- Training Loop ---
    print(f"\n--- Starting Training Loop from episode {start_episode} to {end_episode} ---")
    total_games_played = 0; agent_1_wins = 0; agent_2_wins = 0; draws = 0
    cumulative_agent_1_reward = 0.0; cumulative_agent_2_reward = 0.0
    total_steps_all_episodes = 0; rolling_avg_steps = 0.0
    # Could potentially load win counts from checkpoint if saved there

    for episode in range(start_episode, end_episode):
        # (Inner loop logic remains the same: normalize, select, step, push, train)
        state_2d, active_player_id = env.reset(); done = False
        episode_steps = 0; episode_loss1 = 0.0; episode_loss2 = 0.0
        num_train_steps1 = 0; num_train_steps2 = 0
        while not done:
            normalized_state_flat = normalize_state(state_2d, active_player_id); valid_actions = env.get_valid_actions()
            if active_player_id == 1: epsilon = agent1.get_epsilon(episode, end_episode, a1_epsilon_start, a1_epsilon_end); action = agent1.select_action(normalized_state_flat, valid_actions, epsilon)
            else: epsilon = agent2.get_epsilon(episode, end_episode, a2_epsilon_start, a2_epsilon_end); action = agent2.select_action(normalized_state_flat, valid_actions, epsilon)
            next_state_2d, reward, done, next_player_id = env.step(action); normalized_next_state_flat = normalize_state(next_state_2d, active_player_id)
            replay_buffer.push(normalized_state_flat, action, reward, normalized_next_state_flat, done)
            if active_player_id == 1: cumulative_agent_1_reward += reward
            else: cumulative_agent_2_reward += reward
            loss1 = train_step(agent1, agent1_tgt, optimizer1, replay_buffer, batch_size, gamma, device); loss2 = train_step(agent2, agent2_tgt, optimizer2, replay_buffer, batch_size, gamma, device)
            if loss1 > 0: episode_loss1 += loss1.item(); num_train_steps1 += 1
            if loss2 > 0: episode_loss2 += loss2.item(); num_train_steps2 += 1
            if episode in render_game_at: print(f"\n--- Ep {episode} Step {episode_steps+1} P{active_player_id} Act:{action} Rew:{reward:.1f} Done:{done} ---"); env.render()
            state_2d = next_state_2d; active_player_id = next_player_id
            episode_steps += 1; total_steps_all_episodes += 1
            scheduler1.step()
            scheduler2.step()
            # Optional: Log learning rate
            if episode % tensorboard_status_interval == 0:
                writer.add_scalar('Progress/Learning_Rate', optimizer1.param_groups[0]['lr'], episode)

        # --- End of Episode ---
        # (Stats accumulation and logging remain the same)
        total_games_played += 1; current_run_games = episode - start_episode + 1
        if env.winner == 1: agent_1_wins += 1
        elif env.winner == 2: agent_2_wins += 1
        elif env.done: draws += 1
        if rolling_avg_steps == 0 and episode_steps > 0: rolling_avg_steps = episode_steps
        elif episode_steps > 0 : rolling_avg_steps = 0.99 * rolling_avg_steps + 0.01 * episode_steps
        avg_loss1 = episode_loss1 / num_train_steps1 if num_train_steps1 > 0 else 0; avg_loss2 = episode_loss2 / num_train_steps2 if num_train_steps2 > 0 else 0
        a1_win_rate = agent_1_wins / total_games_played if total_games_played > 0 else 0; a2_win_rate = agent_2_wins / total_games_played if total_games_played > 0 else 0
        draw_rate = draws / total_games_played if total_games_played > 0 else 0
        current_epsilon = agent1.get_epsilon(episode, end_episode, a1_epsilon_start, a1_epsilon_end)
        if episode % short_log_interval == 0 or episode == end_episode - 1:
             winner_str = f"A1 Win" if env.winner == 1 else (f"A2 Win" if env.winner == 2 else "Draw")
             print(f"Ep {episode:<5}/{end_episode} | {winner_str:<7} | Steps: {episode_steps:<3} (Avg: {rolling_avg_steps:.1f}) | Eps: {current_epsilon:.3f} | Buf: {len(replay_buffer):<7} | Loss A1/A2: {avg_loss1:.4f}/{avg_loss2:.4f}")
        if episode % console_status_interval == 0 or episode == end_episode - 1:
            print(f"--- Episode {episode} Detailed Stats ({total_games_played} total games tracked) ---"); print(f"  Scores -> A1 Wins: {agent_1_wins} ({a1_win_rate:.2%}) | A2 Wins: {agent_2_wins} ({a2_win_rate:.2%}) | Draws: {draws} ({draw_rate:.2%})")
            if episode not in render_game_at and env.winner is not None: env.render()
            print("-" * (30 + len(str(episode))))
        if episode % tensorboard_status_interval == 0 or episode == end_episode - 1:
            writer.add_scalar('Progress/Episode', episode, episode); writer.add_scalar('Progress/Epsilon', current_epsilon, episode)
            writer.add_scalar('Performance/Steps_Per_Game', episode_steps, episode); writer.add_scalar('Performance/Avg_Steps_Per_Game_EMA', rolling_avg_steps, episode)
            writer.add_scalar('Loss/Agent1_Avg_Loss', avg_loss1, episode); writer.add_scalar('Loss/Agent2_Avg_Loss', avg_loss2, episode)
            writer.add_scalar('Wins/Agent1_Total_Win_Rate', a1_win_rate, episode); writer.add_scalar('Wins/Agent2_Total_Win_Rate', a2_win_rate, episode)
            writer.add_scalar('Wins/Total_Draw_Rate', draw_rate, episode); writer.add_scalar('Buffer/Size', len(replay_buffer), episode)

        # Target Network Update
        if episode > 0 and episode % target_update_freq == 0:
             agent1_tgt.load_state_dict(agent1.state_dict()); agent2_tgt.load_state_dict(agent2.state_dict())

        # --- Checkpoint Saving (MODIFIED: Only combined checkpoint + separate buffer) ---
        if episode > 0 and episode % ckpt_interval == 0:
            # Combined Checkpoint for Resuming (includes model weights)
            ckpt_path = f'./wts/checkpoint_{hyp_file_root}_ep{episode}.pth'
            checkpoint = {
                'episode': episode + 1, # Save next episode to start from
                'agent1_state_dict': agent1.state_dict(),
                'agent2_state_dict': agent2.state_dict(),
                'optimizer1_state_dict': optimizer1.state_dict(),
                'optimizer2_state_dict': optimizer2.state_dict(),
                # Note: Buffer is NOT included here by default
            }
            try: torch.save(checkpoint, ckpt_path); print(f"--- Checkpoint saved: {ckpt_path} ---")
            except Exception as e: print(f"ERROR saving checkpoint: {e}")

            # Separate Buffer Saving
            replay_buffer_filename = f'./wts/replay_buffer_{hyp_file_root}_ep{episode}.pkl'
            try:
                with open(replay_buffer_filename, 'wb') as f: pickle.dump(replay_buffer.buffer, f)
                print(f"--- Replay buffer saved separately: {replay_buffer_filename} ---")
            except Exception as e: print(f"Warning: Could not save replay buffer separately: {e}")

            # *** REMOVED saving of separate m1_*.pth and m2_*.pth here ***

    # --- End of Training Loop ---
    writer.close(); print("\n--- Training Finished ---")

    # --- Final Saving (Only save LAST checkpoint and buffer) ---
    # Saving final individual models is now redundant, the last checkpoint has them.
    # We can save a final checkpoint for clarity.
    final_ckpt_path = f'./wts/checkpoint_{hyp_file_root}_final_ep{episode}.pth'
    final_checkpoint = {
         'episode': episode + 1, # Final episode + 1
         'agent1_state_dict': agent1.state_dict(),
         'agent2_state_dict': agent2.state_dict(),
         'optimizer1_state_dict': optimizer1.state_dict(),
         'optimizer2_state_dict': optimizer2.state_dict(),
    }
    try: torch.save(final_checkpoint, final_ckpt_path); print(f"Final checkpoint saved: {final_ckpt_path}")
    except Exception as e: print(f"Error saving final checkpoint: {e}")

    final_replay_buffer_filename = f'./wts/replay_buffer_{hyp_file_root}_final_ep{episode}.pkl'
    try:
        with open(final_replay_buffer_filename, 'wb') as f: pickle.dump(replay_buffer.buffer, f)
        print(f"Final buffer saved: {final_replay_buffer_filename}")
    except Exception as e: print(f"Warning: Could not save final replay buffer: {e}")

    # --- Final Summary Print & Save to logdir (Same as before) ---
    end_time_dt = datetime.datetime.now(); elapsed_time = end_time_dt - start_time_dt
    final_episode = episode; current_epsilon = agent1.get_epsilon(final_episode, end_episode, a1_epsilon_start, a1_epsilon_end)
    results_string = f'Training Started: \t{start_time_dt.strftime("%Y-%m-%d %H:%M:%S")}\n'; results_string += f'Training Ended: \t{end_time_dt.strftime("%Y-%m-%d %H:%M:%S")}\n'; results_string += f'Total Duration: \t{str(elapsed_time)}\n'; results_string += f'Episodes Run: \t\t{final_episode + 1 - start_episode} (End Ep: {final_episode}, Range: {start_episode}-{final_episode})\n'; results_string += f'Total Steps: \t\t{total_steps_all_episodes}\n'; results_string += f'Final Epsilon: \t\t{current_epsilon:.5f}\n'; results_string += f'Final Buffer Size: \t{len(replay_buffer)}\n'; results_string += f'Total Games Tracked: \t{total_games_played}\n'; results_string += f'Agent 1 Wins: \t\t{agent_1_wins} ({a1_win_rate:.2%})\n'; results_string += f'Agent 2 Wins: \t\t{agent_2_wins} ({a2_win_rate:.2%})\n'; results_string += f'Draws: \t\t\t{draws} ({draw_rate:.2%})\n'; results_string += f'Final Avg Steps/Game: {rolling_avg_steps:.2f}\n'; results_string += f'Input Parameters:\n{print_parameters(params)}'; results_string += f'Final Checkpoint:\n  {final_ckpt_path}\n';
    print("\n--- Final Training Summary ---"); print(results_string)
    try:
        summary_path = os.path.join(log_dir, 'final_summary.txt');
        with open(summary_path, 'w') as f: f.write(results_string)
        print(f"Final summary saved to: {summary_path}")
    except Exception as e: print(f"Warning: Could not save final summary file: {e}")


if __name__ == '__main__':
    main()
    print("\nScript finished.")
    print(f"End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Use 'tensorboard --logdir=runs' to view TensorBoard logs.")
    print("Use 'python play_human.py <agent_weights_path>' to play against the trained agent.")
    print("Use 'python disassemble_checkpoint.py <checkpoint_path>' to extract individual weights.")
    print("Use 'python play_two_models.py <agent1_path> <agent2_path>' to play two agents against each other.")
    print("Use 'python train_c4.py <hyp_file>' to train a new model.")
    print("Use 'python train_c4.py --resume_from_checkpoint <checkpoint_path>' to resume training.")
    print("Use 'python train_c4.py --load_buffer <buffer_path>' to load a replay buffer.")
    print("Use 'python train_c4.py --load_agent1_weights <weights_path>' to load initial weights for agent 1.")
    print("Use 'python train_c4.py --load_agent2_weights <weights_path>' to load initial weights for agent 2.")
    print("Use 'python train_c4.py --help' for more options.")
    print("End of script.")


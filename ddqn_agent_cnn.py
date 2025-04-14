# filename: ddqn_agent_cnn.py
"""
Double Deep Q-Network (DDQN) Agent with Convolutional Neural Network (CNN).

This module defines a reinforcement learning agent that uses a CNN followed by
fully connected layers to approximate the Q-value function. It implements the
DDQN algorithm logic for action selection (epsilon-greedy with masking) and
includes methods for calculating epsilon decay.
"""

import math
import random
import torch
import torch.nn as nn
import numpy as np

class CNNDDQNAgent(nn.Module):
    """
    A DDQN Agent using CNN for feature extraction from the board state.
    """
    def __init__(self, input_channels, input_height, input_width, output_dim, conv_layers_params, fc_layers_dims):
        """
        Initializes the CNN DDQN Agent.

        Args:
            input_channels (int): Number of channels in the input state (e.g., 1 for basic board).
            input_height (int): Height of the input board state.
            input_width (int): Width of the input board state.
            output_dim (int): Number of possible actions (Q-values to output, e.g., 7 for Connect Four columns).
            conv_layers_params (list): List of tuples defining CNN layers:
                                      [(out_channels, kernel_size, stride, padding), ...].
            fc_layers_dims (list): List of integers defining the output size of each hidden
                                   fully connected layer (excluding the final output layer).
                                   Example: [512, 256] -> Linear(?, 512), ReLU, Linear(512, 256), ReLU, Linear(256, output_dim).
        """
        super(CNNDDQNAgent, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Build Convolutional Layers ---
        current_channels = input_channels
        current_height = input_height
        current_width = input_width
        cnn_modules = []

        print("Building CNN layers:")
        for i, params in enumerate(conv_layers_params):
            out_channels, kernel_size, stride, padding = params
            conv = nn.Conv2d(current_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            cnn_modules.append(conv)
            cnn_modules.append(nn.ReLU())
            print(f"  Layer {i+1}: Conv2d({current_channels}, {out_channels}, kernel={kernel_size}, stride={stride}, padding={padding}), ReLU")

            # Calculate output dimensions after this conv layer
            # Formula: floor(((Input + 2*Padding - Kernel) / Stride) + 1)
            current_height = math.floor(((current_height + 2 * padding - kernel_size) / stride) + 1)
            current_width = math.floor(((current_width + 2 * padding - kernel_size) / stride) + 1)
            current_channels = out_channels

            if current_height <= 0 or current_width <= 0:
                raise ValueError(f"CNN layer {i+1} configuration results in non-positive dimension ({current_height}x{current_width}). Check parameters.")

        self.conv = nn.Sequential(*cnn_modules)
        self.num_flattened_features = current_channels * current_height * current_width
        print(f"CNN output flattened size: {self.num_flattened_features}")

        # --- Build Fully Connected Layers ---
        print("Building FC layers:")
        # Start with the flattened feature size from CNN
        fc_input_size = self.num_flattened_features
        fc_modules = []

        # Add hidden FC layers defined in fc_layers_dims
        for i, hidden_dim in enumerate(fc_layers_dims):
            fc_layer = nn.Linear(fc_input_size, hidden_dim)
            fc_modules.append(fc_layer)
            fc_modules.append(nn.ReLU())
            print(f"  Layer {i+1}: Linear({fc_input_size}, {hidden_dim}), ReLU")
            fc_input_size = hidden_dim # Output of this layer is input to the next

        # Add the final output layer (outputs Q-values for each action)
        output_layer = nn.Linear(fc_input_size, output_dim)
        fc_modules.append(output_layer)
        print(f"  Output Layer: Linear({fc_input_size}, {output_dim})")

        self.fc = nn.Sequential(*fc_modules)

        # Move the entire model to the chosen device
        self.to(self.device)


    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor representing the board state(s).
                              Shape: (batch_size, input_channels, input_height, input_width).

        Returns:
            torch.Tensor: Output tensor of Q-values for each action.
                          Shape: (batch_size, output_dim).
        """
        # Pass input through convolutional layers
        x = self.conv(x)

        # Flatten the output from CNN layers before feeding into FC layers
        # x.view(batch_size, -1) automatically calculates the flattened size
        x = x.view(x.size(0), -1) # Flatten all dimensions except batch

        # Pass the flattened data through fully connected layers
        output = self.fc(x)
        return output

    def get_epsilon(self, current_episode, total_episodes_for_decay, initial_epsilon, minimum_epsilon):
        """
        Calculates the epsilon value for epsilon-greedy exploration using exponential decay.

        Args:
            current_episode (int): The current training episode number.
            total_episodes_for_decay (int): The total number of episodes over which epsilon should decay.
            initial_epsilon (float): The starting value of epsilon (probability of random action).
            minimum_epsilon (float): The minimum value epsilon should decay to.

        Returns:
            float: The calculated epsilon value for the current step.
        """
        if total_episodes_for_decay <= 0: # Avoid division by zero or invalid decay
            return minimum_epsilon

        # Calculate decay rate such that epsilon reaches minimum_epsilon after total_episodes_for_decay
        # formula: epsilon = min_epsilon + (max_epsilon - min_epsilon) * exp(-decay_rate * step)
        # Simplified exponential decay: epsilon = initial_epsilon * exp(-decay_rate * step)
        # Let's use a rate ensuring it hits min_epsilon around total_episodes_for_decay
        # A common approach: decay_rate = -ln(min_epsilon / initial_epsilon) / total_episodes_for_decay
        # Ensure initial_epsilon > minimum_epsilon > 0 for log calculation
        if initial_epsilon <= minimum_epsilon or minimum_epsilon <= 0:
             # Fallback or fixed decay if parameters are unusual
             decay_rate = 1.0 / total_episodes_for_decay # Simple linear factor proxy
        else:
            decay_rate = -math.log(minimum_epsilon / initial_epsilon) / total_episodes_for_decay

        current_epsilon = initial_epsilon * math.exp(-decay_rate * current_episode)

        # Ensure epsilon doesn't go below the minimum value
        return max(current_epsilon, minimum_epsilon)

    def select_action(self, state_flat, valid_actions, epsilon):
        """
        Selects an action using an epsilon-greedy policy with masking for invalid actions.

        Args:
            state_flat (np.array): The normalized (player=1, opp=-1), flattened board state (42,).
            valid_actions (list): A list of column indices that are currently valid moves.
            epsilon (float): The current probability of choosing a random action.

        Returns:
            int: The selected action (column index).
        """
        # Exploration: Choose a random valid action
        if random.random() < epsilon:
            # Ensure valid_actions is not empty (shouldn't happen in Connect Four if not done)
            if not valid_actions:
                 # This case should ideally not be reached if called correctly before game ends
                 # Or could signify an env error. Default to a failsafe if necessary.
                 print("Warning: select_action called with no valid actions!")
                 return 0 # Or raise error, depends on desired handling
            return random.choice(valid_actions)
        # Exploitation: Choose the best action according to the Q-network
        else:
            with torch.no_grad(): # Disable gradient calculation for inference
                # Convert the flat numpy state to a PyTorch tensor
                # Ensure state is float, add batch and channel dimensions, move to device
                # Expected shape: [1, 1, 6, 7] (batch_size, channels, height, width)
                state_tensor = torch.from_numpy(state_flat).float().view(1, 1, 6, 7).to(self.device)

                # Get Q-values from the network
                q_values = self.forward(state_tensor) # Shape: [1, num_actions]

                # --- Masking Invalid Actions ---
                # Create a mask tensor filled with negative infinity
                mask = torch.full_like(q_values, float('-inf'), device=self.device)
                # Set the Q-values for valid actions to 0 in the mask
                # (so they are unaffected when added/used in where)
                valid_action_indices = torch.tensor(valid_actions, device=self.device, dtype=torch.long)
                mask[0, valid_action_indices] = 0.0

                # Apply the mask: Q-values for invalid actions become -infinity
                masked_q_values = q_values + mask
                # Alternative using torch.where:
                # valid_mask_bool = torch.zeros_like(q_values, dtype=torch.bool, device=self.device)
                # valid_mask_bool[0, valid_action_indices] = True
                # masked_q_values = torch.where(valid_mask_bool, q_values, torch.tensor(float('-inf'), device=self.device))


                # Select the action with the highest Q-value among the valid ones
                chosen_action = masked_q_values.argmax(dim=1).item()
                return chosen_action


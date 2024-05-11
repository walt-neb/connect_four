

#filename: ddqn_agent_cnn.py
# Convolutional layers in PyTorch expect the input to be in the form of a 4D tensor: 
# [N,C,H,W], where:
# N is the batch size,
# C is the number of channels,
# H is the height,
# W is the width.


import math
import random
import torch
import torch.nn as nn
import numpy as np

class CNNDDQNAgent(nn.Module):
    def __init__(self, input_channels, input_height, input_width, output_dim, conv_layers, fc_layers):
        super(CNNDDQNAgent, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Example convolutional layers setup
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)  # Output: (16, 6, 7)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Output: (32, 6, 7)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: (64, 6, 7)

        # Calculate the total number of features after flattening
        self.num_flattened_features = 64 * 6 * 7  # Change dimensions according to actual output of conv layers
        
        # Fully connected layers
        fc_layers.insert(0, self.num_flattened_features)
        layers = []
        for i in range(1, len(fc_layers)):
            layers.append(nn.Linear(fc_layers[i-1], fc_layers[i]))
            layers.append(nn.ReLU())
        self.fc = nn.Sequential(*layers)
        
        # Output layer
        self.out = nn.Linear(fc_layers[-1], output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.num_flattened_features)  # Flatten the tensor
        x = self.fc(x)
        x = self.out(x)
        return x

    def get_epsilon(self, epsilon_step_count, num_episodes, initial_epsilon, minimum_epsilon):
        decay_rate = -math.log(minimum_epsilon / initial_epsilon) / num_episodes
        current_epsilon = initial_epsilon * math.exp(-decay_rate * epsilon_step_count)
        return max(current_epsilon, minimum_epsilon)

    def select_action(self, state_array, valid_actions, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                # Convert the NumPy array to a PyTorch tensor, ensure it's float, and add batch and channel dimensions
                state_tensor = torch.from_numpy(state_array).float()
                # Reshape the state tensor from (42,) to [1, 1, 6, 7] (batch size, channels, height, width)
                state_tensor = state_tensor.view(1, 1, 6, 7)  # Adjusted from unsqueeze to view for correct reshaping
                state_tensor = state_tensor.to(self.device)  # Move tensor to the correct device

                # Get Q-values
                q_values = self.forward(state_tensor)

                # Create a mask to invalidate actions
                inf_mask = torch.full_like(q_values, float('inf'), device=self.device)
                valid_mask = torch.zeros_like(q_values, device=self.device)
                valid_mask[0, valid_actions] = 1  # Set valid actions to 1

                # Apply mask: Set invalid actions to negative infinity
                masked_q_values = torch.where(valid_mask.bool(), q_values, -inf_mask)

                # Select the best action from masked Q-values
                chosen_action = masked_q_values.argmax(dim=1).item()
                return chosen_action
        else:
            return random.choice(valid_actions)  # Randomly choose from valid actions
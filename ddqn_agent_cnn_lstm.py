

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
    def __init__(self, input_channels, input_height, input_width, output_dim, 
                 conv_layers, lstm_layers, fc_layers): 
        super(CNNDDQNAgent, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create convolutional layers dynamically
        current_channels = input_channels
        current_height = input_height
        current_width = input_width
        modules = []

        for layer_params in conv_layers:
            out_channels = layer_params[0]
            kernel_size = layer_params[1]
            stride = layer_params[2]
            # Check if padding is provided, otherwise assume padding is 0
            padding = layer_params[3] if len(layer_params) > 3 else 0

            modules.append(nn.Conv2d(current_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            modules.append(nn.ReLU())  # Typically, a non-linear activation is added after each conv layer
            current_channels = out_channels
            # Update current height and width
            current_height = (current_height + 2 * padding - kernel_size) // stride + 1
            current_width = (current_width + 2 * padding - kernel_size) // stride + 1
            # Check for dimension validity
            if current_height <= 0 or current_width <= 0:
                raise ValueError("Convolutional settings result in a non-positive dimension size.")

        self.conv = nn.Sequential(*modules)

        # Calculate the total number of features after flattening
        self.num_flattened_features = current_channels * current_height * current_width


        #fully connected linear layer between the CNN and the LSTM to transform the CNN output to the desired dimensional input for the LSTM:
        self.cnn_to_lstm = nn.Linear(self.num_flattened_features, lstm_layers[0][1]) 


        # --- LSTM Layers ---
        lstm_modules = []
        for i, (num_layers, hidden_size) in enumerate(lstm_layers):
            if i == 0:  # First LSTM layer
                input_size = self.num_flattened_features
            else:
                input_size = lstm_layers[i-1][1]  # Input size is previous layer's hidden size
            lstm_modules.append(nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                                        num_layers=num_layers, batch_first=True))

        self.lstm = nn.Sequential(*lstm_modules)  # Stack LSTM layers

        # --- Fully Connected Layers ---
        lstm_hidden_size = lstm_layers[-1][1]  # Last LSTM's hidden size is the input to FC
        all_fc_layers = [lstm_hidden_size] + fc_layers

        fc_modules = []
        for in_features, out_features in zip(all_fc_layers[:-1], all_fc_layers[1:]):
            fc_modules.append(nn.Linear(in_features, out_features))
            fc_modules.append(nn.ReLU())
        
        # Output layer
        fc_modules.append(nn.Linear(all_fc_layers[-1], output_dim))
        self.fc = nn.Sequential(*fc_modules)

    def forward(self, x):
        # --- CNN
        x = self.conv(x)
        
        # Flatten the output from the convolutional layers 
        x = x.view(-1, self.num_flattened_features)

        x = self.cnn_to_lstm(x)  # Transform CNN output to LSTM input size
        
        # --- LSTM ---
        lstm_out = x.unsqueeze(1)  # Add time dimension
        for lstm_layer in self.lstm:
            lstm_out, _ = lstm_layer(lstm_out)  # Process each LSTM layer separately
    
        # Take the last hidden state of the last LSTM layer 
        lstm_out = lstm_out[:, -1, :]

        # --- FC ---
        output = self.fc(lstm_out)
        return output

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
        

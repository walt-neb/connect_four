

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

import torch
import torch.nn as nn
import numpy as np

class CNN3D(nn.Module):
    def __init__(self, input_channels, seq_length, input_height, input_width, output_dim, conv_layers, fc_layers):
        super(CNN3D, self).__init__()
        self.seq_length = seq_length

        print("CNN3D")
        print(f'input_channels: {input_channels}')
        print(f'seq_length: {seq_length}')
        print(f'input_height: {input_height}')
        print(f'input_width: {input_width}')
        print(f'output_dim: {output_dim}')
        print(f'conv_layers: {conv_layers}')
        print(f'fc_layers: {fc_layers}')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'device: {self.device}')

        # Create convolutional layers dynamically
        self.conv = nn.Sequential(
            # The input to this layer is expected to be [batch_size, channels, depth, height, width]
            nn.Conv3d(input_channels, conv_layers[0][0], kernel_size=conv_layers[0][1], stride=conv_layers[0][2], padding=conv_layers[0][3]),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),  # Example of adding pooling to reduce spatial dimensions
        )
        
        # Calculate the size of the flattened conv output
        self.flatten_size = self._get_conv_output([1, input_channels, seq_length, input_height, input_width])
        
        # Setup fully connected layers
        fc_modules = []
        in_features = self.flatten_size
        for out_features in fc_layers:
            fc_modules.append(nn.Linear(in_features, out_features))
            fc_modules.append(nn.ReLU())
            in_features = out_features
        fc_modules.append(nn.Linear(in_features, output_dim))
        
        self.fc = nn.Sequential(*fc_modules)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.rand(shape)
            output = self.conv(input)
            return int(np.prod(output.size()[1:]))

    def forward(self, x):
        print("Input shape:", x.shape)
        x = self.conv(x)
        print("After conv3d shape:", x.shape)
        x = x.view(x.size(0), -1)
        print("Before FC shape:", x.shape)
        x = self.fc(x)
        print("Output shape:", x.shape)
        return x


    def get_epsilon(self, epsilon_step_count, num_episodes, initial_epsilon, minimum_epsilon):
        decay_rate = -math.log(minimum_epsilon / initial_epsilon) / num_episodes
        current_epsilon = initial_epsilon * math.exp(-decay_rate * epsilon_step_count)
        return max(current_epsilon, minimum_epsilon)


    def select_action(self, state_seq, valid_actions, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                # Convert the NumPy array to a PyTorch tensor, ensure it's float, and format it correctly
                state_seq_array = np.array(state_seq, dtype=np.float32).reshape((-1, self.seq_length, 6, 7))
                state_seq_tensor = torch.tensor(state_seq_array).to(self.device)  # Convert to tensor and send to device
                state_seq_tensor = state_seq_tensor.unsqueeze(1)  # Adding the channel dimension: [batch, channels, height, width]

                # Get Q-values from the network
                q_values = self.forward(state_seq_tensor).squeeze()  # Assume output is (1, num_actions), and we remove batch dim

                # Masking invalid actions by setting their Q-values to a very large negative value
                mask = torch.ones_like(q_values, dtype=torch.bool)  # Start with a mask that invalidates all actions
                for action in valid_actions:
                    mask[action] = False  # Unmask valid actions
                
                q_values[mask] = float('-inf')  # Set Q-values of invalid actions to negative infinity

                # Choose the action with the highest Q-value that is also valid
                chosen_action = q_values.argmax().item()
                return chosen_action
        else:
            return random.choice(valid_actions)  # Randomly choose from valid actions


        

        

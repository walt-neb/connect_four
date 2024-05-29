

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seq_length = seq_length
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width

        # Setup convolutional layers
        self.conv_modules = nn.Sequential()
        current_channels = input_channels
        for idx, (out_channels, kernel_size, stride, padding) in enumerate(conv_layers):
            print(f'adding conv3d_{idx}, current_channels: {current_channels}, out_channels: {out_channels}, kernel_size: {kernel_size}, stride: {stride}, padding: {padding}')
            self.conv_modules.add_module(f"conv3d_{idx}", nn.Conv3d(current_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            self.conv_modules.add_module(f"relu_{idx}", nn.ReLU())
            print(f"adding relu_{idx}")
            current_channels = out_channels

        # Calculate flatten size
        self.flatten_size = self._get_conv_output([1, input_channels, seq_length, input_height, input_width])
        
        # Setup fully connected layers
        self.fc_modules = nn.Sequential()
        in_features = self.flatten_size
        for out_features in fc_layers:
            print(f'adding linear_{len(self.fc_modules)//2}, in_features: {in_features}, out_features: {out_features}') 
            self.fc_modules.add_module(f"linear_{len(self.fc_modules)//2}", nn.Linear(in_features, out_features))
            print(f'adding relu_{len(self.fc_modules)//2}')
            self.fc_modules.add_module(f"relu_{len(self.fc_modules)//2}", nn.ReLU())
            in_features = out_features
        self.fc_modules.add_module("output", nn.Linear(in_features, output_dim))

    def _get_conv_output(self, shape):
        with torch.no_grad():
            print(f'conv_output:')
            input = torch.rand(shape).to(self.device)
            print(f'input.shape: {input.shape}')
            output = self.conv_modules(input)
            print(f'output.shape: {output.shape}')
            total_features = int(np.prod(output.size()[1:]))
            print(f'total_features: {total_features}')
            return total_features

    def forward(self, x):
        print(f'x.shape: {x.shape} before view')
        x = self.conv_modules(x)
        print(f'x.shape: {x.shape} after conv_modules(x)')
        x = x.view(x.size(0), -1)
        print(f'x.shape: {x.shape} after view before fc_modules(x)')
        x = self.fc_modules(x)
        print(f'x.shape: {x.shape} after fc')
        return x

    def select_action(self, state, valid_actions, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.tensor(state, device=self.device).float().view(-1, 1, 6, 7).unsqueeze(2)  # Adjusting for batch, channel, depth
                q_values = self.forward(state_tensor)
                mask = torch.ones_like(q_values, dtype=torch.bool)
                mask[:, valid_actions] = False
                q_values[mask] = float('-inf')
                return q_values.argmax().item()
        else:
            return random.choice(valid_actions)
        

    def get_epsilon(self, epsilon_step_count, num_episodes, initial_epsilon, minimum_epsilon):
        decay_rate = -math.log(minimum_epsilon / initial_epsilon) / num_episodes
        current_epsilon = initial_epsilon * math.exp(-decay_rate * epsilon_step_count)
        return max(current_epsilon, minimum_epsilon)



        

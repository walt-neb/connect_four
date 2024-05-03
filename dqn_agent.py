

#filename: dqn_agent.py

import math
import random
import torch
import torch.nn as nn
import numpy as np

class DQNAgent(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(DQNAgent, self).__init__()
        layers = []

        # Create the first layer from input dimension to the first hidden layer size
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(nn.LeakyReLU())

        # Create all hidden layers
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(nn.LeakyReLU())

        # Create the final layer from the last hidden layer to the output dimension
        layers.append(nn.Linear(hidden_layers[-1], output_dim))

        # Wrap all defined layers in a Sequential module
        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x):
        return self.network(x)
    
    def get_epsilon(self, epsilon_step_count, num_episodes, initial_epsilon, minimum_epsilon):
        decay_rate = -math.log(minimum_epsilon / initial_epsilon) / num_episodes
        current_epsilon = initial_epsilon * math.exp(-decay_rate * epsilon_step_count)
        return max(current_epsilon, minimum_epsilon)

    def select_action(self, state_list, valid_actions, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                states_tensor = torch.tensor([state_list], dtype=torch.float32)  # Ensure it's a batch of one

                # Get Q-values
                q_values = self.forward(states_tensor)

                # Create a mask to invalidate actions
                inf_mask = torch.full_like(q_values, float('inf'))  # Create a mask with 'inf'
                valid_mask = torch.zeros_like(q_values)  # Assume all actions are invalid first
                valid_mask[0, valid_actions] = 1  # Set valid actions to 1

                # Apply mask: Set invalid actions to negative infinity
                masked_q_values = torch.where(valid_mask.bool(), q_values, -inf_mask)

                # Select the best action from masked Q-values
                chosen_action = masked_q_values.argmax(dim=1).item()
                return chosen_action
        else:
            return random.choice(valid_actions)  # Randomly choose from valid actions


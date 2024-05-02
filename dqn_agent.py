

#filename: dqn_agent.py

import math
import random
import torch
import torch.nn as nn

class DQNAgent(nn.Module):
    def __init__(self, input_dim, output_dim, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=500):
        super(DQNAgent, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128), 
            nn.LeakyReLU(),           
            nn.Linear(128, output_dim)
        )

        self.output_dim = output_dim

    def forward(self, x):
        return self.network(x)
    
    def get_epsilon(self, epsilon_step_count, num_episodes, initial_epsilon, minimum_epsilon):
        decay_rate = -math.log(minimum_epsilon / initial_epsilon) / num_episodes
        current_epsilon = initial_epsilon * math.exp(-decay_rate * epsilon_step_count)
        return max(current_epsilon, minimum_epsilon)

    def select_action(self, state, valid_actions, epsilon):
        
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.tensor([state], dtype=torch.float32)
                q_values = self.forward(state)
                # Mask out invalid actions by setting their Q-values to a very large negative value
                q_values[:, [a for a in range(self.output_dim) if a not in valid_actions]] = float('-inf')
                return q_values.max(1)[1].item()
        else:
            return random.choice(valid_actions)  # Explore by choosing randomly from valid actions



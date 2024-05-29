import torch.optim as optim
import numpy as np

from c4_env_tr import ConnectFourEnv
from c4_replay_buf import ReplayBuffer
from c4_transformer import TransformerAgent
import torch
import torch.nn as nn


# Hyperparameters
input_dim = 42
embed_dim = 128
n_heads = 4
ff_dim = 512
n_layers = 2
output_dim = 7
dropout = 0.1
capacity = 10000
batch_size = 32
gamma = 0.99
num_epochs = 1000
learning_rate = 1e-4

# Initialize components
env = ConnectFourEnv()
agent = TransformerAgent(input_dim, embed_dim, n_heads, ff_dim, n_layers, output_dim, dropout)
replay_buffer = ReplayBuffer(capacity)
optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

loss = torch.tensor(0.0)

# Training loop
for epoch in range(num_epochs):
    state, player = env.reset()
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = agent(state_tensor)
        valid_actions = env.get_valid_actions()
        # Create a mask for valid actions
        mask = torch.ones(output_dim) * float('-inf')
        mask[valid_actions] = 0
        masked_q_values = q_values + mask
        action = torch.argmax(masked_q_values).item()

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

            print(f"State batch shape: {state_batch.shape}")
            print(f"Action batch shape: {action_batch.shape}")
            print(f"Reward batch shape: {reward_batch.shape}")
            print(f"Next state batch shape: {next_state_batch.shape}")
            print(f"Done batch shape: {done_batch.shape}")

            q_values = agent(state_batch)  # Shape: [batch_size, output_dim]
            next_q_values = agent(next_state_batch).detach()  # Shape: [batch_size, output_dim]

            print(f"Q-values shape: {q_values.shape}")
            print(f"Next Q-values shape: {next_q_values.shape}")

            # Ensure max_next_q_values has the correct shape [batch_size]
            max_next_q_values = torch.max(next_q_values, dim=1)[0]  # Shape: [batch_size]

            print(f"Max next Q-values shape: {max_next_q_values.shape}")

            # Ensure reward_batch and done_batch have the correct shapes [batch_size]
            reward_batch = reward_batch.unsqueeze(1)  # Shape: [batch_size, 1]
            done_batch = done_batch.unsqueeze(1)  # Shape: [batch_size, 1]

            print(f"Reward batch reshaped: {reward_batch.shape}")
            print(f"Done batch reshaped: {done_batch.shape}")

            # Calculate target_q_values
            target_q_values = reward_batch + gamma * max_next_q_values.unsqueeze(1) * (1 - done_batch)

            print(f"Target Q-values shape: {target_q_values.shape}")

            action_batch = action_batch.unsqueeze(1)  # Shape: [batch_size, 1]

            print(f"Action batch reshaped: {action_batch.shape}")

            q_value = q_values.gather(1, action_batch).squeeze(1)  # Shape: [batch_size]

            print(f"Gathered Q-values shape: {q_value.shape}")

            loss = criterion(q_value, target_q_values.squeeze(1))

            print(f"Loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        state = next_state.flatten()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

print("Training complete.")

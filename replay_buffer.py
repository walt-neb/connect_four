

#filename:  replay_buffer.py

import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Replay buffer parameters
buffer_capacity = 10000
batch_size = 64

# Initialize replay buffers for both agents
replay_buffer1 = ReplayBuffer(buffer_capacity)
replay_buffer2 = ReplayBuffer(buffer_capacity)


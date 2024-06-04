
#c4_replay_buf.py

import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.prob_alpha = 0.6  # You can tune this parameter

    def add(self, state, action, reward, next_state, done):
        # Use the max priority for new elements if the buffer is not empty
        max_prio = max(self.priorities) if len(self.buffer) > 0 else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_prio)  # Initialize with max priority

    def sample(self, batch_size, beta=0.4):  # Beta can also be tuned
        if len(self.buffer) == 0:
            return [], []

        probabilities = np.array(self.priorities) ** self.prob_alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability

        return samples, weights, indices

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


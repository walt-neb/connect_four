# filename: replay_buffer.py
"""
Simple Replay Buffer Implementation.

This module provides a basic ReplayBuffer class using Python's `collections.deque`
to store experiences (transitions) for reinforcement learning agents. It allows
adding new experiences and sampling random batches of experiences.
"""

import random
from collections import deque

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer.

    Stores transitions and allows sampling batches. Useful for breaking
    correlations between consecutive experiences during training.
    """
    def __init__(self, capacity):
        """
        Initializes the ReplayBuffer.

        Args:
            capacity (int): The maximum number of transitions to store in the buffer.
                            Older transitions are discarded when capacity is reached.
        """
        # Use deque for efficient appends and pops from both ends
        # maxlen ensures the deque never exceeds the specified capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Adds a transition to the buffer.

        The transition components are stored as a tuple.

        Args:
            state: The state observed before the action.
            action: The action taken.
            reward: The reward received after the action.
            next_state: The state observed after the action.
            done (bool): Whether the episode terminated after this transition.
        """
        # Note: The state and next_state pushed here should ideally be
        # normalized from the perspective of the agent that took the action.
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Samples a batch of experiences randomly from the buffer.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            list: A list of transition tuples, or an empty list if the buffer
                  contains fewer transitions than the requested batch size.
                  Returns None if batch_size is invalid or buffer is too small.
        """
        # Ensure we don't try to sample more than what's available
        if batch_size <= 0 or batch_size > len(self.buffer):
             # Return empty list or handle as needed if buffer is smaller than batch_size
             # For simplicity, we return what's available if less than batch_size,
             # but the training loop handles the len < batch_size case explicitly.
             # Here we ensure random.sample doesn't get an invalid k.
             actual_batch_size = min(batch_size, len(self.buffer))
             if actual_batch_size <= 0:
                 return [] # Cannot sample if buffer is empty or batch_size is non-positive
        else:
            actual_batch_size = batch_size

        # random.sample selects unique elements without replacement
        return random.sample(self.buffer, actual_batch_size)


    def __len__(self):
        """
        Returns the current number of transitions stored in the buffer.
        """
        return len(self.buffer)


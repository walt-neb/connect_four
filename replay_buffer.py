#replay_buffer.py

import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, sequence_length):
        self.buffer = deque(maxlen=capacity)
        self.sequence_length = sequence_length

    def get_len(self):
        return len(self.buffer)
    
    def set_env(self, env):
        self.env = env  # Store the environment for debugging
    
    def push(self, state_seq, action, reward, next_state_seq, done): 
        """Add a sequence of experiences to the replay buffer.

        Args:
            state_seq (list or np.ndarray): A sequence of states
            action (int): The action taken at the last state in the sequence
            reward (float): The reward received after taking the action
            next_state_seq (list or np.ndarray): The sequence of states after the action
            done (bool): Whether the episode ended after the action
        Indexing Sequenced Replay Buffer:
        replay_buffer.buffer: This accesses the entire buffer which contains all the sequences
        replay_buffer.buffer[i]: This accesses the i-th sequence in the buffer
        replay_buffer.buffer[i][j]: This accesses the j-th transition in the i-th sequence
        replay_buffer.buffer[i][j][k]: This accesses the k-th element of the j-th transition in the i-th sequence 
            (k = 0 for state, 1 for action, 2 for reward, 3 for next_state, 4 for done)
        """

        # Ensure both state sequences have the same length as specified by self.sequence_length
        if len(state_seq) != self.sequence_length or len(next_state_seq) != self.sequence_length:
            print(f'len(state_seq): {len(state_seq)}, len(next_state_seq): {len(next_state_seq)}')
            print(f'self.sequence_length: {self.sequence_length}')
            print(f'state_seq: {state_seq}')
            raise ValueError(f"push error - State sequences must have length {self.sequence_length}")

        # Create an experience for the last step in the sequence
        last_experience = (state_seq[-1], action, reward, next_state_seq[-1], done)

        # Check if the buffer is empty or the last sequence is complete
        if len(self.buffer) == 0 or self.buffer[-1][-1][4]:  # Check done flag of last experience in last sequence
            self.buffer.append(deque(maxlen=self.sequence_length))  # Start a new sequence

        # Append the experiences to the current sequence
        for i in range(self.sequence_length - 1):
            self.buffer[-1].append((state_seq[i], 0, 0, next_state_seq[i], False))  # Actions/rewards/dones are irrelevant for intermediate states
        self.buffer[-1].append(last_experience)  # Append the last experience

    def sample(self, batch_size):
        # Ensure enough sequences for a batch
        sb_len = len(self.buffer)
        if sb_len < batch_size:
            return []

        # Sample random complete sequences
        sequences = random.sample(
            [seq for seq in self.buffer if len(seq) == self.sequence_length], 
            batch_size
        )

            # Debugging: Log sampled sequences for visual inspection
        for i, seq in enumerate(sequences):
            print(f"ReplayBuffer push - Sampled Sequence {i+1}")
            for transition in seq:
                print(transition[0])  # printing the state

        # Transpose to get batches of states, actions, rewards, next states, and dones
        return list(zip(*sequences))




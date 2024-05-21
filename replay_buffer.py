#replay_buffer.py

import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, sequence_length):
        self.buffer = deque(maxlen=capacity)
        self.sequence_length = sequence_length

    def push(self, state_seq, action, reward, next_state_seq, done): 
        """Add a sequence of experiences to the replay buffer.

        Args:
            state_seq (list or np.ndarray): A sequence of states.
            action (int): The action taken at the last state in the sequence.
            reward (float): The reward received after taking the action.
            next_state_seq (list or np.ndarray): The sequence of states after the action. 
            done (bool): Whether the episode ended after the action.
        """
        if len(self.buffer) > 0 and not self.buffer[-1][-1][4]:  # if the buffer is not empty and the last sequence is not done
            last_state = self.buffer[-1][-1][3]  # get last next_state from the last transition in the buffer
            if not np.array_equal(last_state, state_seq[0]):
                print("Error: State transition mismatch")
                print("Expected state:", last_state)
                print("Received state:", state_seq[0])
                return  # Optionally return to skip adding this transition if it is not valid


        # Ensure both state sequences have the same length as specified by self.sequence_length
        if len(state_seq) != self.sequence_length or len(next_state_seq) != self.sequence_length:
            print(f'len(state_seq): {len(state_seq)}, len(next_state_seq): {len(next_state_seq)}')
            print(f'self.sequence_length: {self.sequence_length}')
            print(f'state_seq: {state_seq}')
            raise ValueError(f"State sequences must have length {self.sequence_length}")

        # Create an experience for the last step in the sequence
        last_experience = (state_seq[-1], action, reward, next_state_seq[-1], done)

        # Check if the buffer is empty or the last sequence is complete
        if len(self.buffer) == 0 or self.buffer[-1][-1][4]:  # Check done flag of last experience in last sequence
            self.buffer.append(deque(maxlen=self.sequence_length))  # Start a new sequence

        # Append the experiences to the current sequence
        for i in range(self.sequence_length - 1):
            self.buffer[-1].append((state_seq[i], 0, 0, next_state_seq[i], False))  # Actions/rewards/dones are irrelevant for intermediate states
        self.buffer[-1].append(last_experience)  # Append the last experience

        # Debugging: Check and log the continuity of states
        if len(self.buffer[-1]) > 1:
            last_state_in_seq = self.buffer[-1][-2][3]  # previous next_state
            current_state = state_seq[0]
            if not np.array_equal(last_state_in_seq, current_state):
                print("ReplayBuffer Push Error:")
                print("Discontinuity detected between sequences")
                print("Last state in seq:", last_state_in_seq)
                print("Current state:", current_state)



    def sample(self, batch_size):
        # Ensure enough sequences for a batch
        if len(self.buffer) < batch_size:
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




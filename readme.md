# Connect Four AI - DDQN with CNN

This project implements an AI agent that learns to play Connect Four using a Double Deep Q-Network (DDQN) algorithm with a Convolutional Neural Network (CNN) for state representation. The agents are trained via self-play.

## Features

* Connect Four environment adhering to a Gym-like API.
* DDQN agent implementation using PyTorch, featuring a configurable CNN + FC network architecture.
* Self-play training loop (`train_c4.py`) for two agents.
* Shared Experience Replay Buffer.
* Normalized board state representation for the network (Player=1, Opponent=-1, Empty=0).
* Epsilon-greedy exploration strategy with decay.
* Periodic Hard Target Network Updates.
* Gradient Clipping and Huber Loss option for training stability.
* Robust checkpointing system for resuming training (saves episode, model states, optimizer states).
* Separate saving for replay buffer.
* TensorBoard integration for monitoring training progress.
* Scripts provided for:
    * Playing against a trained agent (`play_human.py`).
    * Playing two trained agents against each other (`play_two_models.py`).
    * Extracting individual model weights from checkpoints (`disassemble_checkpoint.py`).

## Directory Structure

```text
connect_four/
│
├── hyps/                     # Hyperparameter configuration files (.hyp)
│   ├── h4_cnn_optimized.hyp
│   └── ...
│
├── runs/                     # TensorBoard log directories (created automatically)
│   └── h4_cnn_optimized_YYYYMMDD_HHMMSS/
│       ├── events.out.tfevents...
│       └── h4_cnn_optimized.hyp (copy)
│       └── final_summary.txt
│
├── wts/                      # Saved weights and checkpoints (created automatically)
│   ├── checkpoint_*.pth      # Combined checkpoints for resuming training
│   ├── replay_buffer_*.pkl   # Saved replay buffers
│   ├── m1_*.pth              # Extracted weights for Agent 1 (for playing)
│   └── m2_*.pth              # Extracted weights for Agent 2 (for playing)
│
├── ddqn_agent_cnn.py         # Agent class definition (CNN + FC)
├── two_player_env.py         # Connect Four environment logic
├── replay_buffer.py          # Replay buffer class
├── train_c4.py               # Main training script
├── play_human.py             # Script to play human vs AI
├── play_two_models.py        # Script to play AI vs AI
├── disassemble_checkpoint.py # Tool to extract weights from checkpoint
└── README.md                 # This file
'''

## Requirements & Setup

1.  **Python:** Developed with Python 3.10 (should work with >= 3.8).
2.  **PyTorch:** Install based on your system and CUDA version (if using GPU). See [pytorch.org](https://pytorch.org/).
3.  **Other Libraries:** Install required packages using pip:
    ```bash
    # Activate your virtual environment (e.g., venv3.10) first
    pip install numpy torch tensorboard # Add torchvision if needed by PyTorch install
    ```
4.  **Environment:** Ensure you are in the project's root directory (`connect_four`) when running scripts. Create the `hyps/` and `wts/` directories if they don't exist.

## Configuration (`.hyp` Files)

Training parameters are controlled via `.hyp` files located in the `hyps/` directory. Key parameters include:

* `end_episode`: Total episodes for training.
* `learning_rate`: Agent learning rates.
* `batch_size`: Samples per training step.
* `gamma`: Discount factor.
* `max_replay_buffer_size`: Capacity of the replay buffer.
* `cnn_a1`/`cnn_a2`: List of tuples defining CNN layers `(out_channels, kernel, stride, padding)`.
* `fc_a1`/`fc_a2`: List of integers defining hidden FC layer sizes.
* `target_update_frequency`: Episodes between target network hard updates.
* `ckpt_interval`: Episodes between saving checkpoints.
* Epsilon decay parameters (`aX_epsilon_start`/`end`).
* Logging intervals (`short_log_interval`, `console_status_interval`, `tensorboard_status_interval`).

The play scripts (`play_human.py`, `play_two_models.py`) rely on finding a corresponding `.hyp` file in the `hyps/` directory (matching the base name of the model weights file) to load the correct network architecture.

## Usage

Activate your Python environment before running commands.

### Training

* **Start New Run:**
    ```bash
    python train_c4.py hyps/<your_config_name>.hyp
    ```
* **Resume from Checkpoint:** Requires a combined checkpoint file (`checkpoint_*.pth`) and the corresponding separate buffer file (`replay_buffer_*.pkl`).
    ```bash
    python train_c4.py hyps/<your_config_name>.hyp \
                       --resume_from_checkpoint wts/checkpoint_<name>_ep<N>.pth \
                       --load_buffer wts/replay_buffer_<name>_ep<N>.pkl
    ```
* **Start with Initial Weights/Buffer (No Checkpoint):** Useful for starting from specific weights but resetting optimizers/episode count (requires manually setting `start_episode` in `.hyp` file).
    ```bash
    # Example: Start from ep 22001 using weights/buffer from ep 22000
    # (Ensure start_episode = 22001 is set in the .hyp file)
    python train_c4.py hyps/<your_config_name>.hyp \
                       --load_agent1_weights wts/m1_<name>_ep22000.pth \
                       --load_agent2_weights wts/m2_<name>_ep22000.pth \
                       --load_buffer wts/replay_buffer_<name>_ep22000.pkl
    ```

### Monitoring Training

* While `train_c4.py` is running, open a *new terminal*.
* Navigate to the project root directory (`connect_four`).
* Activate the same Python environment.
* Run: `tensorboard --logdir runs`
* Open the URL provided (e.g., `http://localhost:6006/`) in your web browser.

### Extracting Weights for Playing

* Training saves comprehensive checkpoints (`checkpoint_*.pth`). To get individual weights for play scripts, use the disassembly tool.
* Choose a checkpoint file.
* Run the tool:
    ```bash
    python disassemble_checkpoint.py wts/checkpoint_<name>_ep<N>.pth
    ```
    (Optional: `--output_dir <path>` to save extracted weights elsewhere).
* This will create `m1_<name>_ep<N>.pth` and `m2_<name>_ep<N>.pth` in the `./wts/` directory (or the specified output directory).

### Playing vs Human

* Requires an extracted individual weight file (e.g., `m2_*.pth` if you want to play against Agent 2).
* Ensure the corresponding `.hyp` file exists in `hyps/`.
    ```bash
    python play_human.py wts/m2_<name>_ep<N>.pth [num_games]
    ```
    (Human is always Player 1 'X', loaded AI is Player 2 'O').

### Playing Model vs Model

* Requires two extracted individual weight files (e.g., `m1_*.pth`, `m2_*.pth`).
* Ensure the corresponding `.hyp` file(s) exist in `hyps/`.
    ```bash
    python play_two_models.py wts/m1_<name>_ep<N1>.pth wts/m2_<name>_ep<N2>.pth [num_games]
    ```
    (The first model path corresponds to Player 1 'X', the second to Player 2 'O').

## File Descriptions

* **`ddqn_agent_cnn.py`**: Defines the `CNNDDQNAgent` neural network model and its `select_action` logic.
* **`two_player_env.py`**: Implements the Connect Four game rules and environment interactions (`step`, `reset`, `render`).
* **`replay_buffer.py`**: Simple class for the experience replay buffer using `deque`.
* **`train_c4.py`**: The main script coordinating agent training, self-play, logging, checkpointing, and resuming.
* **`play_human.py`**: Loads a trained agent and allows a human user to play against it.
* **`play_two_models.py`**: Loads two trained agents and plays them against each other, reporting results.
* **`disassemble_checkpoint.py`**: Utility script to extract individual model weights from a combined training checkpoint.

## Future Work / Improvements (Optional)

* Implement learning rate scheduling.
* Experiment with Prioritized Experience Replay (PER).
* Explore alternative network architectures (e.g., ResNet blocks).
* Consider different RL algorithms (e.g., PPO, MCTS-based approaches like AlphaZero for potentially stronger play).
* Add more sophisticated evaluation metrics during training.
* Package as installable module.

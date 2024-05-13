# Connect Four AI
This project implements a Connect Four game with a deep reinforcement learning agent using a double deep Q-network (DDQN). The agent is trained with convolutional neural networks to evaluate the game state and make decisions.

## Directory Structure
connect_four/
├── check_cnn_fc_match.py # Script to check matches between CNN outputs and fully connected layer
├── ddqn_agent_cnn.py # DDQN agent using a CNN
├── hyps # Directory for hyperparameter files
│ ├── h3_cnn.hyp # Hyperparameters set for experiment h3
│ └── h4_cnn.hyp # Hyperparameters set for experiment h4
├── play_human.py # Script to play against the AI
├── play_two_models.py # Script for two models to play against each other
├── replay_buffer.py # Replay buffer implementation for training
├── runs # Training runs for different experiments
│ ├── h3_cnn_connect_four_experiment
│ └── h4_cnn_connect_four_experiment
├── test_reward_functions.py # Script to test different reward functions
├── train_c4.py # Main training script
├── two_player_env.py # Two-player Connect Four environment
└── wts # Weights and saved states for trained models
├── eh_agent1_h19.wts
├── model1_h3_cnn.wts
├── model1_h4_cnn.wts
├── model2_h3_cnn.wts
├── model2_h4_cnn.wts
├── replay_buffer_h3_cnn.pkl
└── replay_buffer_h4_cnn.pkl

## Usage
To train the Connect Four AI, run the following command:

python train_c4.py <hyperparameters_file> [agent1_weights] [agent2_weights] [replay_buffer]


Where `<hyperparameters_file>` is the path to a hyperparameter configuration file. Optionally, you can specify paths to the weights for agent1, agent2, and the replay buffer to continue training from a previous state.

### Example
python train_c4.py ./hyps/h3_cnn.hyp ./wts/model1_h3_cnn.wts ./wts/model2_h3_cnn.wts ./wts/replay_buffer_h3_cnn.pkl


## Playing Against the AI
To play against the trained AI:
python play_human.py model_to_play.wts


## Requirements
- Python 3.8+
- PyTorch 1.7+
- Numpy

## Installation
Clone the repository and install the required packages:
git clone <repository-url>
cd connect_four
pip install -r requirements.txt

## Contributing
Contributions to this project are welcome! Please submit a pull request or open an issue for bugs, features, or other concerns.

## License
This project is open-sourced under the MIT License.

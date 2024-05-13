# Connect Four AI
This project implements a Connect Four game with a deep reinforcement learning agent using a double deep Q-network (DDQN). The agent is trained with convolutional neural networks to evaluate the game state and make decisions.

## Directory Structure
connect_four/
├── hyps # Directory for hyperparameter files
├── runs # Training runs for different experiments
├── wts # Modle weights



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

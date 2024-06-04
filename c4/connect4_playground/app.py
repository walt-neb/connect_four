from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
import os
from model.ddqn_agent_cnn import CNNDDQNAgent
from model.two_player_env import TwoPlayerConnectFourEnv


app = Flask(__name__)

# Global hyperparameter dictionary
hyperparams = {}

def load_hyperparams(hyp_file):
    params = {}
    hyp_file = os.path.realpath(hyp_file)
    with open(hyp_file, 'r') as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                # Parse line
                var_name, var_value = line.split("=")
                var_name = var_name.strip()  # remove leading/trailing white spaces
                var_value = var_value.strip()

                # Attempt to convert variable value to int, float, or leave as string
                try:
                    var_value = int(var_value)
                except ValueError:
                    try:
                        var_value = float(var_value)
                    except ValueError:
                        # If it's neither int nor float, keep as string
                        pass

                # Add the variable to the params dictionary
                params[var_name] = var_value
    return params


# # Load initial hyperparameters from the file
# def load_hyperparams(hyp_file="model/h4_cnn.hyp"):
#     global hyperparams  # Access the global hyperparameters dictionary
#     hyperparams = {}  # Initialize if it doesn't exist
    
#     try:
#         with open(hyp_file, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if "=" in line and not line.startswith("#"):  # Exclude comment lines
#                     key, value = line.split("=")
#                     key = key.strip()
#                     value = value.strip()
#                     try:
#                         # Convert values to appropriate types
#                         if value.lower() == 'true':
#                             value = True
#                         elif value.lower() == 'false':
#                             value = False
#                         elif '.' in value:  # Check for floats
#                             value = float(value)
#                         elif value.isdigit():  # Check for integers
#                             value = int(value)
#                         elif value.startswith("[") and value.endswith("]"):  # Check for lists
#                             value = eval(value)  # Evaluate as a Python list
#                         hyperparams[key] = value
#                     except ValueError:
#                         print(f"Warning: Could not parse value for '{key}': {value}")
#                         hyperparams[key] = value  # Keep as string if not parsable

#         # Additional defaults if not in the file
#         hyperparams.setdefault('agent1_learning_rate', 0.00025)
#         hyperparams.setdefault('agent2_learning_rate', 0.00025)
#         hyperparams.setdefault('a1_epsilon_start', 0.91)
#         hyperparams.setdefault('a1_epsilon_end', 0.01)
#         hyperparams.setdefault('a2_epsilon_start', 0.91)
#         hyperparams.setdefault('a2_epsilon_end', 0.01)
#         # ... add other hyperparameter defaults as needed

#     except FileNotFoundError:
#         print(f"Error: Hyperparameter file '{hyp_file}' not found.")
#         return None  # Return None to signal an error
#     except Exception as e:
#         print(f"Error loading hyperparameters: {e}")
#         return None

#     return hyperparams 

# Route to update hyperparameters
@app.route('/update_hyperparams', methods=['POST'])
def update_hyperparams():
    global hyperparams
    for key, value in request.json.items():
        try:
            hyperparams[key] = float(value)  # Convert to float if possible
        except ValueError:
            pass  # Keep as string if not a number
    return jsonify(success=True)

# Load your trained models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_channels = 1
input_height = 6
input_width = 7
output_dim = 7

# Load hyperparameters from h4_cnn.hyp
params = load_hyperparams("./model/h4_cnn.hyp")

cnn_a1 = list(eval(params["cnn_a1"].strip('[]')))
cnn_a2 = list(eval(params["cnn_a2"].strip('[]')))
fc_a1 = list(eval(params["fc_a1"].strip('[]')))
fc_a2 = list(eval(params["fc_a2"].strip('[]')))

agent1 = CNNDDQNAgent(input_channels, input_height, input_width, output_dim, cnn_a1, fc_a1).to(device)
agent2 = CNNDDQNAgent(input_channels, input_height, input_width, output_dim, cnn_a2, fc_a2).to(device)

agent1.load_state_dict(torch.load("model/wts/model1_h4_cnn.wts", map_location=device))
agent2.load_state_dict(torch.load("model/wts/model2_h4_cnn.wts", map_location=device))

env = TwoPlayerConnectFourEnv()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_ai_move', methods=['POST'])
def get_ai_move():
    board_state = request.json['board']
    player = request.json['player']
    epsilon = request.json['epsilon']  # Get epsilon from the front-end

    # Reshape board and convert to tensor
    board = np.array(board_state).reshape(6, 7)
    state_tensor = torch.from_numpy(board).float().unsqueeze(0).unsqueeze(0).to(device)

    if player == 1:
        action = agent1.select_action(state_tensor, env.get_valid_actions(), epsilon)
    else:
        action = agent2.select_action(state_tensor, env.get_valid_actions(), epsilon)

    return jsonify({'column': action})

app = Flask(__name__)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)

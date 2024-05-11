import os
import sys
import time
from typing import List, Tuple, Optional





def explore_network_configurations(input_height: int, input_width: int,
                                   conv_layers: List[Tuple[int, int, int, int]],
                                   fc_layers: List[int]) -> Tuple[bool, List[int]]:
    """
    Explore and validate network configurations for CNN and FC layers.

    :param input_height: The height of the input layer.
    :param input_width: The width of the input layer.
    :param conv_layers: A list of tuples for each conv layer (num_filters, kernel_size, stride, padding).
    :param fc_layers: A list of neuron counts for each fully connected layer.
    :return: A tuple indicating if the configuration is valid and the corrected first FC layer size if needed.
    """
    current_height = input_height
    current_width = input_width

    for num_filters, kernel_size, stride, padding in conv_layers:
        current_height = ((current_height + 2 * padding - kernel_size) // stride) + 1
        current_width = ((current_width + 2 * padding - kernel_size) // stride) + 1

        if current_height <= 0 or current_width <= 0:
            print("Invalid configuration: Non-positive output dimension found.")
            return False, []

    # Calculate the total number of outputs from the final convolutional layer
    output_features = num_filters * current_height * current_width
    print(f"Output features from last convolutional layer: {output_features}")

    # Check if the first FC layer size matches the output features of the last conv layer
    if fc_layers and fc_layers[0] != output_features:
        print(f"*** Mismatch found: Adjust the FC layer size from {fc_layers[0]} to {output_features}")
        fc_layers[0] = output_features
    else:
        print("FC layer size matches the output features of the last convolutional layer.")

    return True, fc_layers



def load_hyperparams(hyp_file):
    params = {}
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

def print_parameters(params):
    if not params:
        print("The parameters dictionary is empty.")
        return

    print("*** Training Parameters: ")
    for key, value in params.items():
        print(f"\t\t{key} = {value}")

def main():

    if len(sys.argv) < 2:
        print("Usage: python cnn_fc_input_check.py <hyperparameters_file>")
        return
    
    hyp_file = sys.argv[1]
    hyp_file_root = hyp_file.rstrip('.hyp')
    hyp_file_root = os.path.basename(hyp_file_root)
    print(f'hyp_file_root: {hyp_file_root}')

    params = load_hyperparams(hyp_file)
    #print_parameters(params)

    try:
        cnn_a1 = list(eval(params["cnn_a1"].strip('[]')))
        cnn_a2 = list(eval(params["cnn_a2"].strip('[]')))  
        fc_a1 = list(eval(params["fc_a1"].strip('[]')))
        fc_a2 = list(eval(params["fc_a2"].strip('[]')))                    
        render_games = eval(params["render_game_at"].strip('[]'))
    except Exception as e:
        print(e)
        sys.exit(1)

    # Example of using this function
    input_height = 6
    input_width = 7
    print("Input dimensions: ", input_height, input_width)
    print("CNN_model1: ", cnn_a1)
    print("FC_model1: ", fc_a1)
    print("CNN_model2: ", cnn_a2)
    print("FC_model2: ", fc_a2)

    #conv_layers = [(16, 3, 1, 1), (32, 3, 1, 1), (64, 3, 1, 1)]  # num_filters, kernel_size, stride, padding
    #fc_layers = [512, 128, 64]

    print("Checking network configurations for model1...")
    valid, adjusted_fc_layers = explore_network_configurations(input_height, input_width, cnn_a1, fc_a1)
    if valid:
        print("Valid configuration with FC layers:", adjusted_fc_layers)
    else:
        print("Invalid configuration. Please adjust the parameters.")

    print("Checking network configurations for model2...")
    valid, adjusted_fc_layers = explore_network_configurations(input_height, input_width, cnn_a2, fc_a2)
    if valid:
        print("Valid configuration with FC layers:", adjusted_fc_layers)
    else:
        print("Invalid configuration. Please adjust the parameters.")        




if __name__ == '__main__':
    main()



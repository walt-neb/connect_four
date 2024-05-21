

#filename: check_cnn_fc_match.py

import os
import sys
import time
from typing import List, Tuple, Optional
import ast

import torch.nn as nn

def calculate_total_parameters(cnn_layers, fc_layers, input_channels=1):
    total_params = 0
    current_channels = input_channels

    # Calculate parameters for CNN layers
    for (out_channels, kernel_size, stride, padding) in cnn_layers:
        # Number of parameters in a conv layer = (input_channels * kernel_size * kernel_size + 1) * out_channels
        conv_params = (current_channels * kernel_size * kernel_size + 1) * out_channels
        total_params += conv_params
        current_channels = out_channels  # Update the channel count for the next layer

    # The output of the last CNN layer needs to be calculated to know the input to the first FC layer
    # Assuming the input size is known, let's calculate the feature map size after the last CNN layer:
    # Example initial feature map size (common for small images or patches)
    feature_map_size = 32  # Adjust based on your actual input feature map dimensions
    for (out_channels, kernel_size, stride, padding) in cnn_layers:
        feature_map_size = ((feature_map_size + 2 * padding - kernel_size) // stride) + 1

    # Total number of outputs from the final convolutional layer
    num_flattened_features = current_channels * feature_map_size * feature_map_size

    # Assuming the first element of fc_layers is correctly set to match the flattened CNN output
    # If not, you can uncomment the next line to adjust it dynamically
    # fc_layers[0] = num_flattened_features

    # Calculate parameters for FC layers
    current_input_features = fc_layers[0]
    for output_features in fc_layers[1:]:
        # Number of parameters in an FC layer = (input_features + 1) * output_features (including bias)
        fc_params = (current_input_features + 1) * output_features
        total_params += fc_params
        current_input_features = output_features  # Update for the next layer

    # Output layer parameters
    if fc_layers:
        output_layer_params = (current_input_features + 1) * fc_layers[-1]
        total_params += output_layer_params

    return total_params


def explore_network_configurations(seq_num: int, input_height: int, input_width: int,
                                   conv_layers: List[Tuple[int, int, int, Optional[int]]],
                                   fc_layers: List[int]) -> Tuple[bool, List[int]]:
    current_height = input_height
    current_width = input_width
    seq_num = seq_num

    for params in conv_layers:
        num_filters, kernel_size, stride = params[:3]
        padding = params[3] if len(params) > 3 else 0  # Assume padding is 0 if not specified

        current_height = ((current_height + 2 * padding - kernel_size) // stride) + 1
        current_width = ((current_width + 2 * padding - kernel_size) // stride) + 1

        if current_height <= 0 or current_width <= 0:
            print("Invalid configuration: Non-positive output dimension found.")
            return False, []

    output_features = num_filters * current_height * current_width
    if fc_layers and fc_layers[0] != output_features:
        print(f"\nMismatch found: Adjust the first FC layer size from {fc_layers[0]} to {output_features}")
        print(f'\noutput_features = num_filters * current_height * current_width = {num_filters} * {current_height} * {current_width} = {output_features}')
        fc_layers[0] = output_features
    else: 
        print(f"\nFC layer sizes are compatible with the output features of the CNN layers.")

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


def ensure_list_of_tuples(data):
    if isinstance(data, tuple):
        return [data]
    return data

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
        cnn_a1 = ast.literal_eval(params["cnn_a1"])
        cnn_a2 = ast.literal_eval(params["cnn_a2"])
        fc_a1 = ast.literal_eval(params["fc_a1"])
        fc_a2 = ast.literal_eval(params["fc_a2"])
        seq_num = params["sequence_length"]

    except Exception as e:
        print(e)
        sys.exit(1)

    cnn_a1 = ensure_list_of_tuples(cnn_a1)
    cnn_a2 = ensure_list_of_tuples(cnn_a2)


    input_height = 6
    input_width = 7
    print("Input dimensions: ", input_height, input_width)
    print("CNN_model1: ", cnn_a1)
    print("FC_model1: ", fc_a1)
    print("CNN_model2: ", cnn_a2)
    print("FC_model2: ", fc_a2)

 

    print("Checking network configurations for model1...")
    valid, adjusted_fc_layers = explore_network_configurations(seq_num, input_height, input_width, cnn_a1, fc_a1)
    if valid:
        print("Valid configuration with FC layers:", adjusted_fc_layers)
    else:
        print("Invalid configuration. Please adjust the parameters.")
    total_parameters = calculate_total_parameters(cnn_a1, fc_a1)
    print("Total parameters in the network 1:", total_parameters)        

    print("Checking network configurations for model2...")
    valid, adjusted_fc_layers = explore_network_configurations(seq_num, input_height, input_width, cnn_a2, fc_a2)
    if valid:
        print("Valid configuration with FC layers:", adjusted_fc_layers)
    else:
        print("Invalid configuration. Please adjust the parameters.")    
    total_parameters = calculate_total_parameters(cnn_a1, fc_a1)
    print("Total parameters in the network 2:", total_parameters)            




if __name__ == '__main__':
    main()



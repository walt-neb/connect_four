

#filename: check_cnn_lstm_fc_match.py

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


def explore_network_configurations(
    input_height: int,
    input_width: int,
    conv_layers: List[Tuple[int, int, int, Optional[int]]],
    lstm_layers: List[Tuple[int, int]],
    fc_layers: List[int],
    cnn_to_lstm_fc_size: int,  # New: Size of the FC layer between CNN and LSTM
) -> Tuple[bool, List[int]]:

    # Calculate output features after CNN
    current_channels = 1 # Number of input channels (assuming single-channel board state)
    current_height = input_height
    current_width = input_width

    for (out_channels, kernel_size, stride, padding) in conv_layers:
        current_height = ((current_height + 2 * padding - kernel_size) // stride) + 1
        current_width = ((current_width + 2 * padding - kernel_size) // stride) + 1
        current_channels = out_channels
    output_features = current_channels * current_height * current_width

    # FC layer validation (CNN to LSTM)
    if output_features != cnn_to_lstm_fc_size:
        print(f"\nError: FC layer (CNN to LSTM) output size ({cnn_to_lstm_fc_size}) "
              f"does not match CNN output size ({output_features}).\n"
              f"Options:\n"
              f"   1. Adjust 'cnn_to_lstm_fc_size' in the hyperparameter file to {output_features}.\n"
              f"   2. Change CNN kernel size, stride, or padding to produce an output of {cnn_to_lstm_fc_size} after flattening.\n"
              f"   3. Introduce another FC layer to resize from {output_features} to {cnn_to_lstm_fc_size}.") 
        return False, []


    # LSTM layer validation
    for i, (num_lstm_layers, lstm_hidden_size) in enumerate(lstm_layers):
        if i == 0 and cnn_to_lstm_fc_size != lstm_hidden_size:
            print(f"\nError: LSTM layer {i + 1} input size ({lstm_hidden_size}) "
                  f"does not match FC layer (CNN to LSTM) output size ({cnn_to_lstm_fc_size}).\n"
                  f"Options:\n"
                  f"   1. Adjust the first LSTM layer's hidden size in 'lstm_a{1 or 2}' to {cnn_to_lstm_fc_size}.\n"
                  f"   2. Change 'cnn_to_lstm_fc_size' to {lstm_hidden_size}.")  # Added this option
        elif i > 0 and lstm_layers[i - 1][1] != lstm_hidden_size:
            print(f"\nError: LSTM layer {i + 1} input size ({lstm_hidden_size}) "
                  f"does not match previous LSTM layer's hidden size ({lstm_layers[i - 1][1]}).\n"
                  f"Options:\n"
                  f"   1. Adjust LSTM layer {i + 1}'s hidden size to {lstm_layers[i - 1][1]}.\n"
                  f"   2. Adjust the previous LSTM layer's hidden size to {lstm_hidden_size}.")

        output_features = lstm_hidden_size  # Update for the next layer

    # FC layer validation (LSTM to output)
    if fc_layers and fc_layers[0] != output_features:
        print(f"\nError: First FC layer input size ({fc_layers[0]}) "
              f"does not match the last LSTM layer's hidden size ({output_features}).\n"
              f"Options:\n"
              f"   1. Adjust the first FC layer's input size to {output_features}.\n"
              f"   2. Adjust the last LSTM layer's hidden size to {fc_layers[0]}.")

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
    
    print("\n\nChecking CNN-LSTM-FC network configurations...\n")


    hyp_file = sys.argv[1]
    hyp_file_root = hyp_file.rstrip('.hyp')
    hyp_file_root = os.path.basename(hyp_file_root)
    print(f'hyp_file_root: {hyp_file_root}')

    params = load_hyperparams(hyp_file)
    #print_parameters(params)

    try:
        cnn_a1 = ast.literal_eval(params["cnn_a1"])
        cnn_a2 = ast.literal_eval(params["cnn_a2"])
        lstm_a1 = ast.literal_eval(params["lstm_a1"])
        lstm_a2 = ast.literal_eval(params["lstm_a2"])
        fc_a1 = ast.literal_eval(params["fc_a1"])
        fc_a2 = ast.literal_eval(params["fc_a2"])
        cnn_to_lstm_fc_size = params["cnn_to_lstm_fc_size"]
    except Exception as e:
        print(e)
        sys.exit(1)

    cnn_a1 = ensure_list_of_tuples(cnn_a1)
    cnn_a2 = ensure_list_of_tuples(cnn_a2)


    input_height = 6
    input_width = 7
    #print using f-string
    print(f"Input dimensions: {input_height} x {input_width}")
    print(f"CNN_model1: \t{cnn_a1}")
    print(f"LSTM_model1:\t{lstm_a1}")
    print(f"FC_model1:  \t{fc_a1}")
    print(f"CNN_model2: \t{cnn_a2}")
    print(f"LSTM_model2:\t{lstm_a2}")
    print(f"FC_model2:  \t{fc_a2}")
    print(f"cnn_to_lstm_fc_size: {cnn_to_lstm_fc_size}")


    print("\nChecking network configurations for model1...")
    valid, adjusted_fc_layers = explore_network_configurations(input_height, input_width, cnn_a1, lstm_a1, fc_a1, cnn_to_lstm_fc_size)    
    if valid:
        print("Valid configuration with FC layers:", adjusted_fc_layers)
    else:
        print("***Invalid configuration. Please adjust the parameters.")
    total_parameters = calculate_total_parameters(cnn_a1, fc_a1)
    print("Total parameters in the network 1:", total_parameters)        

    print("\nChecking network configurations for model2...")

    valid, adjusted_fc_layers = explore_network_configurations(input_height, input_width, cnn_a2, lstm_a2, fc_a2, cnn_to_lstm_fc_size)    
    if valid:
        print("Valid configuration with FC layers:", adjusted_fc_layers)
    else:
        print("***Invalid configuration. Please adjust the parameters.")    
    total_parameters = calculate_total_parameters(cnn_a1, fc_a1)
    print("Total parameters in the network 2:", total_parameters)            




if __name__ == '__main__':
    main()



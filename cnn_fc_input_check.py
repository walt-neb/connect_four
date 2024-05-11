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
        print(f"Mismatch found: Adjusting the first FC layer size from {fc_layers[0]} to {output_features}")
        fc_layers[0] = output_features

    return True, fc_layers

# Example of using this function
input_height = 6
input_width = 7
conv_layers = [(16, 3, 1, 1), (32, 3, 1, 1), (64, 3, 1, 1)]  # num_filters, kernel_size, stride, padding
fc_layers = [512, 128, 64]

valid, adjusted_fc_layers = explore_network_configurations(input_height, input_width, conv_layers, fc_layers)
if valid:
    print("Valid configuration with FC layers:", adjusted_fc_layers)
else:
    print("Invalid configuration. Please adjust the parameters.")

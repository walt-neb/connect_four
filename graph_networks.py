import torch
import torch.nn as nn
from graphviz import Digraph

class SimpleCNN(nn.Module):
    def __init__(self, conv_layers, fc_layers):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, conv_layers[0][0], kernel_size=conv_layers[0][1], stride=conv_layers[0][2], padding=conv_layers[0][3]),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),  # Example of adding pooling to reduce spatial dimensions
        )
        # Fully connected layers
        self.fc_layers = nn.Sequential()
        input_features = 2688  # Adjust based on the output of your last conv layer
        for num_features in fc_layers:
            self.fc_layers.add_module('linear', nn.Linear(input_features, num_features))
            self.fc_layers.add_module('relu', nn.ReLU())
            input_features = num_features



def visualize_model(model):
    dot = Digraph(comment='Neural Network')

    previous_shape = 'input'
    dot.node(previous_shape, 'Input Layer')

    # Visualize convolutional layers
    for i, layer in enumerate(model.conv_layers):
        layer_str = str(layer)
        shape = 'conv' + str(i)
        label = layer_str.split('(')[0] + '\n' + layer_str.split('(')[1].split(')')[0]
        dot.node(shape, label=label)
        dot.edge(previous_shape, shape)
        previous_shape = shape

    # Add transition node
    transition_shape = 'flatten'
    dot.node(transition_shape, 'Flatten/Reshape')
    dot.edge(previous_shape, transition_shape)
    previous_shape = transition_shape

    # Visualize fully connected layers
    for i, layer in enumerate(model.fc_layers):
        if 'Linear' in str(layer):
            layer_str = str(layer)
            shape = 'fc' + str(i)
            label = layer_str.split('(')[0] + '\n' + layer_str.split('(')[1].split(')')[0]
            dot.node(shape, label=label)
            dot.edge(previous_shape, shape)
            previous_shape = shape

    dot.render('network_visualization.gv', view=True)

# Example usage:
cnn_a1 = [(16, 3, 1, 1)]
fc_a1 = [2688, 1024, 512, 256, 128]  
model_a1 = SimpleCNN(cnn_a1, fc_a1)
visualize_model(model_a1)



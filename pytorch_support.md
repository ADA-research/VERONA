## PyTorch Model Support in VERONA
This document describes how to use PyTorch models directly in VERONA, eliminating the need for ONNX conversion while maintaining backward compatibility.

### Overview
VERONA now supports both ONNX and PyTorch models through a unified interface. This allows you to:

- Use PyTorch models directly without ONNX conversion. 
- Maintain existing ONNX workflows. 
- Mix both model types in the same experiment
- Leverage PyTorch-specific features for attacks and verification.


## Networks CSV File
Create a networks.csv file in your /networks/ directory with the following structure:

``name,network_type,architecture,weights``

``mnist-net_256x2,onnx,mnist-net_256x2.onnx,``

``resnet18_pytorch,pytorch,resnet18.py,resnet18_weights.pt``

``vgg16_pytorch,pytorch,vgg16.py,vgg16_weights.pth``

An example is provided at /tests/test_experiment/networks/networks.csv


## Required Fields
network_type: Either "onnx" or "pytorch"
architecture: Path to Python file containing model definition (required for PyTorch networks) or to .onnx path
weights: Path to weights file (required for PyTorch networks)

## PyTorch Model Architecture Files
Your PyTorch model architecture file should contain a model class that inherits from torch.nn.Module. 
Furthermore, the file should contain some information about the required shape of the input. 
The file should either:

Define a model instance at module level:

``import torch.nn as nn

EXPECTED_INPUT_SHAPE = ..

class ResNet18(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()

        # ... model definition ...

    

    def forward(self, x):

        # ... forward pass ...

        return x``

# Create model instance for easy loading

resnet18_model = ResNet18(num_classes=10)

Define a function that creates and returns the model:

``import torch.nn as nn

def create_resnet18(num_classes=10):

    model = ResNet18(num_classes=num_classes)

    return model

class ResNet18(nn.Module):

    # ... model definition ... ``


## Weights Files
Store your trained model weights in .pt or .pth files. These should contain the state dictionary of your PyTorch model.

# Both ONNX and PyTorch networks implement the same interface

``network: Network = experiment_repository.get_network_list()[0]``

# Load PyTorch model (works for both types)

``pytorch_model = network.load_pytorch_model()``

# Get network name

``network_name = network.name``



## Experiment Repository
The ExperimentRepository.get_network_list() method now returns a list of Network objects instead of just ONNXNetwork objects.


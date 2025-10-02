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


## Required Fields
network_type: Either "onnx" or "pytorch"
architecture: Path to Python file containing model definition (required for PyTorch networks) or to .onnx path
weights: Path to weights file (required for PyTorch networks)

## PyTorch Model Architecture Files
Your PyTorch model architecture file should contain a model class that inherits from torch.nn.Module. 
Furthermore, the file should contain some information about the required shape of the input. 
The file should either define a model instance at module level:

```python
import torch.nn as nn

EXPECTED_INPUT_SHAPE = ..

class ResNet18(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()

        # ... model definition ...

    

    def forward(self, x):

        # ... forward pass ...

        return x
```

Instead of the global variable `EXPECTED_INPUT_SHAPE` you could also add the following function:

```python
def get_input_shape()->list[int]:
    return ...
```

# Create model instance for easy loading
```python
resnet18_model = ResNet18(num_classes=10)
```

Define a function that creates and returns the model:

```python
import torch.nn as nn

def create_resnet18(num_classes=10):

    model = ResNet18(num_classes=num_classes)

    return model

class ResNet18(nn.Module):

    # ... model definition ...
```


## Weights Files
Store your trained model weights in .pt or .pth files. These should contain the state dictionary of your PyTorch model.

# Both ONNX and PyTorch networks implement the same interface


In the normal pipeline of VERONA, the networks are created within the ExperimentRepository like so:

```python
networks = experiment_repository.get_network_list()
```

However it is also possible to instantiate your own networks: 

```python
from .database.machine_learning_model.network import Network


#create onnx
 net_info = {
        "network_type": "onnx", 
        "architecture": "some/path/to/network.onnx"
 }
 network = Network.from_file(net_info) 

#create pytorch
 net_info = {
        "network_type": "pytorch", 
        "architecture": "some/path/to/network/architecture.py", 
        "weights: "some/path/to/network/weights.pth or .pt"
 }
 network = Network.from_file(net_info) 

```

Or if you want to create a network directly for a specific type, for example onnx. 

```python
from .database.machine_learning_model.onnx_network import OnnxNetwork

network = OnnxNetwork("./path/to/model.onnx")
```




# Load PyTorch model (works for both types)

``pytorch_model = network.load_pytorch_model()``

# Get network name

``network_name = network.name``



## Experiment Repository
The ExperimentRepository.get_network_list() method now returns a list of Network objects instead of just ONNXNetwork objects.


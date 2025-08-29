# PyTorch Model Support in VERONA

This document describes how to use PyTorch models directly in VERONA, eliminating the need for ONNX conversion while maintaining backward compatibility.

## Overview

VERONA now supports both ONNX and PyTorch models through a unified interface. This allows you to:

- Use PyTorch models directly without ONNX conversion
- Maintain existing ONNX workflows
- Mix both model types in the same experiment
- Leverage PyTorch-specific features for attacks and verification

## Configuration

### Networks CSV File

Create a `networks.csv` file in your `/networks/` directory with the following structure:

```csv
name,type,network_path,architecture,weights
mnist-net_256x2,onnx,mnist-net_256x2.onnx,,
resnet18_pytorch,pytorch,,resnet18.py,resnet18_weights.pt
vgg16_pytorch,pytorch,,vgg16.py,vgg16_weights.pth
```

#### Required Fields

- **name**: Unique identifier for the network
- **type**: Either "onnx" or "pytorch"
- **network_path**: Path to ONNX file (required for ONNX networks)
- **architecture**: Path to Python file containing model definition (required for PyTorch networks)
- **weights**: Path to weights file (required for PyTorch networks)

### PyTorch Model Architecture Files

Your PyTorch model architecture file should contain a model class that inherits from `torch.nn.Module`. The file should either:

1. **Define a model instance** at module level:
```python
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # ... model definition ...
    
    def forward(self, x):
        # ... forward pass ...
        return x

# Create model instance for easy loading
resnet18_model = ResNet18(num_classes=10)
```

2. **Define a function** that creates and returns the model:
```python
import torch.nn as nn

def create_resnet18(num_classes=10):
    model = ResNet18(num_classes=num_classes)
    return model

class ResNet18(nn.Module):
    # ... model definition ...
```

### Weights Files

Store your trained model weights in `.pt` or `.pth` files. These should contain the state dictionary of your PyTorch model.

## Usage Examples

### Basic PyTorch Model Setup

1. **Create your model architecture file** (`models/resnet18.py`):
```python
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # ... ResNet18 implementation ...
    
    def forward(self, x):
        # ... forward pass ...
        return x

# Create model instance
resnet18_model = ResNet18(num_classes=10)
```

2. **Train your model and save weights**:
```python
import torch

model = ResNet18(num_classes=10)
# ... training code ...
torch.save(model.state_dict(), "models/resnet18_weights.pt")
```

3. **Add to networks.csv**:
```csv
name,type,network_path,architecture,weights
resnet18_pytorch,pytorch,,models/resnet18.py,models/resnet18_weights.pt
```

### Mixed ONNX and PyTorch Models

You can use both model types in the same experiment:

```csv
name,type,network_path,architecture,weights
mnist_onnx,onnx,mnist-net_256x2.onnx,,
resnet18_pytorch,pytorch,,models/resnet18.py,models/resnet18_weights.pt
vgg16_pytorch,pytorch,,models/vgg16.py,models/vgg16_weights.pth
```

## Backward Compatibility

If no `networks.csv` file is found, VERONA automatically falls back to the previous behavior:

- Scans the `/networks/` directory for `.onnx` files
- Creates `Network` objects for each ONNX file found
- Maintains existing workflow compatibility

## API Changes

### Network Interface

All networks now implement the abstract `Network` interface:

```python
from ada_verona.robustness_experiment_box.database.network import Network

# Both ONNX and PyTorch networks implement the same interface
network: Network = experiment_repository.get_network_list()[0]

# Load PyTorch model (works for both types)
pytorch_model = network.load_pytorch_model()

# Get network name
network_name = network.name

# Get input shape
input_shape = network.get_input_shape()
```

### Experiment Repository

The `ExperimentRepository.get_network_list()` method now returns a list of `Network` objects instead of just `ONNXNetwork` objects.

## Benefits

### For PyTorch Users

- **Direct model usage**: No need to convert models to ONNX
- **Full PyTorch features**: Access to all PyTorch layers and functionality
- **Easier debugging**: Work with familiar PyTorch code
- **Flexible architectures**: Support for complex model structures

### For Existing Users

- **Backward compatibility**: Existing ONNX workflows continue to work
- **Gradual migration**: Mix ONNX and PyTorch models as needed
- **No breaking changes**: Existing code continues to function

### For Development

- **Extensible**: Easy to add new model types in the future
- **Unified interface**: Common API for all network types
- **Better testing**: Direct access to PyTorch models for unit tests

## Error Handling

The system provides clear error messages for common issues:

- **Missing required fields**: Clear indication of which fields are missing
- **Invalid model files**: Helpful error messages for architecture loading issues
- **Graceful fallback**: Automatic fallback to directory scanning if CSV loading fails

## Testing

Run the test suite to verify PyTorch model support:

```bash
pytest tests/test_database/test_pytorch_network.py
pytest tests/test_database/test_network_factory.py
pytest tests/test_database/test_experiment_repository.py
```



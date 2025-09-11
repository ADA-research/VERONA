from pathlib import Path

import pytest
import torch

from robustness_experiment_box.database.machine_learning_method.pytorch_network import PyTorchNetwork


@pytest.fixture
def architecture_file(tmp_path):
    """Create a temporary architecture file."""
    arch_file = tmp_path / "test_model.py"
    arch_file.write_text("""
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

test_model = TestModel()
""")

    return arch_file

@pytest.fixture
def weights_file(tmp_path, architecture_file):
    """Create a temporary weights file."""
    weights_file = tmp_path / "test_weights.pt"

    # Import TestModel from the generated architecture file
    import importlib.util

    spec = importlib.util.spec_from_file_location("test_model", architecture_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.TestModel()
    torch.save(model.state_dict(), weights_file)

    return weights_file


@pytest.fixture
def pytorch_network(architecture_file, weights_file):
    """Create a PyTorchNetwork instance."""
    return PyTorchNetwork(architecture_path=architecture_file, weights_path=weights_file)


def test_pytorch_network_initialization(architecture_file, weights_file):
    """Test PyTorchNetwork initialization."""
    network = PyTorchNetwork(architecture_path=architecture_file, weights_path=weights_file)
    
    assert network.architecture_path == architecture_file
    assert network.weights_path == weights_file
    assert network.model is None
    assert network.torch_model_wrapper is None


def test_pytorch_network_name_property(weights_file):
    """Test the name property returns the weights file stem."""
    arch_file = Path("dummy_arch.py")
    network = PyTorchNetwork(architecture_path=arch_file, weights_path=weights_file)
    
    assert network.name == "test_weights"


def test_load_model_success(architecture_file, weights_file):
    """Test successful model loading."""
    network = PyTorchNetwork(architecture_path=architecture_file, weights_path=weights_file)
    
    model = network.load_model()
    
    assert isinstance(model, torch.nn.Module) 
    assert network.model is not None
    assert network.model is model


def test_load_model_cached(architecture_file, weights_file):
    """Test that model loading is cached."""
    network = PyTorchNetwork(architecture_path=architecture_file, weights_path=weights_file)
    
    model1 = network.load_model()
    model2 = network.load_model()
    
    assert model1 is model2
    assert network.model is model1


def test_load_model_architecture_not_found(tmp_path, weights_file):
    """Test error when architecture file doesn't exist."""
    non_existent_arch = tmp_path / "non_existent.py"
    network = PyTorchNetwork(architecture_path=non_existent_arch, weights_path=weights_file)
    
    with pytest.raises(FileNotFoundError):
        network.load_model()


def test_load_model_no_model_in_file(tmp_path, weights_file):
    """Test error when no model is found in architecture file."""
    arch_file = tmp_path / "empty_model.py"
    arch_file.write_text("import torch\n# No model here")
    
    network = PyTorchNetwork(architecture_path=arch_file, weights_path=weights_file)
    
    with pytest.raises(ValueError, match="No PyTorch model found"):
        network.load_model()


def test_get_input_shape():
    """Test getting input shape returns default shape."""
    arch_file = Path("dummy_arch.py")
    weights_file = Path("dummy_weights.pt")
    network = PyTorchNetwork(architecture_path=arch_file, weights_path=weights_file)
    
    input_shape = network.get_input_shape()
    
    assert input_shape.shape == (4,)
    assert input_shape[0] == 1
    assert input_shape[1] == 3
    assert input_shape[2] == 224
    assert input_shape[3] == 224


def test_load_pytorch_model(pytorch_network):
    """Test loading PyTorch model with wrapper."""
    model = pytorch_network.load_pytorch_model()
    
    assert isinstance(model, torch.nn.Module)
    assert pytorch_network.torch_model_wrapper is not None
    assert pytorch_network.torch_model_wrapper is model


def test_load_pytorch_model_cached(pytorch_network):
    """Test that PyTorch model loading is cached."""
    model1 = pytorch_network.load_pytorch_model()
    model2 = pytorch_network.load_pytorch_model()
    
    assert model1 is model2


def test_to_dict(architecture_file, weights_file):
    """Test converting to dictionary."""
    network = PyTorchNetwork(architecture_path=architecture_file, weights_path=weights_file)
    
    result = network.to_dict()
    
    assert result["architecture_path"] == str(architecture_file)
    assert result["weights_path"] == str(weights_file)
    assert result["type"] == "PyTorchNetwork"


def test_from_dict(architecture_file, weights_file):
    """Test creating from dictionary."""
    data = {
        "architecture_path": str(architecture_file),
        "weights_path": str(weights_file),
        "type": "PyTorchNetwork"
    }
    
    network = PyTorchNetwork.from_dict(data)
    
    assert network.architecture_path == architecture_file
    assert network.weights_path == weights_file


def test_weights_file_not_found(tmp_path):
    """Test behavior when weights file doesn't exist."""
    arch_file = tmp_path / "test_model.py"
    arch_file.write_text("""
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

test_model = TestModel()
    """)
    
    non_existent_weights = tmp_path / "non_existent.pt"
    network = PyTorchNetwork(architecture_path=arch_file, weights_path=non_existent_weights)
    
    # Should not raise an error, just load model without weights
    model = network.load_model()
    assert isinstance(model, torch.nn.Module)
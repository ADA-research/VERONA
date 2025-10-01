from pathlib import Path

import numpy as np
import pytest
import torch

from ada_verona.database.machine_learning_model.pytorch_network import PyTorchNetwork


def test_pytorch_network_initialization(architecture_file, weights_file):
    """Test PyTorchNetwork initialization."""
    network = PyTorchNetwork(architecture=architecture_file, weights=weights_file)
    
    assert network.architecture == architecture_file
    assert network.weights == weights_file
    assert network.model is None
    assert network.torch_model_wrapper is None


def test_pytorch_network_name_property(weights_file):
    """Test the name property returns the weights file stem."""
    arch_file = Path("dummy_arch.py")
    network = PyTorchNetwork(architecture=arch_file, weights=weights_file)
    
    assert network.name == "test_weights"


def test_load_model_success(architecture_file, weights_file):
    """Test successful model loading."""
    network = PyTorchNetwork(architecture=architecture_file, weights=weights_file)
    
    model = network.load_model()
    
    assert isinstance(model, torch.nn.Module) 
    assert network.model is not None
    assert network.model is model


def test_load_model_cached(architecture_file, weights_file):
    """Test that model loading is cached."""
    network = PyTorchNetwork(architecture=architecture_file, weights=weights_file)
    
    model1 = network.load_model()
    model2 = network.load_model()
    
    assert model1 is model2
    assert network.model is model1


def test_load_model_architecture_not_found(tmp_path, weights_file):
    """Test error when architecture file doesn't exist."""
    non_existent_arch = tmp_path / "non_existent.py"
    network = PyTorchNetwork(architecture=non_existent_arch, weights=weights_file)
    
    with pytest.raises(FileNotFoundError):
        network.load_model()


def test_load_model_no_model_in_file(tmp_path, weights_file):
    """Test error when no model is found in architecture file."""
    arch_file = tmp_path / "empty_model.py"
    arch_file.write_text("import torch\n# No model here")
    
    network = PyTorchNetwork(architecture=arch_file, weights=weights_file)
    
    with pytest.raises(ValueError, match="No PyTorch model found"):
        network.load_model()


def create_temp_arch_file(content: str, tmp_path: Path) -> Path:
    """Helper to create a temporary architecture Python file."""
    arch_file = tmp_path / "temp_model.py"
    arch_file.write_text(content)
    return arch_file

def test_get_input_shape_expected(pytorch_network):
    
    shape = pytorch_network.get_input_shape()
    
    assert np.array_equal(shape, [0,2])

def test_get_input_shape_method(pytorch_network, tmp_path):
    arch_file = tmp_path / "test_model.py"
    arch_file.write_text("""
import torch.nn as nn
import numpy as np
def get_input_shape():
    return np.array([0,2])
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

test_model = TestModel()
""")
    pytorch_network = PyTorchNetwork(arch_file, pytorch_network.weights)
    shape = pytorch_network.get_input_shape()
    assert np.array_equal(shape, np.array([0,2]))

def test_get_input_shape_missing(tmp_path):
    # architecture file without input shape info
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

    net = PyTorchNetwork(architecture=arch_file, weights=Path("temp.pth"))
    with pytest.raises(RuntimeError, match="Model architecture does not expose an input shape"):
        net.get_input_shape()


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

def test_load_model_import_error(tmp_path, weights_file):
    bad_arch = tmp_path / "bad_dir"
    bad_arch.mkdir()  # not a file, so spec loader fails

    net = PyTorchNetwork(architecture=bad_arch, weights=weights_file)
    with pytest.raises(ImportError, match="Could not load model architecture"):
        net.load_model()

def test_load_model_with_state_dict_mismatched_keys(tmp_path):
    arch_file = tmp_path / "model_mismatch.py"
    arch_file.write_text("""
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x): 
        return self.fc(x)
test_model = M()
""")

    weights_file = tmp_path / "weights.pt"
    dummy = torch.nn.Linear(5, 5) 
    torch.save({"state_dict": dummy.state_dict()}, weights_file)

    net = PyTorchNetwork(arch_file, weights_file)
    model = net.load_model()  
    assert isinstance(model, torch.nn.Module)
    
    
    
def test_get_input_shape_cached(pytorch_network):
    pytorch_network.input_shape = np.array([1, 2, 3])
    result = pytorch_network.get_input_shape()
    assert np.array_equal(result, [1, 2, 3])       
        
        
def test_to_dict(architecture_file, weights_file):
    """Test converting to dictionary."""
    network = PyTorchNetwork(architecture=architecture_file, weights=weights_file)
    
    result = network.to_dict()
    
    assert result["architecture"] == str(architecture_file)
    assert result["weights"] == str(weights_file)
    assert result["type"] == "PyTorchNetwork"


def test_from_dict(architecture_file, weights_file):
    """Test creating from dictionary."""
    data = {
        "architecture": str(architecture_file),
        "weights": str(weights_file),
        "type": "PyTorchNetwork"
    }
    
    network = PyTorchNetwork.from_dict(data)
    
    assert network.architecture == architecture_file
    assert network.weights == weights_file


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
    network = PyTorchNetwork(architecture=arch_file, weights=non_existent_weights)
    
    model = network.load_model()
    assert isinstance(model, torch.nn.Module)
    
def test_from_file_architecture_missing(tmp_path, weights_file):
    arch_file = tmp_path / "missing_arch.py"
    with pytest.raises(FileNotFoundError, match="Architecture file not found"):
        PyTorchNetwork.from_file(arch_file, weights_file)
        
def test_from_file_weights_missing(tmp_path, architecture_file):
    w_file = tmp_path / "missing_w.py"
    with pytest.raises(FileNotFoundError, match="Weights file not found"):
        PyTorchNetwork.from_file(architecture_file, w_file)
        

def test_from_file_success(architecture_file, weights_file):
    network = PyTorchNetwork.from_file(architecture_file, weights_file)
    assert isinstance(network, PyTorchNetwork)
    assert network.architecture == architecture_file
    assert network.weights == weights_file

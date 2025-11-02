import numpy as np
import pytest
import torch

from ada_verona.database.machine_learning_model.pytorch_network import PyTorchNetwork


def test_pytorch_network_initialization(mock_torch_model):
    """Test PyTorchNetwork initialization."""
    network = PyTorchNetwork(model=mock_torch_model, input_shape=[224,224], name="test_model")
    
    assert network.input_shape == [224,224]
    assert network.name == "test_model"
    assert network.model is not None
    assert network.torch_model_wrapper is None

def test_load_pytorch_model_success(mock_torch_model):
    """Test successful model loading."""
    network = PyTorchNetwork(model=mock_torch_model, input_shape=[224,224], name="test_model")
    
    model = network.load_pytorch_model()
    
    assert isinstance(model, torch.nn.Module) 
    assert network.torch_model_wrapper is not None

def test_get_input_shape_expected(pytorch_network):
    
    shape = pytorch_network.get_input_shape()
    
    assert np.array_equal(shape, [224,224])

def test_get_input_shape_cached(pytorch_network):
    pytorch_network.input_shape = np.array([1, 2, 3])
    result = pytorch_network.get_input_shape()
    assert np.array_equal(result, [1, 2, 3])       
        
def test_to_dict(pytorch_network):
    """Test converting to dictionary."""

    with pytest.raises(NotImplementedError):
        pytorch_network.to_dict()

def test_from_dict(pytorch_network):
    """Test creating from dictionary."""
  
    with pytest.raises(NotImplementedError):
        pytorch_network.from_dict(dict())

def test_properties(pytorch_network):
    assert pytorch_network.path is None
    assert pytorch_network.name == "test_model"
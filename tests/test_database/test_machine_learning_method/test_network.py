from unittest.mock import Mock

import pytest

from robustness_experiment_box.database.machine_learning_method.network import Network


class ConcreteNetwork(Network):
    """Concrete implementation of the abstract Network for testing."""
    
    def __init__(self, name: str):
        self._name = name
    
    def load_pytorch_model(self):
        return Mock()
    
    def get_input_shape(self):
        return [1, 3, 224, 224]  # Return list to test flexibility
    
    def to_dict(self):
        return {"name": self._name}
    
    @classmethod
    def from_dict(cls, data):
        return cls(data["name"])
    
    @property
    def name(self):
        return self._name


# Abstract Network Class Tests
def test_base_network_cannot_instantiate():
    """Test that Network cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Network()


def test_concrete_network_instantiation():
    """Test that concrete implementation can be instantiated."""
    network = ConcreteNetwork("test_network")
    assert network.name == "test_network"


def test_concrete_network_methods():
    """Test that concrete implementation has all required methods."""
    network = ConcreteNetwork("test_network")
    
    # Test abstract methods are implemented
    assert hasattr(network, 'load_pytorch_model')
    assert hasattr(network, 'get_input_shape')
    assert hasattr(network, 'to_dict')
    assert hasattr(network, 'from_dict')
    assert hasattr(network, 'name')
    
    # Test methods return expected types
    assert callable(network.load_pytorch_model)
    assert callable(network.get_input_shape)
    assert callable(network.to_dict)
    assert callable(network.from_dict)
    
    # Test property access
    assert isinstance(network.name, str)


def test_concrete_network_inheritance():
    """Test that concrete implementation properly inherits from Network."""
    network = ConcreteNetwork("test_network")
    
    assert isinstance(network, Network)
    assert issubclass(ConcreteNetwork, Network)


def test_concrete_network_abstract_methods():
    """Test that all abstract methods are properly implemented."""
    network = ConcreteNetwork("test_network")
    
    # These should not raise NotImplementedError
    model = network.load_pytorch_model()
    input_shape = network.get_input_shape()
    network_dict = network.to_dict()
    
    assert model is not None
    assert input_shape is not None
    assert network_dict is not None


def test_concrete_network_class_methods():
    """Test that class methods work correctly."""
    data = {"name": "test_from_dict"}
    network = ConcreteNetwork.from_dict(data)
    
    assert isinstance(network, ConcreteNetwork)
    assert network.name == "test_from_dict"


def test_concrete_network_property():
    """Test that the name property works correctly."""
    network = ConcreteNetwork("test_property")
    
    assert network.name == "test_property"
    assert isinstance(network.name, str)


def test_input_shape_flexibility():
    """Test that input_shape can return different types."""
    network = ConcreteNetwork("test_network")
    input_shape = network.get_input_shape()
    
    # Should work with both list and numpy array
    assert input_shape == [1, 3, 224, 224]
    
    # Test that it can be converted to tuple for PyTorch operations
    shape_tuple = tuple(input_shape.tolist()) if hasattr(input_shape, 'tolist') else tuple(input_shape)
    
    assert shape_tuple == (1, 3, 224, 224)


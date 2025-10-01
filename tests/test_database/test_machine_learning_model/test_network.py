
import pytest

from ada_verona.database.machine_learning_model.network import Network
from ada_verona.database.machine_learning_model.onnx_network import ONNXNetwork
from ada_verona.database.machine_learning_model.pytorch_network import PyTorchNetwork


def test_cannot_instantiate_network():
    """Ensure Network cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Network()


def test_abstract_methods_raise_notimplementederror():
    # Call the abstract methods on the class itself (unbound)
    with pytest.raises(NotImplementedError):
        Network.load_pytorch_model(Network)
        
    with pytest.raises(NotImplementedError):
        Network.get_input_shape(Network)
        
def test_from_dict_onnx(tmp_path):
    onnx_file = tmp_path / "model.onnx"
    onnx_file.touch()

    data = {
        "type": "ONNXNetwork",
        "module": "ada_verona.database.machine_learning_model.onnx_network",
        "network_path": str(onnx_file),
    }

    network = Network.from_dict(data)

    assert isinstance(network, ONNXNetwork)
    assert str(network.path) == str(onnx_file)


def test_from_dict_pytorch(tmp_path):
    arch_file = tmp_path / "model.py"
    weights_file = tmp_path / "weights.pth"
    arch_file.touch()
    weights_file.touch()

    data = {
        "type": "PyTorchNetwork",
        "module": "ada_verona.database.machine_learning_model.pytorch_network",
        "architecture": str(arch_file),
        "weights": str(weights_file),
    }

    network = Network.from_dict(data)

    assert isinstance(network, PyTorchNetwork)
    assert str(network.architecture) == str(arch_file)
    assert str(network.weights) == str(weights_file)


def test_from_dict_missing_keys(tmp_path):
    data_missing_type = {"module": "ada_verona.database.machine_learning_model.onnx_network"}
    with pytest.raises(ValueError, match="Missing 'class' or 'module' key"):
        Network.from_dict(data_missing_type)

    data_missing_module = {"type": "ONNXNetwork"}
    with pytest.raises(ValueError, match="Missing 'class' or 'module' key"):
        Network.from_dict(data_missing_module)


def test_from_dict_nonexistent_class_or_module(tmp_path):
    data_wrong_class = {
        "type": "NonExistentNetwork",
        "module": "ada_verona.database.machine_learning_model.onnx_network",
    }
    with pytest.raises(ValueError, match="Could not import NonExistentNetwork"):
        Network.from_dict(data_wrong_class)

    data_wrong_module = {"type": "ONNXNetwork", "module": "non.existent.module"}
    with pytest.raises(ValueError, match="Could not import ONNXNetwork"):
        Network.from_dict(data_wrong_module)
        
def test_from_file_onnx(tmp_path):
  
    onnx_file = tmp_path / "model.onnx"
    onnx_file.touch()

    file_dict = {"network_type": "onnx", "architecture": onnx_file}
    network = Network.from_file(file_dict)

    assert isinstance(network, ONNXNetwork)
    assert str(network.path).endswith("model.onnx")


def test_from_file_pytorch(tmp_path):
    
    arch_file = tmp_path / "model.py"
    weights_file = tmp_path / "weights.pth"
    arch_file.touch()
    weights_file.touch()

    file_dict = {
        "network_type": "pytorch",
        "architecture": arch_file,
        "weights": weights_file,
    }
    network = Network.from_file(file_dict)

    assert isinstance(network, PyTorchNetwork)
    assert str(network.architecture).endswith("model.py")
    assert str(network.weights).endswith("weights.pth")


def test_from_file_invalid_type():
    file_dict = {"network_type": "tensorflow"}
    with pytest.raises(NotImplementedError, match="Only .onnx and pytorch files are supported"):
        Network.from_file(file_dict)
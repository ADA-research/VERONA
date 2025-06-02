import onnx

from robustness_experiment_box.database.network import Network
from robustness_experiment_box.database.torch_model_wrapper import TorchModelWrapper


def test_network_initialization(network):
    assert network.path.exists()
    assert network.onnx_model is None
    assert network.torch_model_wrapper is None


def test_to_dict(network):
    network_dict = network.to_dict()

    assert network_dict == {"network_path": str(network.path)}


def test_from_dict(tmp_path):
    network_path = tmp_path / "mock_model.onnx"
    network_path.touch()
    network_dict = {"network_path": str(network_path)}

    network = Network.from_dict(network_dict)

    assert isinstance(network, Network)
    assert network.path == network_path


def test_load_onnx_model(network):

    model = onnx.helper.make_model(onnx.helper.make_graph([], "test_graph", [], []))
    onnx.save(model, str(network.path))

    loaded_model = network.load_onnx_model()

    assert isinstance(loaded_model, onnx.ModelProto)
    assert network.onnx_model == loaded_model


def test_get_input_shape(network, mock_graph):

    # Create and save the model
    model = onnx.helper.make_model(mock_graph)
    onnx.save(model, str(network.path))

    # Run the actual function
    input_shape = network.get_input_shape()
    assert input_shape == [1, 3, 224, 224]


def test_load_pytorch_model(network, mock_graph):
    # Create a minimal ONNX model for testing
     # Create and save the ONNX model
    model = onnx.helper.make_model(mock_graph)
    onnx.save(model, str(network.path))

    # Test the loading logic
    torch_model_wrapper = network.load_pytorch_model()

    assert isinstance(torch_model_wrapper, TorchModelWrapper)
    assert torch_model_wrapper.torch_model is not None
    assert torch_model_wrapper.input_shape == [1, 3, 224, 224]
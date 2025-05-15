import pytest
from pathlib import Path
import onnx
import torch
import numpy as np
from robustness_experiment_box.database.network import Network
from robustness_experiment_box.database.torch_model_wrapper import TorchModelWrapper


@pytest.fixture
def mock_network(tmp_path):
    onnx_file = tmp_path / "mock_model.onnx"
    onnx_file.touch()
    return Network(path=onnx_file)

@pytest.fixture
def mock_graph():
    # Define the input and output tensors
    input_tensor = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 3, 224, 224])

    # Create a dummy node (e.g., identity operation)
    node = onnx.helper.make_node(
        "Relu",
        inputs=["input"],
        outputs=["output"]
    )

    # Construct the graph
    graph = onnx.helper.make_graph(
        nodes=[node],
        name="test_graph",
        inputs=[input_tensor],
        outputs=[output_tensor]
    )
    return graph


def test_network_initialization(mock_network):
    assert mock_network.path.exists()
    assert mock_network.onnx_model is None
    assert mock_network.torch_model_wrapper is None


def test_to_dict(mock_network):
    network_dict = mock_network.to_dict()

    assert network_dict == {"network_path": str(mock_network.path)}


def test_from_dict(tmp_path):
    network_path = tmp_path / "mock_model.onnx"
    network_path.touch()
    network_dict = {"network_path": str(network_path)}

    network = Network.from_dict(network_dict)

    assert isinstance(network, Network)
    assert network.path == network_path


def test_load_onnx_model(mock_network):

    model = onnx.helper.make_model(onnx.helper.make_graph([], "test_graph", [], []))
    onnx.save(model, str(mock_network.path))

    loaded_model = mock_network.load_onnx_model()

    assert isinstance(loaded_model, onnx.ModelProto)
    assert mock_network.onnx_model == loaded_model


def test_get_input_shape(mock_network, mock_graph):

    # Create and save the model
    model = onnx.helper.make_model(mock_graph)
    onnx.save(model, str(mock_network.path))

    # Run the actual function
    input_shape = mock_network.get_input_shape()
    assert input_shape == [1, 3, 224, 224]


def test_load_pytorch_model(mock_network, mock_graph):
    # Create a minimal ONNX model for testing
     # Create and save the ONNX model
    model = onnx.helper.make_model(mock_graph)
    onnx.save(model, str(mock_network.path))

    # Test the loading logic
    torch_model_wrapper = mock_network.load_pytorch_model()

    assert isinstance(torch_model_wrapper, TorchModelWrapper)
    assert torch_model_wrapper.torch_model is not None
    assert torch_model_wrapper.input_shape == [1, 3, 224, 224]
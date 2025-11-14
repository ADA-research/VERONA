# Copyright 2025 ADA Reseach Group and VERONA council. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import onnx
import pytest

from ada_verona.database.machine_learning_model.onnx_network import ONNXNetwork
from ada_verona.database.machine_learning_model.torch_model_wrapper import TorchModelWrapper


def test_network_initialization(network):
    assert network.path.exists()
    assert network.onnx_model is None
    assert network.torch_model_wrapper is None


def test_network_name_property(network):
    """Test the name property returns the path stem."""
    assert network.name == "network"


def test_load_onnx_model(network):

    model = onnx.helper.make_model(onnx.helper.make_graph([], "test_graph", [], []))
    onnx.save(model, str(network.path))

    loaded_model = network.load_onnx_model()

    assert isinstance(loaded_model, onnx.ModelProto)
    assert network.onnx_model == loaded_model


def test_get_input_shape(network, mock_graph):

    model = onnx.helper.make_model(mock_graph)
    onnx.save(model, str(network.path))

    input_shape = network.get_input_shape()
    assert input_shape == [1, 3, 224, 224]

def test_load_pytorch_model(network, mock_graph):

    model = onnx.helper.make_model(mock_graph)
    onnx.save(model, str(network.path))

    torch_model_wrapper = network.load_pytorch_model()

    assert isinstance(torch_model_wrapper, TorchModelWrapper) 
    assert torch_model_wrapper.torch_model is not None
    assert torch_model_wrapper.input_shape == [1, 3, 224, 224]

def test_to_dict(network):
    network_dict = network.to_dict()
    assert network_dict == {
        "network_path": str(network.path), 
        'type':'ONNXNetwork', 
        'module': 'ada_verona.database.machine_learning_model.onnx_network'
        }


def test_from_dict(tmp_path):
    network_path = tmp_path / "mock_model.onnx"
    network_path.touch()
    network_dict = {"network_path": network_path}

    network = ONNXNetwork.from_dict(network_dict)

    assert isinstance(network, ONNXNetwork)
    assert network.path == network_path


def test_from_file(tmp_path):
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_text("fake onnx content")

    network = ONNXNetwork.from_file(onnx_file)

    assert isinstance(network, ONNXNetwork)
    assert network.path == onnx_file
    

def test_from_file_not_found(tmp_path):
    missing_file = tmp_path / "missing.onnx"

    with pytest.raises(FileNotFoundError) as exc:
        ONNXNetwork.from_file(missing_file)

    assert str(missing_file) in str(exc.value)

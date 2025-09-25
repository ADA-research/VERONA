import onnx
import pytest
import torch

from ada_verona.database.dataset.data_point import DataPoint
from ada_verona.database.epsilon_value_result import EpsilonValueResult
from ada_verona.database.experiment_repository import ExperimentRepository
from ada_verona.database.machine_learning_model.onnx_network import ONNXNetwork
from ada_verona.database.machine_learning_model.pytorch_network import PyTorchNetwork
from ada_verona.database.machine_learning_model.torch_model_wrapper import TorchModelWrapper
from ada_verona.database.verification_context import VerificationContext


class MockVerificationContext:
    """
    A mock class for VerificationContext to simulate its behavior for testing.
    """

    def get_dict_for_epsilon_result(self):
        return {"mock_key": "mock_value"}
    
    def to_dict(self):
        return {"mock_key": "mock_value"}

@pytest.fixture
def mock_verification_context():
    return MockVerificationContext()


@pytest.fixture
def experiment_repository(tmp_path):
    base_path = tmp_path / "experiments"
    base_path.mkdir()
    network_folder = base_path / "networks"
    network_folder.mkdir()
    return ExperimentRepository(base_path=base_path, network_folder=network_folder)


@pytest.fixture 
def epsilon_value_result(mock_verification_context):
    epsilon = 0.5
    smallest_sat_value = 0.3
    time_taken = 1.23

    result = EpsilonValueResult(
        verification_context=mock_verification_context,
        epsilon=epsilon,
        smallest_sat_value=smallest_sat_value,
        time=time_taken,
    )

    return result



@pytest.fixture
def network(tmp_path):
    onnx_file = tmp_path / "network.onnx"
    onnx_file.touch()
    return ONNXNetwork(path=onnx_file)

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

class MockTorchModel(torch.nn.Module):
    """
    A mock PyTorch model for testing purposes.
    """

    def forward(self, x):
        return torch.sum(x).unsqueeze(0)



@pytest.fixture
def mock_torch_model():
    return MockTorchModel()


@pytest.fixture
def torch_model_wrapper(mock_torch_model):
    # Define an input shape for the wrapper
    input_shape = (2, 3)
    return TorchModelWrapper(torch_model=mock_torch_model, input_shape=input_shape)


@pytest.fixture
def datapoint():
    return DataPoint("1", 0, torch.tensor([0.1, 0.2, 0.3]))  


@pytest.fixture
def verification_context(network, datapoint, tmp_path, property_generator):
    return VerificationContext(network, datapoint, tmp_path, property_generator)



@pytest.fixture
def architecture_file(tmp_path):
    """Create a temporary architecture file."""
    arch_file = tmp_path / "test_model.py"
    arch_file.write_text("""
import torch.nn as nn
EXPECTED_INPUT_SHAPE = [0,2]
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
    return PyTorchNetwork(architecture=architecture_file, weights=weights_file)

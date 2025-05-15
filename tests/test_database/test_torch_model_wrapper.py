import pytest
import torch
from robustness_experiment_box.database.torch_model_wrapper import TorchModelWrapper


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


def test_torch_model_wrapper_initialization(torch_model_wrapper, mock_torch_model):

    assert torch_model_wrapper.torch_model == mock_torch_model
    assert torch_model_wrapper.input_shape == (2, 3)


def test_torch_model_wrapper_forward(torch_model_wrapper):
    input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

    output = torch_model_wrapper(input_tensor)
    print("help me", output)
    assert torch.allclose(output, torch.tensor([21.0]))


def test_torch_model_wrapper_reshape(torch_model_wrapper):
    input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

    reshaped_tensor = input_tensor.reshape(torch_model_wrapper.input_shape)

    assert reshaped_tensor.shape == torch_model_wrapper.input_shape
    assert torch.equal(
        reshaped_tensor, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )
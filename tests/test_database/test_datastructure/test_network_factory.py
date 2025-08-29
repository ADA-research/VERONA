
import pandas as pd
import pytest
import re
from robustness_experiment_box.database.datastructure.network import Network
from robustness_experiment_box.database.datastructure.network_factory import NetworkFactory
from robustness_experiment_box.database.datastructure.pytorch_network import PyTorchNetwork


@pytest.fixture
def networks_dir(tmp_path):
    """Create a temporary networks directory."""
    networks_dir = tmp_path / "networks"
    networks_dir.mkdir()
    return networks_dir


@pytest.fixture
def onnx_file(networks_dir):
    """Create a temporary ONNX file."""
    onnx_file = networks_dir / "test_model.onnx"
    onnx_file.touch()
    return onnx_file


@pytest.fixture
def architecture_file(networks_dir):
    """Create a temporary architecture file."""
    arch_file = networks_dir / "test_model.py"
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
def weights_file(networks_dir):
    """Create a temporary weights file."""
    weights_file = networks_dir / "test_weights.pt"
    weights_file.touch()
    return weights_file


@pytest.fixture
def networks_csv(networks_dir):
    """Create a temporary networks CSV file."""
    csv_file = networks_dir / "networks.csv"
    data = {
        "name": ["test_onnx", "test_pytorch"],
        "type": ["onnx", "pytorch"],
        "network_path": ["test_model.onnx", ""],
        "architecture": ["", "test_model.py"],
        "weights": ["", "test_weights.pt"]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return csv_file


def test_create_network_from_csv_row_onnx(networks_dir, onnx_file):
    """Test creating an ONNX network from CSV row."""
    row = pd.Series({
        "name": "test_onnx",
        "type": "onnx",
        "network_path": "test_model.onnx"
    })
    
    network = NetworkFactory.create_network_from_csv_row(row, networks_dir)
    
    assert isinstance(network, Network)
    assert network.path == onnx_file


def test_create_network_from_csv_row_onnx_default_type(networks_dir, onnx_file):
    """Test creating an ONNX network when type is not specified."""
    row = pd.Series({
        "name": "test_onnx",
        "network_path": "test_model.onnx"
    })
    
    network = NetworkFactory.create_network_from_csv_row(row, networks_dir)
    
    assert isinstance(network, Network)
    assert network.path == onnx_file


def test_create_network_from_csv_row_pytorch(networks_dir, architecture_file, weights_file):
    """Test creating a PyTorch network from CSV row."""
    row = pd.Series({
        "name": "test_pytorch",
        "type": "pytorch",
        "architecture": "test_model.py",
        "weights": "test_weights.pt"
    })
    
    network = NetworkFactory.create_network_from_csv_row(row, networks_dir)
    
    assert isinstance(network, PyTorchNetwork)
    assert network.architecture_path == architecture_file
    assert network.weights_path == weights_file


def test_create_network_from_csv_row_onnx_missing_path(networks_dir):
    """Test error when ONNX network is missing network_path."""
    row = pd.Series({
        "name": "test_onnx",
        "type": "onnx"
    })
    
    with pytest.raises(ValueError, match="ONNX network requires 'network_path' field"):
        NetworkFactory.create_network_from_csv_row(row, networks_dir)


def test_create_network_from_csv_row_pytorch_missing_architecture(networks_dir, weights_file):
    """Test error when PyTorch network is missing architecture."""
    row = pd.Series({
        "name": "test_pytorch",
        "type": "pytorch",
        "weights": "test_weights.pt"
    })
    
    with pytest.raises(ValueError, match="PyTorch network requires both 'architecture' and 'weights' fields"):
        NetworkFactory.create_network_from_csv_row(row, networks_dir)


def test_create_network_from_csv_row_pytorch_missing_weights(networks_dir, architecture_file):
    """Test error when PyTorch network is missing weights."""
    row = pd.Series({
        "name": "test_pytorch",
        "type": "pytorch",
        "architecture": "test_model.py"
    })
    
    with pytest.raises(ValueError, match="PyTorch network requires both 'architecture' and 'weights' fields"):
        NetworkFactory.create_network_from_csv_row(row, networks_dir)


def test_create_network_from_csv_row_unsupported_type(networks_dir):
    """Test error when network type is not supported."""
    row = pd.Series({
        "name": "test_unsupported",
        "type": "unsupported_type"
    })
    
    with pytest.raises(ValueError, match="Unsupported network type: unsupported_type"):
        NetworkFactory.create_network_from_csv_row(row, networks_dir)


def test_create_networks_from_csv_success(networks_csv, networks_dir):
    """Test creating multiple networks from CSV successfully."""
    networks = NetworkFactory.create_networks_from_csv(networks_csv, networks_dir)
    
    assert len(networks) == 2
    assert isinstance(networks[0], ONNXNetwork)
    assert isinstance(networks[1], PyTorchNetwork)
    assert networks[0].name == "test_model"
    assert networks[1].name == "test_weights"


def test_create_networks_from_csv_missing_required_columns(networks_dir):
    """Test error when CSV is missing required columns."""
    csv_file = networks_dir / "networks.csv"
    data = {"type": ["onnx"]}  # Missing 'name' column
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    
    with pytest.raises(ValueError, match="Missing required columns in networks CSV"):
        NetworkFactory.create_networks_from_csv(csv_file, networks_dir)


def test_create_networks_from_csv_file_not_found(networks_dir):
    """Test error when CSV file doesn't exist."""
    non_existent_csv = networks_dir / "non_existent.csv"
    
    with pytest.raises(FileNotFoundError):
        NetworkFactory.create_networks_from_csv(non_existent_csv, networks_dir)


def test_create_networks_from_csv_empty_file(networks_dir):
    """Test error when CSV file is empty."""
    csv_file = networks_dir / "networks.csv"
    csv_file.write_text("")  # Empty file
    with pytest.raises(ValueError) as excinfo:
        NetworkFactory.create_networks_from_csv(csv_file, networks_dir)
    with pytest.raises(ValueError, match="Networks CSV file is empty"):
        NetworkFactory.create_networks_from_csv(csv_file, networks_dir)


def test_create_networks_from_csv_invalid_format(networks_dir):
    """Test error when CSV has invalid format."""
    csv_file = networks_dir / "networks.csv"
    csv_file.write_text("invalid,csv,format\nwith,wrong,delimiters")  # Invalid CSV
    
    with pytest.raises(ValueError, match=re.escape("Missing required columns in networks CSV: ['name']")):
        NetworkFactory.create_networks_from_csv(csv_file, networks_dir)


def test_create_networks_from_csv_row_error_handling(networks_dir):
    """Test error handling when individual row creation fails."""
    csv_file = networks_dir / "networks.csv"
    data = {
        "name": ["test_onnx"],
        "type": ["onnx"],
        "network_path": ["non_existent.onnx"]  # This will cause an error
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    
    with pytest.raises(ValueError, match="Error creating network from row"):
        NetworkFactory.create_networks_from_csv(csv_file, networks_dir)
        #TODO: this should cause an error, but it doesnt.


def test_create_networks_from_directory(networks_dir, onnx_file):
    """Test creating networks from directory scanning."""
    # Create another ONNX file
    onnx_file2 = networks_dir / "test_model2.onnx"
    onnx_file2.touch()
    
    # Create a non-ONNX file (should be ignored)
    txt_file = networks_dir / "readme.txt"
    txt_file.write_text("This should be ignored")
    
    networks = NetworkFactory.create_networks_from_directory(networks_dir)
    
    assert len(networks) == 2
    # assert all(isinstance(network, Network) for network in networks)
    assert {network.path.name for network in networks} == {"test_model.onnx", "test_model2.onnx"}


def test_create_networks_from_directory_no_onnx_files(networks_dir):
    """Test creating networks from directory with no ONNX files."""
    # Create only non-ONNX files
    txt_file = networks_dir / "readme.txt"
    txt_file.write_text("No ONNX files here")
    
    networks = NetworkFactory.create_networks_from_directory(networks_dir)
    
    assert len(networks) == 0

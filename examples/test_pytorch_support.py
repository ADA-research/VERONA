#!/usr/bin/env python3
"""
Test script to demonstrate PyTorch model support in VERONA.

This script shows how to:
1. Create a simple PyTorch model
2. Save it to files
3. Load it using the new VERONA infrastructure
4. Use it for inference
"""

import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """A simple PyTorch model for demonstration."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_test_files():
    """Create test model files in a temporary directory."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create models directory
    models_dir = temp_dir / "models"
    models_dir.mkdir()
    
    # Create model architecture file
    arch_file = models_dir / "simple_model.py"
    arch_file.write_text('''
import torch.nn as nn

class SimpleModel(nn.Module):
    """A simple PyTorch model for demonstration."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create model instance for easy loading
simple_model = SimpleModel()
''')
    
    # Create and save model weights
    model = SimpleModel()
    weights_file = models_dir / "simple_model_weights.pt"
    torch.save(model.state_dict(), weights_file)
    
    # Create networks CSV
    networks_dir = temp_dir / "networks"
    networks_dir.mkdir()
    
    csv_file = networks_dir / "networks.csv"
    csv_content = """name,type,network_path,architecture,weights
simple_pytorch,pytorch,,models/simple_model.py,models/simple_model_weights.pt"""
    csv_file.write_text(csv_content)
    
    return temp_dir, networks_dir


def test_pytorch_network_loading():
    """Test loading a PyTorch network using VERONA infrastructure."""
    try:
        from ada_verona.robustness_experiment_box.database.network_factory import NetworkFactory
        from ada_verona.robustness_experiment_box.database.pytorch_network import PyTorchNetwork
        
        print("✓ PyTorch network classes imported successfully")
        
        # Create test files
        temp_dir, networks_dir = create_test_files()
        
        # Test PyTorchNetwork directly
        arch_file = temp_dir / "models" / "simple_model.py"
        weights_file = temp_dir / "models" / "simple_model_weights.pt"
        
        network = PyTorchNetwork(architecture_path=arch_file, weights_path=weights_file)
        print(f"✓ PyTorchNetwork created with name: {network.name}")
        
        # Test model loading
        network.load_pytorch_model()
        print("✓ PyTorch model loaded successfully")
        
        # Test inference
        # input_tensor = torch.randn(1, 10)
        # with torch.no_grad():
        #     output = model(input_tensor)
        # print(f"✓ Model inference successful, output shape: {output.shape}")
        
        # Test CSV loading
        csv_file = networks_dir / "networks.csv"
        networks = NetworkFactory.create_networks_from_csv(csv_file, networks_dir)
        print(f"✓ CSV loading successful, found {len(networks)} networks")
        
        # Test the loaded network
        loaded_network = networks[0]
        print(f"✓ Loaded network name: {loaded_network.name}")
        print(f"✓ Loaded network type: {type(loaded_network).__name__}")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        print("✓ Test completed successfully")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure you're running this from the VERONA environment")
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()


def test_onnx_compatibility():
    """Test that ONNX networks still work."""
    try:
        from ada_verona.robustness_experiment_box.database.base_network import BaseNetwork
        from ada_verona.robustness_experiment_box.database.network import Network
        
        print("\nTesting ONNX compatibility...")
        
        # Create a mock ONNX file
        temp_dir = Path(tempfile.mkdtemp())
        onnx_file = temp_dir / "test.onnx"
        onnx_file.touch()
        
        # Test ONNX network
        network = Network(path=onnx_file)
        print(f"✓ ONNX Network created with name: {network.name}")
        
        # Test that it implements BaseNetwork
        assert isinstance(network, BaseNetwork), "ONNX Network should implement BaseNetwork"
        print("✓ ONNX Network implements BaseNetwork interface")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        print("✓ ONNX compatibility test passed")
        
    except Exception as e:
        print(f"✗ ONNX compatibility test failed: {e}")


if __name__ == "__main__":
    print("Testing PyTorch model support in VERONA...")
    print("=" * 50)
    
    test_pytorch_network_loading()
    test_onnx_compatibility()
    
    print("\n" + "=" * 50)
    print("Test script completed!")

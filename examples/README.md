# VERONA Examples

This directory contains examples demonstrating various VERONA features, including the new PyTorch model support.

## PyTorch Model Support Examples

### `networks.csv`
A sample networks configuration file showing how to configure both ONNX and PyTorch models:

```csv
name,type,network_path,architecture,weights
mnist-net_256x2,onnx,mnist-net_256x2.onnx,,
resnet18_pytorch,pytorch,,resnet18.py,resnet18_weights.pt
vgg16_pytorch,pytorch,,vgg16.py,vgg16_weights.pth
```

### `resnet18.py`
A sample PyTorch model architecture file demonstrating the required structure:

- Defines a `ResNet18` class inheriting from `torch.nn.Module`
- Creates a model instance at module level for easy loading
- Shows proper PyTorch model structure

### `test_pytorch_support.py`
A comprehensive test script that demonstrates:

1. **Creating PyTorch models** programmatically
2. **Saving model weights** to `.pt` files
3. **Loading models** using VERONA's new infrastructure
4. **Running inference** with loaded models
5. **CSV-based configuration** for mixed model types
6. **Backward compatibility** with existing ONNX workflows

## Running the Examples

### Prerequisites
- VERONA environment set up
- PyTorch installed
- Access to the `ada_verona` package

### Test PyTorch Support
```bash
cd examples
python test_pytorch_support.py
```

Expected output:
```
Testing PyTorch model support in VERONA...
==================================================
✓ PyTorch network classes imported successfully
✓ PyTorchNetwork created with name: simple_model_weights
✓ PyTorch model loaded successfully
✓ Model inference successful, output shape: torch.Size([1, 2])
✓ CSV loading successful, found 1 networks
✓ Loaded network name: simple_pytorch
✓ Loaded network type: PyTorchNetwork
✓ Test completed successfully

Testing ONNX compatibility...
✓ ONNX Network created with name: test
✓ ONNX Network implements BaseNetwork interface
✓ ONNX compatibility test passed

==================================================
Test script completed!
```

## File Structure

```
examples/
├── README.md              # This file
├── networks.csv           # Sample networks configuration
├── resnet18.py           # Sample PyTorch model architecture
└── test_pytorch_support.py # Comprehensive test script
```

## Customization

### Adding Your Own Models

1. **Create your model architecture file** following the pattern in `resnet18.py`
2. **Train and save your model weights** to a `.pt` or `.pth` file
3. **Add an entry to `networks.csv`** with the appropriate paths
4. **Test the setup** using the test script as a template

### Model Architecture Requirements

Your PyTorch model file must either:
- Define a model instance at module level (e.g., `model = MyModel()`)
- Define a function that creates and returns a model (e.g., `def create_model(): return MyModel()`)

The model class must inherit from `torch.nn.Module` and implement the `forward` method.

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're running from the VERONA environment
2. **File not found**: Check that all paths in `networks.csv` are correct
3. **Model loading failures**: Verify your architecture file has proper Python syntax
4. **CSV parsing errors**: Ensure the CSV format matches the expected structure

### Debug Tips

- Run the test script to verify basic functionality
- Check console output for warning messages
- Verify file paths are relative to the networks directory
- Test your architecture file independently in Python

## Next Steps

After running the examples successfully:

1. **Read the main documentation** in `PYTORCH_MODEL_SUPPORT.md`
2. **Explore the test suite** in `tests/test_database/`
3. **Integrate PyTorch models** into your VERONA experiments
4. **Contribute improvements** to the PyTorch support infrastructure

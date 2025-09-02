import torch


class TorchModelWrapper(torch.nn.Module):
    """
    A wrapper class for a PyTorch model to reshape the input before passing it to the model.
    """

    def __init__(self, torch_model: torch.nn.Module, input_shape):
        """
        Initialize the TorchModelWrapper with the given PyTorch model and input shape.

        Args:
            torch_model (torch.nn.Module): The PyTorch model to wrap.
            input_shape: The input shape to reshape the input tensor. Can be tuple[int] or np.ndarray.
        """
        super().__init__()
        self.torch_model = torch_model
        self.input_shape = input_shape

    def forward(self, x):
        """
        Forward pass of the TorchModelWrapper.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor from the wrapped PyTorch model.
        """
        # Convert input_shape to tuple for PyTorch reshape
        if hasattr(self.input_shape, 'tolist'):
            # Handle numpy arrays
            shape_tuple = tuple(self.input_shape.tolist())
        else:
            # Handle tuples and other iterables
            shape_tuple = tuple(self.input_shape)
        
        x = x.reshape(shape_tuple)
        x = self.torch_model(x)

        return x

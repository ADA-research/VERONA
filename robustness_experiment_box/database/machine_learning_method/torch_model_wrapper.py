import numpy as np
import torch


class TorchModelWrapper(torch.nn.Module):
    """
    A wrapper class for a PyTorch model to reshape the input before passing it to the model.
    """

    def __init__(self, torch_model: torch.nn.Module, input_shape: tuple[int]):
        """
        Initialize the TorchModelWrapper with the given PyTorch model and input shape.

        Args:
            torch_model (torch.nn.Module): The PyTorch model to wrap.
            input_shape (tuple[int]): The input shape to reshape the input tensor.
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
        
        if isinstance(x, np.ndarray):
        # ensure correct dtype/device if needed
            x = torch.from_numpy(x).to(
                dtype=torch.float32,
                device=next(self.torch_model.parameters()).device
            )
            x = x.reshape(*self.input_shape)  # tuple unpacking

        else:
        # Assume it's already a torch.Tensor
            x = x.reshape(*self.input_shape)

        x = self.torch_model(x)
        return x

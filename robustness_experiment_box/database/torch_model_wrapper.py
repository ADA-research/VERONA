import torch

class TorchModelWrapper(torch.nn.Module):
    def __init__(self, torch_model: torch.nn.Module, input_shape: tuple[int]):
        super(TorchModelWrapper, self).__init__()
        self.torch_model = torch_model
        self.input_shape = input_shape
    
    def forward(self, x):
        x = x.reshape(self.input_shape)
        x = self.torch_model(x)

        return x
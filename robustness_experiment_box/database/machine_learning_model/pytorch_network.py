import importlib.util
from pathlib import Path

import numpy as np
import torch

from robustness_experiment_box.database.machine_learning_model.network import Network
from robustness_experiment_box.database.machine_learning_model.torch_model_wrapper import TorchModelWrapper


class PyTorchNetwork(Network):
    """
    A class representing a PyTorch network with architecture and weights files.

    Attributes:
        architecture_path (Path): Path to the .py file containing the model architecture.
        weights (Path): Path to the .pt/.pth file containing the model weights.
        model (torch.nn.Module, optional): The loaded PyTorch model. Defaults to None.
        torch_model_wrapper (TorchModelWrapper, optional): The PyTorch model wrapper. Defaults to None.
    """

    def __init__(self, architecture: Path, weights: Path) -> None:
        """
        Initialize the PyTorchNetwork with architecture and weights paths.

        Args:
            architecture_path (Path): Path to the .py file containing the model architecture.
            weights (Path): Path to the .pt/.pth file containing the model weights.
        """
        self.architecture = architecture
        self.weights = weights
        self.model = None
        self.torch_model_wrapper = None
        self.input_shape = None

    @property
    def name(self) -> str:
        """
        Get the name of the network.

        Returns:
            str: The name of the network.
        """
        return self.weights.stem



    def _find_model(self,mod) -> torch.nn.Module | None:
        # Directly defined model instance
        model_instance = next(
            (getattr(mod, name) for name in dir(mod)
            if isinstance(getattr(mod, name), torch.nn.Module)),
            None
        )
        if model_instance:
            return model_instance

        # Callable that returns a model
        for name in dir(mod):
            if name.startswith("_"):
                continue
            attr = getattr(mod, name)
            if callable(attr):
                try:
                    candidate = attr()
                    if isinstance(candidate, torch.nn.Module):
                        return candidate
                except Exception:
                    continue
        return None
    
    def load_model(self) -> torch.nn.Module:
        """
        Load the PyTorch model from the architecture and weights files.

        Returns:
            torch.nn.Module: The loaded PyTorch model.
        """

        if self.model is not None:
            return self.model

        # Load the model architecture module
        spec = importlib.util.spec_from_file_location("torch.nn.module", self.architecture)
        if not spec or not spec.loader:
            raise ImportError(f"Could not load model architecture from {self.architecture}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Helper to locate a torch.nn.Module instance or builder
        model = self._find_model(module)
        if model is None:
            raise ValueError(f"No PyTorch model found in {self.architecture}")

        # Load the weights if available
        if self.weights and self.weights.exists():
            checkpoint = torch.load(self.weights, map_location="cpu")
            # extract state_dict if checkpoint contains extra info
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # load state dict into model
            model.load_state_dict(state_dict, strict=False)  # strict=False to tolerate extra/missing keys

        self.model = model
        return self.model

    def get_input_shape(self) -> np.ndarray:
        """
        Get the input shape of the PyTorch model.
        This is a placeholder - PyTorch models don't have fixed input shapes like ONNX models.

        Returns:
            np.ndarray: the input_shape
        """

        if self.input_shape is None:
            if self.model is None:
                self.load_model()

            # load the same module you already import for the model
            spec = importlib.util.spec_from_file_location("model_module", self.architecture)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # look for a conventionally named attribute or function
            if hasattr(module, "EXPECTED_INPUT_SHAPE"):
                self.input_shape = np.array(module.EXPECTED_INPUT_SHAPE, dtype=int)
            elif hasattr(module, "get_input_shape"):
                self.input_shape = np.array(module.get_input_shape(), dtype=int)
            else:
                raise RuntimeError(
                    "Model architecture does not expose an input shape. "
                    "Add an EXPECTED_INPUT_SHAPE variable or get_input_shape() function to the architecture file."
                )

        return self.input_shape

    def load_pytorch_model(self) -> torch.nn.Module:
        """
        Load the PyTorch model and wrap it in a TorchModelWrapper.

        Returns:
            torch.nn.Module: The wrapped PyTorch model.
        """
        if self.torch_model_wrapper is None:
            model = self.load_model()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            
            self.torch_model_wrapper = TorchModelWrapper(model, self.get_input_shape())
        
        return self.torch_model_wrapper

    def to_dict(self) -> dict:
        """
        Convert the PyTorchNetwork to a dictionary.

        Returns:
            dict: The dictionary representation of the PyTorchNetwork.
        """
        return dict(
            architecture=str(self.architecture),
            weights=str(self.weights),
            type=self.__class__.__name__,
            module=self.__class__.__module__,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "PyTorchNetwork":
        """
        Create a PyTorchNetwork from a dictionary.

        Args:
            data (dict): The dictionary containing the PyTorchNetwork attributes.

        Returns:
            PyTorchNetwork: The created PyTorchNetwork.
        """
        return cls(
            architecture=Path(data["architecture"]),
            weights=Path(data["weights"])
        )
        
    @classmethod
    def from_file(cls, architecture:Path, weights:Path)-> "PyTorchNetwork":
        """
        Create a PyTorchNetwork from a dictionary.

        Args:
            file (Path): Path at which the network is stored. 

        Returns:
            PyTorchNetwork: The created ONNXNetwork.
        """
        #TODO: what if weights is none, is that an issue?
   
        
        return cls(architecture=architecture, weights=weights)
      
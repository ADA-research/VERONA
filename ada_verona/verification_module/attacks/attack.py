from abc import ABC, abstractmethod

import torch


class Attack(ABC):
    """Abstract base class for implementing adversarial attacks on neural network models.

    This class defines the interface that all attack methods must implement.

    Methods:
        execute(model, data, target, epsilon):
            Executes the attack on the given model using the provided data and target.

        ABC (class): Abstract Base Class from the abc module."""

    @abstractmethod
    def execute(self, model: torch.nn.Module, data: torch.Tensor, target: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Executes the attack on the given model using the provided data and target.
        Args:
            model (torch.nn.Module): The neural network model to attack.
            data (torch.Tensor): The input data for the model.
            target (torch.Tensor): The target labels for the input data.
            epsilon (float): The perturbation magnitude for the attack.

        Returns:
            error
        """
        raise NotImplementedError

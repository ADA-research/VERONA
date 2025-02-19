from abc import ABC, abstractmethod
import torch

class Attack(ABC):
    @abstractmethod
    def execute(self, model: torch.nn.Module, data: torch.Tensor, target: torch.Tensor, epsilon: float) -> torch.Tensor:
        pass
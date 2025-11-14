# Copyright 2025 ADA Reseach Group and VERONA council. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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
            torch.Tensor: The perturbed data after the attack.
        """
        raise NotImplementedError("This is an abstract method and should be implemented in subclasses.")
 

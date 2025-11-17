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

import torch
from torch import Tensor, nn
from torch.nn.modules import Module

from ada_verona.verification_module.attacks.attack import Attack


class FGSMAttack(Attack):
    """
    A class to perform the Fast Gradient Sign Method (FGSM) attack.

    Methods:
        execute(model: Module, data: Tensor, target: Tensor, epsilon: float) -> Tensor:
            Executes the FGSM attack on the given model and data.
    """
    def __init__(self) -> None:
        """
        Initialize the FGSM to record the name
        Args:
            
        """
        super().__init__()
        self.name = "FGSM"

    def execute(self, model: Module, data: Tensor, target: Tensor, epsilon: float) -> Tensor:
        """
        Execute the FGSM attack on the given model and data.

        Args:
            model (Module): The model to attack.
            data (Tensor): The input data to perturb.
            target (Tensor): The target labels for the data.
            epsilon (float): The perturbation magnitude.

        Returns:
            Tensor: The perturbed data.
        """
        self.name = f"FGSMAttack (epsilon={epsilon}, target={target.item()})"
        data.requires_grad = True
        output = model(data)
        loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(output, target)

        model.zero_grad()

        loss.backward()

        data_grad = data.grad.data

        sign_data_grad = data_grad.sign()

        perturbed_image = data + epsilon * sign_data_grad
        perturbed_image = torch.clamp(
            perturbed_image, 0, 1
        )
        return perturbed_image

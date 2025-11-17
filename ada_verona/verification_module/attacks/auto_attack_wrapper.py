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

from autoattack import AutoAttack
from torch import Tensor
from torch.nn.modules import Module

from ada_verona.verification_module.attacks.attack import Attack


class AutoAttackWrapper(Attack):
    """
    A wrapper for the AutoAttack adversarial attack.
    Install using: uv pip install git+https://github.com/fra31/auto-attack

    Args:
        Attack (class): The base class for attacks.
    """

    def __init__(self, device="cuda", norm="Linf", version="standard", verbose=False) -> None:
        """
        Initialize the AutoAttackWrapper with specific parameters.

        Args:
            device (str, optional): The device to run the attack on. Defaults to 'cuda'.
            norm (str, optional): The norm to use for the attack. Defaults to 'Linf'.
            version (str, optional): The version of AutoAttack to use. Defaults to 'standard'.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        """
        super().__init__()
        self.device = device
        self.norm = norm
        self.version = version
        self.verbose = verbose
        self.name = f"AutoAttackWrapper (norm={self.norm}, version={self.version})"

    def execute(self, model: Module, data: Tensor, target: Tensor, epsilon: float) -> Tensor:
        """
        Execute the AutoAttack on the given model and data.

        Args:
            model (Module): The model to attack.
            data (Tensor): The input data to perturb.
            target (Tensor): The target labels for the data.
            epsilon (float): The perturbation magnitude.

        Returns:
            Tensor: The perturbed data.
        """
        
        adversary = AutoAttack(
            model, norm=self.norm, eps=epsilon, version=self.version, device=self.device, verbose =self.verbose
        )
        data = data.unsqueeze(0)

        # auto attack requires NCHW input format
        perturbed_data = adversary.run_standard_evaluation(data, target)
        
        # Handle the case where perturbed_data is a tuple (for autoattack: The first element is the perturbed image 
        # and second element is the model's prediction after the attack)
        if isinstance(perturbed_data, tuple):
            # Return the first element of the tuple (the perturbed images)
            return perturbed_data[0].to(self.device)
        else:
            # Original behavior for backward compatibility
            return perturbed_data.to(self.device)

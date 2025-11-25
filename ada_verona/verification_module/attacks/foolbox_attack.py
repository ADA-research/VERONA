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

import foolbox
from torch import Tensor, nn

from ada_verona.verification_module.attacks.attack import Attack


class FoolboxAttack(Attack):
    """
    A wrapper for Foolbox adversarial attacks.
    Requires foolbox to be installed: pip install foolbox

    Attributes:
        attack_cls (class): The Foolbox attack class to use.
        kwargs (dict): Arguments to pass to the attack constructor.
    """

    def __init__(self, attack_cls, bounds=(0, 1), **kwargs) -> None:
        """
        Initialize the FoolboxAttack wrapper.

        Args:
            attack_cls (class): The Foolbox attack class (e.g., foolbox.attacks.LinfPGD).
            bounds (tuple, optional): The bounds of the input data. Defaults to (0, 1).
            **kwargs: Arguments to be passed to the attack constructor (e.g., steps=40).
        """
        super().__init__()
        self.attack_cls = attack_cls
        self.bounds = bounds
        self.kwargs = kwargs
        self.name = f"FoolboxAttack ({attack_cls.__name__}, bounds={bounds}, {kwargs})"

    def execute(self, model: nn.Module, data: Tensor, target: Tensor, epsilon: float) -> Tensor:
        """
        Execute the Foolbox attack on the given model and data.

        Args:
            model (nn.Module): The model to attack.
            data (Tensor): The input data to perturb.
            target (Tensor): The target labels for the data.
            epsilon (float): The perturbation magnitude.

        Returns:
            Tensor: The perturbed data.
        """
        fmodel = foolbox.PyTorchModel(model, bounds=self.bounds)

        attack = self.attack_cls(**self.kwargs)

        if data.dim() == 3:
            data = data.unsqueeze(0)

        if target.dim() == 0:
            target = target.unsqueeze(0)
        _, clipped_advs, _ = attack(fmodel, data, target, epsilons=epsilon)

        return clipped_advs

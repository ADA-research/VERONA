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

    Both untargeted and targeted attacks are supported via the `target` parameter
    of `execute()`, which is always the correct class label:

    - Untargeted (default): foolbox's `Misclassification` criterion is used, i.e.
      the attack searches for an input no longer classified as `target`.
    - Targeted (`targeted=True` plus a `target_class`): foolbox's
      `TargetedMisclassification` criterion is used, i.e. the attack searches for
      an input classified as `target_class`.

    Targeted mode is attack-dependent: iterative attacks such as `LinfPGD`,
    `LinfBasicIterativeAttack` and `LinfDeepFoolAttack` support it, while
    single-step attacks such as `LinfFastGradientAttack` (FGSM) do not and raise
    `ValueError: unsupported criterion`. Within the One2Any robustness pipeline a
    successful targeted attack also implies the true label is no longer predicted,
    so it is scored as a (typically harder) counterexample; `target_class` should
    therefore differ from the true label, otherwise the search is degenerate.

    Attributes:
        attack_cls (class): The Foolbox attack class to use.
        kwargs (dict): Arguments to pass to the attack constructor.
    """

    def __init__(
        self, attack_cls, bounds=(0, 1), *, targeted: bool = False, target_class: int | None = None, **kwargs
    ) -> None:
        """
        Initialize the FoolboxAttack wrapper.

        Args:
            attack_cls (class): The Foolbox attack class (e.g., foolbox.attacks.LinfPGD).
            bounds (tuple, optional): The bounds of the input data. Defaults to (0, 1).
            targeted (bool, optional): If True, run the attack in targeted mode. Defaults to False.
            target_class (int, optional): The class to drive predictions toward in targeted mode.
                Required when ``targeted`` is True; ignored otherwise.
            **kwargs: Arguments to be passed to the attack constructor (e.g., steps=40).
        """
        super().__init__()
        if targeted and target_class is None:
            raise ValueError("A targeted FoolboxAttack requires `target_class` (the class to aim predictions at).")
        self.attack_cls = attack_cls
        self.bounds = bounds
        self.targeted = targeted
        self.target_class = target_class
        self.kwargs = kwargs
        mode = f"targeted->{target_class}" if targeted else "untargeted"
        self.name = f"FoolboxAttack ({attack_cls.__name__}, bounds={bounds}, {mode}, {kwargs})"

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

        # Ensure data has batch dimension (Foolbox requires batch dimension)
        # Data should be (batch_size, channels, height, width) or (batch_size, features)
        # Foolbox expects at least 2D tensors: (batch_size, ...)
        if data.dim() == 0:
            # Scalar, add batch dimension: (1,)
            data = data.unsqueeze(0)
        elif data.dim() == 1:
            # 1D tensor, add batch dimension: (1, features)
            data = data.unsqueeze(0)
        elif data.dim() == 3:
            # 3D tensor (C, H, W), add batch dimension: (1, C, H, W)
            data = data.unsqueeze(0)
        # If data is already 4D (B, C, H, W) or 2D (B, features), keep as is
        # But verify it has a batch dimension
        if data.dim() >= 2 and data.shape[0] == 0:
            raise ValueError(f"Data tensor has invalid batch size: {data.shape}")

        # Ensure target has batch dimension
        # Target should be 1D with shape (batch_size,) for a single sample: (1,)
        if target.dim() == 0:
            # Scalar target, add batch dimension
            target = target.unsqueeze(0)
        elif target.dim() == 1 and target.shape[0] == 0:
            raise ValueError("Target tensor cannot be empty")
        # If target is already correct shape, keep as is

        # Pick the foolbox criterion: aim away from the true label (untargeted) or
        # toward a chosen class (targeted). Foolbox would auto-wrap a bare label
        # tensor as Misclassification; we build it explicitly for symmetry.
        if self.targeted:
            target_classes = target.new_full(target.shape, self.target_class)
            criterion = foolbox.criteria.TargetedMisclassification(target_classes)
        else:
            criterion = foolbox.criteria.Misclassification(target)

        _, clipped_advs, _ = attack(fmodel, data, criterion, epsilons=epsilon)

        return clipped_advs

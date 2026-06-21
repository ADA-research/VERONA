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

import foolbox as fb
import pytest
import torch
from torch import nn

from ada_verona.verification_module.attacks.foolbox_attack import FoolboxAttack


def test_foolbox_attack_init_stores_configuration():
    attack = FoolboxAttack(attack_cls=fb.attacks.LinfFastGradientAttack, bounds=(-1, 1), steps=7)
    assert attack.attack_cls is fb.attacks.LinfFastGradientAttack
    assert attack.bounds == (-1, 1)
    assert attack.kwargs == {"steps": 7}
    # The name should expose the attack class, bounds and kwargs for traceability.
    assert "LinfFastGradientAttack" in attack.name
    assert "bounds=(-1, 1)" in attack.name
    assert "steps" in attack.name


def test_foolbox_attack_execute(foolbox_attack, model, data, target):
    epsilon = 0.1
    normalized_data = torch.sigmoid(data)
    perturbed_data = foolbox_attack.execute(model, normalized_data, target, epsilon)
    assert isinstance(perturbed_data, torch.Tensor)
    assert perturbed_data.shape == normalized_data.shape
    assert torch.all(perturbed_data >= 0) and torch.all(perturbed_data <= 1)


def test_foolbox_attack_execute_3d_data(foolbox_attack, target):
    epsilon = 0.1

    class FlattenModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = FlattenModel()
    data_3d = torch.randn(1, 1, 10)
    normalized_data = torch.sigmoid(data_3d)
    perturbed_data = foolbox_attack.execute(model, normalized_data, target, epsilon)
    assert isinstance(perturbed_data, torch.Tensor)
    assert perturbed_data.shape == (1, 1, 1, 10)
    assert torch.all(perturbed_data >= 0) and torch.all(perturbed_data <= 1)


def test_foolbox_attack_execute_0d_target(foolbox_attack, model, data):
    epsilon = 0.1
    target_0d = torch.tensor(1)
    normalized_data = torch.sigmoid(data)
    perturbed_data = foolbox_attack.execute(model, normalized_data, target_0d, epsilon)
    assert isinstance(perturbed_data, torch.Tensor)
    assert perturbed_data.shape == normalized_data.shape
    assert torch.all(perturbed_data >= 0) and torch.all(perturbed_data <= 1)


def test_foolbox_attack_execute_1d_data(foolbox_attack, model, target):
    epsilon = 0.1
    # 1D input (features only); execute() should add the batch dimension.
    data_1d = torch.sigmoid(torch.randn(10))
    perturbed_data = foolbox_attack.execute(model, data_1d, target, epsilon)
    assert isinstance(perturbed_data, torch.Tensor)
    assert perturbed_data.shape == (1, 10)
    assert torch.all(perturbed_data >= 0) and torch.all(perturbed_data <= 1)


def test_foolbox_attack_execute_kwargs_passed_to_attack(model, data, target):
    # steps controls the number of PGD iterations; passing it should not break execution.
    attack = FoolboxAttack(attack_cls=fb.attacks.LinfPGD, steps=3)
    perturbed_data = attack.execute(model, torch.sigmoid(data), target, 0.1)
    assert isinstance(perturbed_data, torch.Tensor)
    assert perturbed_data.shape == data.shape


def test_foolbox_attack_execute_empty_data_raises(foolbox_attack, model, target):
    # A tensor with batch size 0 is invalid and must be rejected.
    empty_data = torch.zeros(0, 10)
    with pytest.raises(ValueError):
        foolbox_attack.execute(model, empty_data, target, 0.1)


def test_foolbox_attack_execute_empty_target_raises(foolbox_attack, model, data):
    # A valid batch of data but an empty target tensor must be rejected.
    empty_target = torch.zeros(0, dtype=torch.long)
    with pytest.raises(ValueError):
        foolbox_attack.execute(model, torch.sigmoid(data), empty_target, 0.1)


def test_foolbox_attack_execute_0d_data_adds_batch_dimension(target):
    # Scalar input exercises the 0D branch, which unsqueezes to a batch dimension.
    class SingleFeatureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1, 2)

        def forward(self, x):
            return self.fc(x)

    attack = FoolboxAttack(attack_cls=fb.attacks.LinfFastGradientAttack)
    # Foolbox cannot run crossentropy on the resulting 1D logits, so a ValueError
    # is expected; the assertion guards the 0D unsqueeze branch in execute().
    with pytest.raises(ValueError):
        attack.execute(SingleFeatureModel(), torch.tensor(0.5), target, 0.1)

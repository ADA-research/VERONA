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

import pytest
import torch
from torch import nn

from ada_verona.verification_module.attacks.auto_attack_wrapper import AutoAttackWrapper
from ada_verona.verification_module.attacks.fgsm_attack import FGSMAttack
from ada_verona.verification_module.attacks.pgd_attack import PGDAttack


@pytest.fixture
def model():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    return SimpleModel()

@pytest.fixture
def data():
    return torch.randn(1, 10)

@pytest.fixture
def target():
    return torch.tensor([1])

@pytest.fixture
def attack_wrapper():
    return AutoAttackWrapper(device="cpu", norm="Linf", version="standard", verbose=False)

@pytest.fixture
def pgd_attack():
     return PGDAttack(number_iterations=10, step_size=0.01, randomise=True)

@pytest.fixture
def fgsm_attack():
    return FGSMAttack()
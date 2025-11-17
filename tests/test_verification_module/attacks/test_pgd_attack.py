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


def test_pgd_attack_initialization(pgd_attack):
    assert pgd_attack.number_iterations == 10
    assert pgd_attack.step_size == 0.01
    assert pgd_attack.randomise

def test_pgd_attack_execute(pgd_attack, model, data, target):
    epsilon = 0.1
    perturbed_data = pgd_attack.execute(model, data, target, epsilon)
    assert isinstance(perturbed_data, torch.Tensor)
    assert perturbed_data.shape == data.shape
    assert torch.all(perturbed_data >= 0) and torch.all(perturbed_data <= 1)
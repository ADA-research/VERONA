import pytest
import torch
from torch import nn

from robustness_experiment_box.verification_module.attacks.auto_attack_wrapper import AutoAttackWrapper
from robustness_experiment_box.verification_module.attacks.fgsm_attack import FGSMAttack
from robustness_experiment_box.verification_module.attacks.pgd_attack import PGDAttack


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
    return AutoAttackWrapper(device='cpu', norm='Linf', version='standard', verbose=False)

@pytest.fixture
def pgd_attack():
     return PGDAttack(number_iterations=10, step_size=0.01, randomise=True)

@pytest.fixture
def fgsm_attack():
    return FGSMAttack()
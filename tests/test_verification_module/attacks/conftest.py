import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

fake_pyautoattack = types.ModuleType("pyautoattack")

class DummyAutoAttack:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    def run_standard_evaluation(self, data, target):
        return torch.tensor([[0.1, 0.0, 2.0]])

fake_pyautoattack.AutoAttack = DummyAutoAttack

sys.modules["pyautoattack"] = fake_pyautoattack
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
def mock_attack():
    with patch("ada_verona.verification_module.attacks.auto_attack_wrapper.AutoAttack") as MockAttack:
        # create a mock instance that AutoAttack() will return
        mock_instance = MagicMock()
        mock_instance.run_standard_evaluation.return_value = torch.tensor([[0.1, 0.0, 2.0]])
        
        # make AutoAttack(...) return our mock instance
        MockAttack.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def attack_wrapper(mock_attack):
    return AutoAttackWrapper(device="cpu", norm="Linf", version="standard", verbose=False)

@pytest.fixture
def pgd_attack():
     return PGDAttack(number_iterations=10, step_size=0.01, randomise=True)

@pytest.fixture
def fgsm_attack():
    return FGSMAttack()
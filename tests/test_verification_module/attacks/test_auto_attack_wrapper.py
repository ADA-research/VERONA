from unittest.mock import MagicMock, patch

import pytest
import torch

from ada_verona.verification_module.attacks.auto_attack_wrapper import AutoAttackWrapper


@pytest.fixture
def dummy_model():
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x
    return DummyModel()


@pytest.fixture
def dummy_data():
    return torch.rand((3, 32, 32))  


@pytest.fixture
def dummy_target():
    return torch.tensor([1])


def test_auto_attack_wrapper_runs(dummy_model, dummy_data, dummy_target):
    with patch("ada_verona.verification_module.attacks.auto_attack_wrapper.AutoAttack") as MockAttack:
        mock_adversary = MagicMock()
        
        mock_adversary.run_standard_evaluation.return_value = dummy_data.unsqueeze(0) + 0.1
        MockAttack.return_value = mock_adversary

        wrapper = AutoAttackWrapper(device="cpu", norm="L2", version="plus", verbose=True)
        perturbed = wrapper.execute(dummy_model, dummy_data, dummy_target, epsilon=0.3)

        MockAttack.assert_called_once_with(
            dummy_model, norm="L2", eps=0.3, version="plus", device="cpu", verbose=True
        )
        mock_adversary.run_standard_evaluation.assert_called_once()
        assert isinstance(perturbed, torch.Tensor)
        assert perturbed.shape == dummy_data.unsqueeze(0).shape

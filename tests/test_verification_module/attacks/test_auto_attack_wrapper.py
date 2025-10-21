import pytest
import torch


class DummyAutoAttack:
    def __init__(self, model, *, attacks=None, device='cpu', eps=0.3, norm='Linf', seed=None, version='standard'):
        self.model = model
        self.device = device
        self.epsilon = eps
        self.norm = norm
        self.version = version
        self.verbose = True  
    def run_standard_evaluation(self, data, target):
        return data.clone()  

@pytest.mark.parametrize("verbose_value", [True, False])
def test_autoattack_verbose_assignment(monkeypatch,attack_wrapper, model,data,target,verbose_value):
    monkeypatch.setattr(
        "ada_verona.verification_module.attacks.auto_attack_wrapper.AutoAttack",
        DummyAutoAttack
    )

    assert attack_wrapper.device == 'cpu'
    assert attack_wrapper.norm == 'Linf'
    assert attack_wrapper.version == 'standard'

    epsilon = 0.123
    out = attack_wrapper.execute(model, data.squeeze(0), target, epsilon) 

    attack_wrapper.verbose = verbose_value

    assert hasattr(attack_wrapper, "verbose")
    assert attack_wrapper.verbose == verbose_value

    assert isinstance(out, torch.Tensor)
    assert out.device.type == "cpu"



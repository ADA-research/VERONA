import pytest
import torch


class DummyAutoAttack:
    def __init__(self, model, *, attacks=None, device='cpu', eps=0.3, norm='Linf',
                 seed=None, version='standard', verbose=False):
        self.model = model
        self.device = device
        self.epsilon = eps
        self.norm = norm
        self.version = version
        self.verbose = verbose 
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
    assert not attack_wrapper.verbose

    epsilon = 0.123
    out = attack_wrapper.execute(model, data.squeeze(0), target, epsilon) 

    assert isinstance(out, torch.Tensor)
    assert out.device.type == "cpu"



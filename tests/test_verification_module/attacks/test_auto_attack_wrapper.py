import pytest
import torch


@pytest.mark.parametrize("verbose_value", [True, False])
def test_autoattack_verbose_assignment(attack_wrapper, model,data,target,verbose_value):

    assert attack_wrapper.device == 'cpu'
    assert attack_wrapper.norm == 'Linf'
    assert attack_wrapper.version == 'standard'

    epsilon = 0.123
    out = attack_wrapper.execute(model, data.squeeze(0), target, epsilon, verbose_value) 

    attack_wrapper.verbose = verbose_value

    assert hasattr(attack_wrapper, "verbose")
    assert attack_wrapper.verbose == verbose_value

    assert isinstance(out, torch.Tensor)
    assert out.device.type == "cpu"



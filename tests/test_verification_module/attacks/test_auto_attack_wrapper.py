import pytest
from pyautoattack import AutoAttack


class DummyModel:
    def __call__(self, x):
        return x

@pytest.fixture
def dummy_model():
    return DummyModel()

def test_verbose_assignment(dummy_model):
    epsilon = 0.1
    norm = 'Linf'
    version = 'standard'
    verbose_value = False  

    adversary = AutoAttack(
        dummy_model,
        norm=norm,
        eps=epsilon,
        version=version,
        device='cpu',
    )

    adversary.verbose = verbose_value

    assert hasattr(adversary, 'verbose'), "AutoAttack should have a 'verbose' attribute"
    assert adversary.verbose == verbose_value, "Verbose flag not correctly set"
    
    
def test_auto_attack_wrapper_initialization(attack_wrapper):
    assert attack_wrapper.device == 'cpu'
    assert attack_wrapper.norm == 'Linf'
    assert attack_wrapper.version == 'standard'


def test_auto_attack_wrapper_initialization(attack_wrapper):
    assert attack_wrapper.device == 'cpu'
    assert attack_wrapper.norm == 'Linf'
    assert attack_wrapper.version == 'standard'
    assert not attack_wrapper.verbose


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
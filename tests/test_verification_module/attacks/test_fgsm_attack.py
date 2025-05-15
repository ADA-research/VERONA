import torch


def test_fgsm_attack_execute(fgsm_attack, model, data, target):
    epsilon = 0.1
    perturbed_data = fgsm_attack.execute(model, data, target, epsilon)
    assert isinstance(perturbed_data, torch.Tensor)
    assert perturbed_data.shape == data.shape
    assert torch.all(perturbed_data >= 0) and torch.all(perturbed_data <= 1)


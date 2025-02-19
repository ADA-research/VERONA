from robustness_experiment_box.verification_module.attacks.attack import Attack
from torch import Tensor, nn
from torch.nn.modules import Module
import torch


class FGSMAttack(Attack):
    def execute(self, model: Module, data: Tensor, target: Tensor, epsilon: float) -> Tensor:
        data.requires_grad = True
        output = model(data)
        loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(output, target)

        model.zero_grad()

        loss.backward()

        data_grad = data.grad.data

        sign_data_grad = data_grad.sign()

        perturbed_image = data + epsilon*sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image
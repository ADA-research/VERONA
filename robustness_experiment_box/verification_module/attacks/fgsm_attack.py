from robustness_experiment_box.verification_module.attacks.attack import Attack
from torch import Tensor, nn
from torch.nn.modules import Module
import torch


class FGSMAttack(Attack):
    """
    A class to perform the Fast Gradient Sign Method (FGSM) attack.

    Methods:
        execute(model: Module, data: Tensor, target: Tensor, epsilon: float) -> Tensor:
            Executes the FGSM attack on the given model and data.
    """

    def execute(self, model: Module, data: Tensor, target: Tensor, epsilon: float) -> Tensor:
        """
        Execute the FGSM attack on the given model and data.

        Args:
            model (Module): The model to attack.
            data (Tensor): The input data to perturb.
            target (Tensor): The target labels for the data.
            epsilon (float): The perturbation magnitude.

        Returns:
            Tensor: The perturbed data.
        """
        data.requires_grad = True
        output = model(data)
        loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(output, target)

        model.zero_grad()

        loss.backward()

        data_grad = data.grad.data

        sign_data_grad = data_grad.sign()

        perturbed_image = data + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)  # TODO: 
        # adjust the torch clamp as Konstantin says this gives errors
        return perturbed_image

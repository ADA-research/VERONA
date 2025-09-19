import torch
from torch import Tensor, nn
from torch.nn.modules import Module

from ada_verona.verification_module.attacks.attack import Attack


class PGDAttack(Attack):
    """
    A class to perform the Projected Gradient Descent (PGD) attack.

    Attributes:
        number_iterations (int): The number of iterations for the attack.
        step_size (float): The step size for each iteration.
        randomise (bool): Whether to randomize the initial perturbation.
    """

    def __init__(self, number_iterations: int, step_size: float = None, randomise: bool = False) -> None:
        """
        Initialize the PGDAttack with specific parameters.

        Args:
            number_iterations (int): The number of iterations for the attack.
            step_size (float, optional): The step size for each iteration. Defaults to None.
            randomise (bool, optional): Whether to randomize the initial perturbation. Defaults to False.
        """
        super().__init__()
        self.number_iterations = number_iterations
        self.step_size = step_size
        self.randomise = randomise
        self.name = (
            f"PGDAttack (iterations={self.number_iterations}, "
            f"step_size={self.step_size}, randomise={self.randomise})"
        )

    def execute(self, model: Module, data: Tensor, target: Tensor, epsilon: float) -> Tensor:
        """
        Execute the PGD attack on the given model and data.

        Args:
            model (Module): The model to attack.
            data (Tensor): The input data to perturb.
            target (Tensor): The target labels for the data.
            epsilon (float): The perturbation magnitude.

        Returns:
            Tensor: The perturbed data.
        """
        # adapted from: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgd.py
        loss_fn = nn.CrossEntropyLoss()

        adv_images = data.clone().detach()

        step_size = self.step_size

        if not step_size:
            step_size = epsilon / self.number_iterations

        if self.randomise:
            adv_images = adv_images + torch.empty_like(data).uniform_(-epsilon, epsilon)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(0, self.number_iterations):
            adv_images.requires_grad = True
            output = model(adv_images)

            loss = loss_fn(output, target)
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + step_size * grad.sign()
            delta = torch.clamp(adv_images - data, min=-epsilon, max=epsilon)
            adv_images = torch.clamp(data + delta, min=0, max=1).detach()

        return adv_images

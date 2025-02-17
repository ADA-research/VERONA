from robustness_experiment_box.verification_module.attacks.attack import Attack
from torch import Tensor, nn
from torch.nn.modules import Module
import torch


class PGDAttack(Attack):
    def __init__(self, number_iterations: int, step_size: float = None, randomise: bool = False) -> None:
        super().__init__()
        self.number_iterations = number_iterations
        self.step_size = step_size
        self.randomise = randomise

    def execute(self, model: Module, data: Tensor, target: Tensor, epsilon: float) -> Tensor:
        # adapted from: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgd.py
        loss_fn = nn.CrossEntropyLoss()

        adv_images = data.clone().detach()

        step_size = self.step_size

        if not step_size:
            step_size = epsilon / self.number_iterations

        if self.randomise:
            adv_images = adv_images + torch.empty_like(data).uniform_(-epsilon,epsilon)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        
        for i in range(0, self.number_iterations):
            adv_images.requires_grad = True
            output = model(adv_images)

            loss = loss_fn(output, target)
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + step_size * grad.sign()
            delta = torch.clamp(adv_images - data, min=-epsilon, max=epsilon)
            adv_images = torch.clamp(data + delta, min=0, max=1).detach()
        

        return adv_images
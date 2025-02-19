from robustness_experiment_box.verification_module.attacks.attack import Attack
from torch import Tensor, nn
from torch.nn.modules import Module
import torch
from autoattack import AutoAttack


class AutoAttackWrapper(Attack):

    def __init__(self, device='cuda', norm='Linf', version='standard', verbose=False) -> None:
        super().__init__()
        self.device = device
        self.norm = norm
        self.version = version
        self.verbose = verbose

    def execute(self, model: Module, data: Tensor, target: Tensor, epsilon: float) -> Tensor:
        adversary = AutoAttack(model, norm=self.norm, eps=epsilon, version=self.version, device=self.device, verbose=self.verbose)
        data = data.unsqueeze(0)
        # auto attack requires NCHW input format
        perturbed_data = adversary.run_standard_evaluation(data, target)
        return perturbed_data.to(self.device)
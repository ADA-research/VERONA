from robustness_experiment_box.verification_module.attacks.attack import Attack
from torch import Tensor
from torch.nn.modules import Module
from autoattack import AutoAttack


class AutoAttackWrapper(Attack):
    """
    A wrapper for the AutoAttack adversarial attack.
    install in pip using: pip install git+https://github.com/fra31/auto-attack

    Args:
        Attack (class): The base class for attacks.
    """

    def __init__(self, device='cuda', norm='Linf',
                 version='standard', verbose=False) -> None:
        """
        Initialize the AutoAttackWrapper with specific parameters.

        Args:
            device (str, optional): The device to run the attack on.
            Defaults to 'cuda'.
            norm (str, optional): The norm to use for the attack.
            Defaults to 'Linf'.
            version (str, optional): The version of AutoAttack to use.
            Defaults to 'standard'.
            verbose (bool, optional): Whether to print verbose output.
            Defaults to False.
        """
        super().__init__()
        self.device = device
        self.norm = norm
        self.version = version
        self.verbose = verbose

    def execute(self, model: Module, data: Tensor,
                target: Tensor, epsilon: float) -> Tensor:
        """
        Execute the AutoAttack on the given model and data.

        Args:
            model (Module): The model to attack.
            data (Tensor): The input data to perturb.
            target (Tensor): The target labels for the data.
            epsilon (float): The perturbation magnitude.

        Returns:
            Tensor: The perturbed data.
        """
        adversary = AutoAttack(model, norm=self.norm, eps=epsilon,
                               version=self.version, device=self.device,
                               verbose=self.verbose)
        data = data.unsqueeze(0)
        # auto attack requires NCHW input format
        perturbed_data = adversary.run_standard_evaluation(data, target)
        return perturbed_data.to(self.device)

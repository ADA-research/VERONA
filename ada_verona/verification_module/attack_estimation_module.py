import time

import torch

from ada_verona.database.verification_context import VerificationContext
from ada_verona.database.verification_result import CompleteVerificationData, VerificationResult
from ada_verona.verification_module.attacks.attack import Attack
from ada_verona.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)
from ada_verona.verification_module.verification_module import VerificationModule


class AttackEstimationModule(VerificationModule):
    """
    A module for estimating the robustness of a model against adversarial attacks.

    """

    def __init__(self, attack: Attack) -> None:
        """
        Initialize the AttackEstimationModule with a specific attack.

        Args:
            attack (Attack): The attack to be used for robustness estimation.
        """
        self.attack = attack
        self.name = f"AttackEstimationModule ({attack.name})"

    def verify(self, verification_context: VerificationContext, epsilon: float) -> str | CompleteVerificationData:
        """
        Verify the robustness of the model within the given epsilon perturbation.

        Args:
            verification_context (VerificationContext): The context for verification, 
            including the model and data point.
            epsilon (float): The perturbation magnitude for the attack.

        Returns:
            str | CompleteVerificationData: The result of the verification,
                                either SAT or UNSAT, along with the duration.
        """

        if isinstance(verification_context.property_generator, One2AnyPropertyGenerator):
            # Check if the property generator is of type One2AnyPropertyGenerator

            start = time.time()  
            torch_model = verification_context.network.load_pytorch_model() 
            device = 'cuda' if torch.cuda.is_available() else 'cpu'  
            target = verification_context.data_point.label  
            target_on_device = torch.tensor([target], device=device)  
            data_on_device = verification_context.data_point.data.clone().detach().to(device)  
            perturbed_data = self.attack.execute(torch_model, data_on_device, target_on_device, epsilon)  
        
            output = torch_model(perturbed_data) 

            _, final_pred = output.max(1, keepdim=True)

            duration = time.time() - start 
            if final_pred == target:
                return CompleteVerificationData(result=VerificationResult.UNSAT, took=duration)
            else:
                return CompleteVerificationData(result=VerificationResult.SAT, took=duration)
        else:
            raise NotImplementedError("Currently, only one 2 any verification is implemented for adversarial attacks.")

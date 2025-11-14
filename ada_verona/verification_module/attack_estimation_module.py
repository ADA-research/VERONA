# Copyright 2025 ADA Reseach Group and VERONA council. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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

    def __init__(self, attack: Attack, top_k: int = 1) -> None:
        """
        Initialize the AttackEstimationModule with a specific attack.

        Args:
            attack (Attack): The attack to be used for robustness estimation.
            top_k: Number of top scores to take into account for checking the prediction.
        """
        self.attack = attack
        self.top_k = top_k
        self.name = f"AttackEstimationModule [{attack.name}, top-{top_k}]"

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

            _, predicted_labels = torch.topk(output, self.top_k) 

            duration = time.time() - start 
            if target in predicted_labels:
                return CompleteVerificationData(result=VerificationResult.UNSAT, took=duration, 
                                                obtained_labels=predicted_labels)
            else:
                return CompleteVerificationData(result=VerificationResult.SAT, took=duration, 
                                                obtained_labels=predicted_labels)
        else:
            raise NotImplementedError("Currently, only one 2 any verification is implemented for adversarial attacks.")

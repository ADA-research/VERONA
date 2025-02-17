from autoverify.verifier.verification_result import CompleteVerificationData
import torch
from autoverify.verifier.verification_result import CompleteVerificationData
import time

from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.verification_module.verification_module import VerificationModule
from robustness_experiment_box.database.verification_result import VerificationResult

from robustness_experiment_box.verification_module.attacks.attack import Attack


class AttackEstimationModule(VerificationModule):

    def __init__(self, attack: Attack) -> None:
        self.attack = attack

    def verify(self, verification_context: VerificationContext, epsilon: float) -> str | CompleteVerificationData:

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
from autoverify.verifier.verification_result import CompleteVerificationData

from robustness_experiment_box.verification_module.verification_module import VerificationModule
from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.database.verification_result import VerificationResult


class TestVerificationModule(VerificationModule):

    def verify(self, verification_context: VerificationContext, epsilon: float) -> str | CompleteVerificationData:

        if not verification_context.network.path.exists():
            raise Exception("[TestVerificationModule]: network path not found")
        
        if not verification_context.labeled_image.image_path.exists():
            raise Exception("[TestVerificationModule]: image path not found")
        
        if epsilon > 0.5:
            return CompleteVerificationData(result=VerificationResult.SAT, took=10.0)

        else:
            return CompleteVerificationData(result=VerificationResult.UNSAT, took=10.0)
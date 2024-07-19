from autoverify.verifier.verification_result import CompleteVerificationData
import subprocess

from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.verification_module.property_generator.property_generator import PropertyGenerator
from robustness_experiment_box.verification_module.verification_module import VerificationModule


class NnenumModule(VerificationModule):

    def __init__(self, timeout) -> None:
        self.property_generator = PropertyGenerator()
        self.timeout = timeout

    def verify(self, verification_context: VerificationContext, epsilon: float) -> str | CompleteVerificationData:
        image, _ = verification_context.labeled_image.load(-1)

        vnnlib_property = self.property_generator.create_vnnlib_property(image, verification_context.labeled_image.label, epsilon, 10, 0, 1)
        
        verification_context.save_vnnlib_property(vnnlib_property)
        result = subprocess.run(f"python -m nnenum.nnenum {str(verification_context.network.path)} {str(vnnlib_property.path)} {str(self.timeout)}", shell=True, capture_output=True, text=True)

        print(result)

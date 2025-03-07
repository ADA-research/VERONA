import subprocess

from autoverify.verifier.verification_result import CompleteVerificationData

from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.verification_module.verification_module import VerificationModule


class NnenumModule(VerificationModule):
    """
    A module for verifying the robustness of a model using the NNENUM verifier.
    """

    def __init__(self, timeout) -> None:
        """
        Initialize the NnenumModule with a specific timeout.
            timeout (float): The timeout for the verification process.
        """
        self.timeout = timeout

    def verify(self, verification_context: VerificationContext, epsilon: float) -> str | CompleteVerificationData:
        """
        Verify the robustness of the model within the given epsilon perturbation.
        Args:
            verification_context (VerificationContext): The context for verification, including the model and data point.
            epsilon (float): The perturbation magnitude for the attack.

        Returns:
            str | CompleteVerificationData: The result of the verification, either SAT or UNSAT, along with the duration.
        """
        image = verification_context.data_point.data.reshape(-1).detach().numpy()

        vnnlib_property = verification_context.property_generator.create_vnnlib_property(image, verification_context.data_point.label, epsilon)
        
        verification_context.save_vnnlib_property(vnnlib_property)
        result = subprocess.run(f"python -m nnenum.nnenum {str(verification_context.network.path)} {str(vnnlib_property.path)} {str(self.timeout)}", shell=True, capture_output=True, text=True)
        #TODO: we should be returning CompleteVerificationData here
        return result

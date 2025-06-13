from abc import ABC, abstractmethod

from autoverify.verifier.verification_result import CompleteVerificationData

from robustness_experiment_box.database.verification_context import VerificationContext


class VerificationModule(ABC):
    @abstractmethod
    def verify(self, verification_context: VerificationContext, epsilon: float) -> str | CompleteVerificationData:
        """Main method to verify an image for a given network and epsilon value"""
        raise NotImplementedError

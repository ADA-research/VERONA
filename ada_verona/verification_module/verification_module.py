from abc import ABC, abstractmethod

from ada_verona.database.verification_context import VerificationContext
from ada_verona.database.verification_result import CompleteVerificationData


class VerificationModule(ABC):
    @abstractmethod
    def verify(self, verification_context: VerificationContext, epsilon: float) -> str | CompleteVerificationData:
        """Main method to verify an image for a given network and epsilon value"""
        raise NotImplementedError

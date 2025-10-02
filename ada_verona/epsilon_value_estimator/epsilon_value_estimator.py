from abc import ABC, abstractmethod

from ada_verona.database.epsilon_value_result import EpsilonValueResult
from ada_verona.database.verification_context import VerificationContext
from ada_verona.verification_module.verification_module import VerificationModule


class EpsilonValueEstimator(ABC):
    """
    An abstract base class for estimating epsilon values.
    """

    def __init__(self, epsilon_value_list: list[float], verifier: VerificationModule) -> None:
        """
        Initialize the EpsilonValueEstimator with the given epsilon value list and verifier.

        Args:
            epsilon_value_list (list[float]): The list of epsilon values to estimate.
            verifier (VerificationModule): The verifier to use for verification.
        """
        self.epsilon_value_list = epsilon_value_list
        self.verifier = verifier

    @abstractmethod
    def compute_epsilon_value(self, verification_context: VerificationContext) -> EpsilonValueResult:
        """
        Compute the epsilon value for the given verification context.

        Args:
            verification_context (VerificationContext): The context for verification.

        Returns:
            EpsilonValueResult: The result of the epsilon value estimation.
        """
        raise NotImplementedError("This is an abstract method and should be implemented in subclasses.")

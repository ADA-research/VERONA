from abc import ABC, abstractmethod

from autoverify.verifier.verification_result import CompleteVerificationData

from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.database.verification_result import ProbabilisticCertificationResult


class VerificationModule(ABC):
    """
    Abstract base class for all verification modules.

    This interface supports both traditional deterministic verification and
    probabilistic verification methods like randomized smoothing.
    """

    @abstractmethod
    def verify(self, verification_context: VerificationContext, epsilon: float) -> str | CompleteVerificationData:
        """
        Main method to verify an image for a given network and epsilon value.

        For traditional verification, this performs deterministic verification.
        For probabilistic methods, this may perform Monte Carlo sampling.

        Args:
            verification_context: The context containing network, data point, and property generator
            epsilon: The perturbation magnitude to verify

        Returns:
            Either a string result ("SAT"/"UNSAT") or CompleteVerificationData object
        """
        raise NotImplementedError

    def verify_probabilistic(
        self,
        verification_context: VerificationContext,
        n0: int,
        n: int,
        alpha: float,
        sigma: float,
        batch_size: int = 1000,
        diffusion_timestep: int | None = None
    ) -> "ProbabilisticCertificationResult":
        """
        Perform probabilistic certification using Monte Carlo sampling.

        This method supports randomized smoothing and diffusion denoised smoothing approaches.
        Default implementation raises NotImplementedError - subclasses should override.

        Args:
            verification_context: The context containing network, data point, and property generator
            n0: Number of samples for initial prediction
            n: Number of samples for certification
            alpha: Confidence level (failure probability)
            sigma: Noise level for randomized smoothing
            batch_size: Batch size for sampling
            diffusion_timestep: Optional timestep for diffusion denoising

        Returns:
            ProbabilisticCertificationResult with predicted class and certified radius

        Raises:
            NotImplementedError: If subclass doesn't support probabilistic certification
        """
        raise NotImplementedError("Probabilistic certification not implemented in this module")

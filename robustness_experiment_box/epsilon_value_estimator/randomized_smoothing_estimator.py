"""
Randomized Smoothing Estimator for VERONA.

This estimator provides direct computation of certified radii using randomized smoothing,
eliminating the need for binary search over epsilon values. It works with the
RandomizedSmoothingModule to provide probabilistic robustness guarantees.

This approach is more efficient than binary search for probabilistic certification
since it directly computes the certified radius rather than searching for boundaries.
"""

import logging

from robustness_experiment_box.database.epsilon_value_result import EpsilonValueResult
from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.database.verification_result import ProbabilisticCertificationResult
from robustness_experiment_box.epsilon_value_estimator.epsilon_value_estimator import EpsilonValueEstimator
from robustness_experiment_box.verification_module.randomized_smoothing_module import RandomizedSmoothingModule

logger = logging.getLogger(__name__)


class RandomizedSmoothingEstimator(EpsilonValueEstimator):
    """
    Estimator for probabilistic robustness using randomized smoothing.

    This estimator directly computes certified radii using Monte Carlo sampling
    rather than performing binary search over epsilon values. It is
    designed to work with the RandomizedSmoothingModule.

    Args:
        smoothing_module: The randomized smoothing verification module
        n0: Number of samples for initial prediction
        n: Number of samples for certification
        alpha: Confidence level (failure probability)
        batch_size: Batch size for sampling
    """

    def __init__(
        self,
        smoothing_module: RandomizedSmoothingModule,
        n0: int = 100,
        n: int = 100000,
        alpha: float = 0.001,
        batch_size: int = 1000
    ):
        # Pass empty epsilon list since we don't use binary search
        super().__init__([], smoothing_module)

        self.smoothing_module = smoothing_module
        self.n0 = n0
        self.n = n
        self.alpha = alpha
        self.batch_size = batch_size

    def compute_epsilon_value(self, verification_context: VerificationContext) -> EpsilonValueResult:
        """
        Compute the certified epsilon value using randomized smoothing.

        This method performs direct computation of the certified radius.

        Args:
            verification_context: The verification context

        Returns:
            EpsilonValueResult with the certified radius as epsilon value
        """
        # Perform probabilistic verification
        probabilistic_result = self.smoothing_module.verify_probabilistic(
            verification_context=verification_context,
            n0=self.n0,
            n=self.n,
            alpha=self.alpha,
            sigma=self.smoothing_module.sigma,
            batch_size=self.batch_size
        )

        # Convert to traditional EpsilonValueResult format
        # Use certified_radius as the epsilon value for compatibility
        epsilon_value_result = EpsilonValueResult(
            verification_context=verification_context,
            epsilon=probabilistic_result.certified_radius,
            smallest_sat_value=probabilistic_result.certified_radius,
            time=probabilistic_result.certification_time,
            verifier=self.smoothing_module.name
        )

        logger.info(
            f"Randomized smoothing result: predicted_class={probabilistic_result.predicted_class}, "
            f"certified_radius={probabilistic_result.certified_radius:.4f}, "
            f"confidence={probabilistic_result.confidence:.4f}, "
            f"time={probabilistic_result.certification_time:.2f}s"
        )

        return epsilon_value_result

    def get_probabilistic_result(self, verification_context: VerificationContext) -> ProbabilisticCertificationResult:
        """
        Get the full probabilistic certification result.

        This method provides access to the complete probabilistic result
        including predicted class and confidence information.

        Args:
            verification_context: The verification context

        Returns:
            Probabilistic certification result
        """
        return self.smoothing_module.verify_probabilistic(
            verification_context=verification_context,
            n0=self.n0,
            n=self.n,
            alpha=self.alpha,
            sigma=self.smoothing_module.sigma,
            batch_size=self.batch_size
        )

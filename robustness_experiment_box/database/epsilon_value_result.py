from dataclasses import dataclass

from robustness_experiment_box.database.verification_context import VerificationContext


@dataclass
class EpsilonValueResult:
    """
    A dataclass defining the verification result of a single verification.

    This class is designed for traditional deterministic verification approaches
    where we search for specific epsilon values. For probabilistic certification
    (like randomized smoothing), use ProbabilisticCertificationResult directly.

    For traditional verification:
    - epsilon: The certified epsilon value
    - smallest_sat_value: The smallest epsilon where SAT was found
    """

    verification_context: VerificationContext
    epsilon: float
    smallest_sat_value: float
    time: float = None
    verifier: str = None

    def to_dict(self) -> dict:
        """
        Convert the EpsilonValueResult to a dictionary.

        Returns:
            dict: The dictionary representation of the EpsilonValueResult.
        """
        return dict(
            **self.verification_context.get_dict_for_epsilon_result(),
            epsilon_value=self.epsilon,
            smallest_sat_value=self.smallest_sat_value,
            total_time=self.time,
            verifier=self.verifier,
        )

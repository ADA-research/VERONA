from dataclasses import dataclass
import numpy as np

from robustness_experiment_box.database.verification_result import VerificationResult


@dataclass
class EpsilonStatus:
    """
    A class to represent the status of the verification.
    meaning it records the epsilon value, the result (SAT, UNSAT, TIMEOUT, ERROR) and running time.
    """

    value: float
    result: VerificationResult | None
    time: float = None

    def to_dict(self) -> dict:
        """
        Convert the EpsilonStatus to a dictionary.

        Returns:
            dict: The dictionary representation of the EpsilonStatus.
        """
        return dict(epsilon_value=self.value, result=self.result, time=self.time)

from dataclasses import dataclass
import numpy as np

from robustness_experiment_box.database.verification_result import VerificationResult

@dataclass
class EpsilonStatus:
    value: float
    result: VerificationResult | None
    time: float = None

    def to_dict(self) -> dict:
        return dict(epsilon_value=self.value, result=self.result, time=self.time)

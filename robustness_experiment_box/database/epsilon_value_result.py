from dataclasses import dataclass
import numpy as np

from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.database.epsilon_status import EpsilonStatus

@dataclass
class EpsilonValueResult:
    """
    class defining the verification result of a single verification
    """
    verification_context: VerificationContext
    epsilon: float
    smallest_sat_value: float
    time: float = None
    
    def to_dict(self):
        ret = dict(**self.verification_context.get_dict_for_epsilon_result(), epsilon_value=self.epsilon, smallest_sat_value=self.smallest_sat_value, total_time=self.time) 
        return ret
    
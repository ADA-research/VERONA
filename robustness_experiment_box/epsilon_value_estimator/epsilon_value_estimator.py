from abc import ABC, abstractmethod

from robustness_experiment_box.verification_module.verification_module import VerificationModule
from robustness_experiment_box.database.epsilon_value_result import EpsilonValueResult
from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.database.epsilon_status import EpsilonStatus

class EpsilonValueEstimator(ABC):

    def __init__(self, epsilon_value_list: list[float], verifier: VerificationModule) -> None:
        self.epsilon_value_list = epsilon_value_list
        self.verifier = verifier

    @abstractmethod
    def compute_epsilon_value(self, verification_context: VerificationContext) -> EpsilonValueResult:
        pass
import logging

logger = logging.getLogger(__name__)

from robustness_experiment_box.verification_module.verification_module import VerificationModule
from robustness_experiment_box.epsilon_value_estimator.epsilon_value_estimator import EpsilonValueEstimator
from robustness_experiment_box.database.epsilon_value_result import EpsilonValueResult
from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.database.verification_result import VerificationResult
from robustness_experiment_box.database.epsilon_status import EpsilonStatus


class IterativeEpsilonValueEstimator(EpsilonValueEstimator):

    def compute_epsilon_value(self, verification_context: VerificationContext) -> EpsilonValueResult:

        epsilon_status_list = [EpsilonStatus(x, None) for x in self.epsilon_value_list]

        highest_unsat_value, lowest_sat_value, epsilon_status_list = self.iterative_search(verification_context, epsilon_status_list)
        epsilon_value_result = EpsilonValueResult(verification_context, highest_unsat_value, epsilon_status_list)

        return epsilon_value_result

    def iterative_search(self, verification_context: VerificationContext, epsilon_status_list: list[EpsilonStatus]) -> float:

        for index in range(0, len(epsilon_status_list)):
            
            result = self.verifier.verify(verification_context, epsilon_status_list[index].value)
            epsilon_status_list[index].result = result.result
            epsilon_status_list[index].time = result.took
            logger.info(f"epsilon value: {epsilon_status_list[index].value}, result: {result.result}")

        highest_unsat = None

        if len([x for x in epsilon_status_list if x.result == VerificationResult.UNSAT]) > 0:
            highest_unsat = max([index for index, x in enumerate(epsilon_status_list) if x.result == VerificationResult.UNSAT])

        highest_unsat_value = epsilon_status_list[highest_unsat].value if not highest_unsat is None else 0

        lowest_sat = None

        if len([x for x in epsilon_status_list if x.result == VerificationResult.SAT]) > 0:
            lowest_sat = min([index  for index, x in enumerate(epsilon_status_list) if x.result == VerificationResult.SAT])

        lowest_sat_value = epsilon_status_list[lowest_sat].value if not lowest_sat is None else "undefined"

        return highest_unsat_value, lowest_sat_value, epsilon_status_list

import logging

import time
logger = logging.getLogger(__name__)
import torch
from robustness_experiment_box.verification_module.verification_module import VerificationModule
from robustness_experiment_box.epsilon_value_estimator.epsilon_value_estimator import BinarySearchEpsilonValueEstimator
from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.database.verification_result import VerificationResult
from robustness_experiment_box.database.epsilon_status import EpsilonStatus


class QuarteredBinarySearchEpsilonValueEstimator(BinarySearchEpsilonValueEstimator):


    def get_next_epsilon(self, midpoint:int, first:int, last:int) -> int:
        random_number = torch.rand(1).item()
        if random_number<= 0.5:
            next = (first + midpoint) // 2
            if next == midpoint:
                return first
        else: 
            next = (midpoint + last) // 2
            if next == midpoint:
                return last

    def binary_search(self, verification_context: VerificationContext, epsilon_status_list: list[EpsilonStatus]) -> float:

        if len(epsilon_status_list) == 1:
            outcome = self.verifier.verify(verification_context, epsilon_status_list[0].value)
            result = outcome.result
            epsilon_status_list[0].time = outcome.took
            epsilon_status_list[0].result = result
            logger.debug(f'current epsilon value: {epsilon_status_list[0].result}, took: {epsilon_status_list[0].time}')
            verification_context.save_result(epsilon_status_list[0])
            if result == VerificationResult.UNSAT:
                return epsilon_status_list[0].value, self.get_smallest_sat(epsilon_status_list)
            else:
                return 0, self.get_smallest_sat(epsilon_status_list)
        
        first = 0
        last = len(epsilon_status_list) - 1
        midpoint = (first + last) // 2

        while first<=last:

            if not epsilon_status_list[midpoint].result:

                outcome = self.verifier.verify(verification_context, epsilon_status_list[midpoint].value)
                epsilon_status_list[midpoint].result = outcome.result
                epsilon_status_list[midpoint].time = outcome.took
                verification_context.save_result(epsilon_status_list[midpoint])
                logger.debug(f'current epsilon value: {epsilon_status_list[midpoint].result}, took: {epsilon_status_list[midpoint].time}')
                
            if epsilon_status_list[midpoint].result == VerificationResult.UNSAT:
                first = midpoint + 1
                midpoint = (first + last) // 2
            elif epsilon_status_list[midpoint].result == VerificationResult.SAT:
                last = midpoint - 1
                midpoint = (first + last) // 2
            else:
                if len(epsilon_status_list)>3:
                    midpoint = self.get_next_epsilon(midpoint=midpoint, first=first,last=last)
                    epsilon_status_list.pop(midpoint)
                    last = last - 1
                else:
                    epsilon_status_list.pop(midpoint)
                    last = last - 1
                    midpoint = (first + last) // 2


        
        logger.debug(f"epsilon status list: {[(x.value, x.result, x.time) for x in epsilon_status_list]}")

        highest_unsat_value = self.get_highest_unsat(epsilon_status_list)

        smallest_sat_value = self.get_smallest_sat(epsilon_status_list)

        return highest_unsat_value, smallest_sat_value

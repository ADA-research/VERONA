from pathlib import Path
from result import Err, Ok
import numpy as np
import re
import logging
logger = logging.getLogger(__name__)

import autoverify
from autoverify.verifier.verification_result import CompleteVerificationData

from robustness_experiment_box.verification_module.property_generator.property_generator import PropertyGenerator
from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.verification_module.verification_module import VerificationModule

class AutoVerifyModule(VerificationModule):

    def __init__(self, verifier: autoverify.verifier.verifier.Verifier,  property_generator: PropertyGenerator, timeout: float, config: Path = None) -> None:
        self.verifier = verifier
        self.property_generator = property_generator
        self.timeout = timeout
        self.config = config


    def verify(self, verification_context: VerificationContext, epsilon: float) -> str | CompleteVerificationData:

        image  = verification_context.data_point.data.reshape(-1).detach().numpy()
        vnnlib_property = self.property_generator.create_vnnlib_property(image, verification_context.data_point.label, epsilon)
    
        verification_context.save_vnnlib_property(vnnlib_property)

        if self.config:
            result = self.verifier.verify_property(verification_context.network.path, vnnlib_property.path, timeout=self.timeout, config=self.config)
        else:
            result = self.verifier.verify_property(verification_context.network.path, vnnlib_property.path, timeout=self.timeout)

        if isinstance(result, Ok):
            outcome = result.unwrap()
            return outcome
        elif isinstance(result, Err):
            logger.info(f"Error during verification: {result.unwrap_err()}")
            return result.unwrap_err()



def parse_counter_example(result: Ok):
    string_list_without_sat = [x for x in result.unwrap().counter_example.split("\n") if not "sat" in x]
    numbers = [x.replace("(", "").replace(")", "") for x in string_list_without_sat if "Y" not in x]
    counter_example_array = np.array([float(re.sub(r'X_\d*', '', x).strip()) for x in numbers])

    return counter_example_array.reshape(28,28)

def parse_counter_example_label(result: Ok):
    string_list_without_sat = [x for x in result.unwrap().counter_example.split("\n") if not "sat" in x]
    numbers = [x.replace("(", "").replace(")", "") for x in string_list_without_sat if "X" not in x]
    counter_example_array = np.array([float(re.sub(r'Y_\d*', '', x).strip()) for x in numbers])

    return np.argmax(counter_example_array)
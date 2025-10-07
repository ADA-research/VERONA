import logging
import re
from pathlib import Path

import numpy as np
from autoverify.verifier.verifier import Verifier
from result import Err, Ok

from ada_verona.database.verification_context import VerificationContext
from ada_verona.database.verification_result import CompleteVerificationData
from ada_verona.verification_module.verification_module import VerificationModule

logger = logging.getLogger(__name__)



class AutoVerifyModule(VerificationModule):
    """
    A module for automatically verifying the robustness of a model using a specified verifier.
    """

    def __init__(self, verifier: Verifier, timeout: float, config: Path = None) -> None:
        """
        Initialize the AutoVerifyModule with a specific verifier, timeout, and optional configuration.
        Args:
            verifier (Verifier): The verifier to be used for robustness verification.
            timeout (float): The timeout for the verification process.
            config (Path, optional): The configuration file for the verifier.
        """

        self.verifier = verifier
        self.timeout = timeout
        self.config = config
        self.name = f"AutoVerifyModule ({verifier.name})" 

    def verify(self, verification_context: VerificationContext, epsilon: float) -> str | CompleteVerificationData:
        """
        Verify the robustness of the model within the given epsilon perturbation.
        Args:
            verification_context (VerificationContext): The context for verification,
            including the model and data point.
            epsilon (float): The perturbation magnitude for the attack.

        Returns:
            str | CompleteVerificationData: The result of the verification,
            either SAT or UNSAT, along with the duration.
        """
        image = verification_context.data_point.data.reshape(-1).detach().numpy()
        vnnlib_property = verification_context.property_generator.create_vnnlib_property(
            image, verification_context.data_point.label, epsilon
        )

        verification_context.save_vnnlib_property(vnnlib_property)

        if self.config:
            result = self.verifier.verify_property(
                verification_context.network.path, vnnlib_property.path, timeout=self.timeout, config=self.config
            )
        else:
            result = self.verifier.verify_property(
                verification_context.network.path, vnnlib_property.path, timeout=self.timeout
            )

        if isinstance(result, Ok):
            outcome = result.unwrap()
            return outcome
        elif isinstance(result, Err):
            logger.info(f"Error during verification: {result.unwrap_err()}")
            return result.unwrap_err()

def parse_counter_example(result: Ok, verification_context: VerificationContext) -> np.ndarray:
    """
    Parse the counter example from the verification result.

    Args:
        result (Ok): The verification result containing the counter example.

    Returns:
        np.ndarray: The parsed counter example as a numpy array.
    """
    string_list_without_sat = [x for x in result.unwrap().counter_example.split("\n") if "sat" not in x]
    numbers = [x.replace("(", "").replace(")", "") for x in string_list_without_sat if "Y" not in x]
    counter_example_array = np.array([float(re.sub(r'X_\d*', '', x).strip()) for x in numbers if x.strip()])
    

    return counter_example_array.reshape(verification_context.data_point.data.shape)

def parse_counter_example_label(result: Ok) -> int:
    """
    Parse the counter example label from the verification result.

    Args:
        result (Ok): The verification result containing the counter example.

    Returns:
        int: The parsed counter example label.
    """
    string_list_without_sat = [x for x in result.unwrap().counter_example.split("\n") if "sat" not in x]
    numbers = [x.replace("(", "").replace(")", "") for x in string_list_without_sat if "X" not in x]
    counter_example_array = np.array([float(re.sub(r'Y_\d*', '', x).strip()) for x in numbers if x.strip()])

    return int(np.argmax(counter_example_array))
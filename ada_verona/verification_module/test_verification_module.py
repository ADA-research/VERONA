import re
from pathlib import Path

from result import Ok
from torch import Tensor, load

from ada_verona.database.verification_context import VerificationContext
from ada_verona.database.verification_result import CompleteVerificationData, VerificationResult
from ada_verona.verification_module.verification_module import VerificationModule


class TestVerificationModule(VerificationModule):
    def __init__(self) -> None:
        """
        Initialize the TestVerificationModule.
        """
        super().__init__()
        self.name = "TestVerificationModule"
        
    def verify(self, verification_context: VerificationContext, epsilon: float) -> str | CompleteVerificationData:
        """
        A module for testing other parts of the pipeline. This module does not actually verify anything.
        It returns SAT or UNSAT based on the size of epsilon.
        Args:
            verification_context (VerificationContext): The context for verification,
            including the model and data point.
            epsilon (float): The test perturbation.

        Raises:
            Exception: When no network path is found in the verification context.
            Exception: When no image path is found in the verification context.
        Returns:
            str | CompleteVerificationData: the result including verification result and time taken.
        """

        if not verification_context.network.path.exists():
            raise Exception("[TestVerificationModule]: network path not found")

        if epsilon > 0.5:
            return CompleteVerificationData(result=VerificationResult.SAT, took=10.0)

        else:
            return CompleteVerificationData(result=VerificationResult.UNSAT, took=10.0)
        

    def _extract_epsilon_from_file_name(self,vnnlib_property_path: Path):
        stem = vnnlib_property_path.stem
        parts = stem.split("_")
        if len(parts) >= 3:
            try:
                return float(".".join(parts[-2:]))
            except ValueError:
                m = re.search(r"([0-9]+(?:\.[0-9]+)?)", stem)
                return float(m.group(1)) if m else 0.0
        else:
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", stem)
            return float(m.group(1)) if m else 0.0
        
    def verify_property(
        self, network_path: Path, vnnlib_property_path: Path, timeout: int, config=None
    ) -> str | CompleteVerificationData:
        """ 
        A module for testing other parts of the pipeline. This module does not actually verify anything.
        It returns SAT or UNSAT based on the size of epsilon. 
        Args:
            verification_context (VerificationContext): The context for verification, 
            including the model and data point.
            epsilon (float): The test perturbation. 

        Raises:
            Exception: When no network path is found in the verification context. 
            Exception: When no image path is found in the verification context.
        Returns:
            str | CompleteVerificationData: the result including verification result and time taken.
        """
        

        if not Path(network_path).exists():
            raise Exception("[TestVerificationModule]: network path not found")
        
        if not Path(vnnlib_property_path).exists():
            raise Exception("[TestVerificationModule]: image path not found")
        
        epsilon = self._extract_epsilon_from_file_name(vnnlib_property_path)
        result = VerificationResult.SAT if epsilon > 0.5 else VerificationResult.UNSAT

        return Ok(CompleteVerificationData(result=result, took=timeout))
        
   
    def execute(self, torch_model, data_on_device, target_on_device, epsilon) -> Tensor:
        """ 
        A module for testing other parts of the pipeline. This module does not actually verify anything.
        It returns a tensor of the data
        Args:
            torch_model (Module): The model to attack.
            data_on_device (Tensor): The input data to perturb.
            target_on_device (Tensor): The target labels for the data.
            epsilon (float): The perturbation magnitude
        Not all args are used as this is a test module and we do not need to perturb the data. 
        We do need the args as the function signature is fixed by the parent class.        
        Returns:
            tensor: the data tensor, either perturbed or not
        """

        if epsilon >= 0.5:
            return load("examples/example_experiment/data/images/mnist_train_1.pt")

        else:
            return data_on_device

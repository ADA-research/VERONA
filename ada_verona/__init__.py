"""
ADA-VERONA: Neural Network Robustness Analysis Framework

A framework for analyzing neural network robustness
through verification and adversarial testing.
"""

import importlib.util
import warnings
from importlib.metadata import version

# Database classes
from .database.dataset.data_point import DataPoint
from .database.dataset.experiment_dataset import ExperimentDataset
from .database.dataset.image_file_dataset import ImageFileDataset
from .database.dataset.pytorch_experiment_dataset import PytorchExperimentDataset
from .database.epsilon_status import EpsilonStatus
from .database.epsilon_value_result import EpsilonValueResult
from .database.experiment_repository import ExperimentRepository
from .database.machine_learning_model.network import Network
from .database.machine_learning_model.onnx_network import ONNXNetwork
from .database.machine_learning_model.pytorch_network import PyTorchNetwork
from .database.machine_learning_model.torch_model_wrapper import TorchModelWrapper
from .database.verification_context import VerificationContext
from .database.verification_result import VerificationResult
from .database.vnnlib_property import VNNLibProperty

# Dataset sampler classes
from .dataset_sampler.dataset_sampler import DatasetSampler
from .dataset_sampler.predictions_based_sampler import PredictionsBasedSampler

# Epsilon value estimator classes
from .epsilon_value_estimator.binary_search_epsilon_value_estimator import (
    BinarySearchEpsilonValueEstimator,
)
from .epsilon_value_estimator.epsilon_value_estimator import EpsilonValueEstimator
from .epsilon_value_estimator.iterative_epsilon_value_estimator import (
    IterativeEpsilonValueEstimator,
)

# Verification module classes
from .verification_module.attack_estimation_module import AttackEstimationModule
from .verification_module.attacks.attack import Attack
from .verification_module.attacks.fgsm_attack import FGSMAttack
from .verification_module.attacks.pgd_attack import PGDAttack
from .verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)
from .verification_module.property_generator.one2one_property_generator import (
    One2OnePropertyGenerator,
)
from .verification_module.property_generator.property_generator import PropertyGenerator
from .verification_module.verification_module import VerificationModule

__version__ = version("ada-verona")
__author__ = "ADA Research Group"
# Check for autoattack availability
HAS_AUTOATTACK = importlib.util.find_spec("autoattack") is not None
if not HAS_AUTOATTACK:
    warnings.warn(
        "AutoAttack not found. Some adversarial attack features will be limited. "
        "To install: uv pip install git+https://github.com/fra31/auto-attack",
        stacklevel=2,
    )

# Check for autoverify availability
HAS_AUTOVERIFY = importlib.util.find_spec("autoverify") is not None
if not HAS_AUTOVERIFY:
    warnings.warn(
        "AutoVerify not found. Some complete verification features will be limited. "
        "To install: uv pip install auto-verify",
        stacklevel=2,
    )
    
    
__all__ = [
    "__version__",
    "__author__",
    "HAS_AUTOATTACK",
    "HAS_AUTOVERIFY",
    
    # Core abstract classes
    "DatasetSampler",
    "EpsilonValueEstimator",
    "VerificationModule",
    "Network",
    "PropertyGenerator",
    "Attack",
    "ExperimentDataset",
   
    # Database classes
    "ExperimentRepository",
    "VerificationContext",
    "ONNXNetwork",
    "PyTorchNetwork",
    "TorchModelWrapper",
    "VNNLibProperty",
    "VerificationResult",
    "EpsilonValueResult",
    "EpsilonStatus",
    "DataPoint",

    # Dataset sampler classes
    "PredictionsBasedSampler",
    "PytorchExperimentDataset",
    "ImageFileDataset",

    # Epsilon value estimator classes
    "BinarySearchEpsilonValueEstimator",
    "IterativeEpsilonValueEstimator",

    # Verification module classes
    "AttackEstimationModule",
    "PGDAttack",
    "FGSMAttack",

    # Property generator classes
    "One2AnyPropertyGenerator",
    "One2OnePropertyGenerator",
]



if HAS_AUTOATTACK:
    auto_attack_module = importlib.import_module(
        ".verification_module.attacks.auto_attack_wrapper", __package__
    )
    AutoAttackWrapper = auto_attack_module.AutoAttackWrapper
    __all__.extend(["AutoAttackWrapper"])

if HAS_AUTOVERIFY:
    autoverify_module = importlib.import_module(
        ".verification_module.auto_verify_module", __package__
    )
    AutoVerifyModule = autoverify_module.AutoVerifyModule
    parse_counter_example = autoverify_module.parse_counter_example
    parse_counter_example_label = autoverify_module.parse_counter_example_label
    __all__.extend([
        "AutoVerifyModule",
        "parse_counter_example",
        "parse_counter_example_label",
    ])

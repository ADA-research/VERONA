"""
ADA-VERONA: Neural Network Robustness Analysis Framework

A framework for analyzing neural network robustness
through verification and adversarial testing.
"""

__version__ = "1.0.0"
__author__ = "ADA Research Group"

import contextlib

from .robustness_experiment_box import (
    analysis,
    database,
    dataset_sampler,
    epsilon_value_estimator,
    util,
    verification_module,
)

with contextlib.suppress(ImportError):
    from .robustness_experiment_box.database.dataset.image_file_dataset import ImageFileDataset
    from .robustness_experiment_box.database.dataset.pytorch_experiment_dataset import PytorchExperimentDataset
    from .robustness_experiment_box.database.experiment_repository import ExperimentRepository
    from .robustness_experiment_box.database.machine_learning_model.network import Network
    from .robustness_experiment_box.database.verification_context import VerificationContext
    from .robustness_experiment_box.dataset_sampler.dataset_sampler import DatasetSampler
    from .robustness_experiment_box.dataset_sampler.predictions_based_sampler import PredictionsBasedSampler
    from .robustness_experiment_box.epsilon_value_estimator.binary_search_epsilon_value_estimator import (
        BinarySearchEpsilonValueEstimator,
    )
    from .robustness_experiment_box.epsilon_value_estimator.epsilon_value_estimator import EpsilonValueEstimator
    from .robustness_experiment_box.verification_module.attack_estimation_module import AttackEstimationModule
    from .robustness_experiment_box.verification_module.attacks.auto_attack_wrapper import AutoAttackWrapper
    from .robustness_experiment_box.verification_module.attacks.pgd_attack import PGDAttack
    from .robustness_experiment_box.verification_module.auto_verify_module import (
        AutoVerifyModule,
        parse_counter_example,
        parse_counter_example_label,
    )
    from .robustness_experiment_box.verification_module.property_generator.one2any_property_generator import (
        One2AnyPropertyGenerator,
    )
    from .robustness_experiment_box.verification_module.verification_module import VerificationModule

# Check for autoattack availability
try:
    import importlib.util
    HAS_AUTOATTACK = importlib.util.find_spec("pyautoattack") is not None
except ImportError:
    HAS_AUTOATTACK = False
    
if not HAS_AUTOATTACK:
    import warnings
    warnings.warn(
        "PyAutoAttack not found. Some adversarial attack features will be limited. "
        "To install: pip install pyautoattack",
        stacklevel=2
    )



__all__ = [
    "__version__",
    "__author__",
    "HAS_AUTOATTACK",
    "analysis",
    "database",
    "dataset_sampler",
    "epsilon_value_estimator",
    "util",
    "verification_module",
    
    "DatasetSampler",
    "EpsilonValueEstimator",
    "VerificationModule",
   
    "ExperimentRepository",
    "Network",
    "VerificationContext",

    "PredictionsBasedSampler",
    "PytorchExperimentDataset",
    "ImageFileDataset",
    "BinarySearchEpsilonValueEstimator",
    "AttackEstimationModule",
    "PGDAttack",
    "AutoAttackWrapper",
    "AutoVerifyModule",
    "parse_counter_example",
    "parse_counter_example_label",

    "One2AnyPropertyGenerator",
    "list_auto_verify_verifiers",
    "create_auto_verify_verifier",
]

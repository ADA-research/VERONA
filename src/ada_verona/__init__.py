"""
ADA-VERONA: Neural Network Robustness Analysis Framework

A comprehensive framework for analyzing neural network robustness
through verification and adversarial testing.
"""

__version__ = "0.1.5"
__author__ = "ADA Research Group"

# Import main components for easy access
# Make key classes available at package level for easier imports
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
    # These will be imported if available
    from .robustness_experiment_box.dataset_sampler.dataset_sampler import DatasetSampler
    from .robustness_experiment_box.epsilon_value_estimator.epsilon_value_estimator import EpsilonValueEstimator
    from .robustness_experiment_box.verification_module.verification_module import VerificationModule

# Check for autoattack availability
try:
    import importlib.util
    HAS_AUTOATTACK = importlib.util.find_spec("pyautoattack") is not None
except ImportError:
    HAS_AUTOATTACK = False
    
# Warn if autoattack is not available
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
]
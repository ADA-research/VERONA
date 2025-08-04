"""
ADA-VERONA: Neural Network Robustness Analysis Framework

A comprehensive framework for analyzing neural network robustness
through verification and adversarial testing.
"""

__version__ = "0.1.7"
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
    
# Check for auto-verify availability and load plugin
try:
    from .robustness_experiment_box.verification_module.plugins.auto_verify_plugin import (
        create_auto_verify_verifier,
        detect_auto_verify,
        get_auto_verify_plugin,
        list_auto_verify_verifiers,
    )
    HAS_AUTO_VERIFY = detect_auto_verify()
    
    if HAS_AUTO_VERIFY:
        # Initialize the plugin to discover verifiers
        _plugin = get_auto_verify_plugin()
        AUTO_VERIFY_VERIFIERS = _plugin.get_available_verifiers()
    else:
        AUTO_VERIFY_VERIFIERS = []
        
except ImportError:
    HAS_AUTO_VERIFY = False
    AUTO_VERIFY_VERIFIERS = []
    
    def list_auto_verify_verifiers():
        return []
    
    def create_auto_verify_verifier(*args, **kwargs):
        return None
    
# Warn if autoattack is not available
if not HAS_AUTOATTACK:
    import warnings
    warnings.warn(
        "PyAutoAttack not found. Some adversarial attack features will be limited. "
        "To install: pip install pyautoattack",
        stacklevel=2
    )

# Log auto-verify status
if HAS_AUTO_VERIFY:
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Auto-verify detected! Available verifiers: {AUTO_VERIFY_VERIFIERS}")
else:
    import warnings
    warnings.warn(
        "Auto-verify not found. Verification features will be limited to attacks. "
        "To enable: install auto-verify in the same environment",
        stacklevel=2
    )

__all__ = [
    "__version__",
    "__author__",
    "HAS_AUTOATTACK",
    "HAS_AUTO_VERIFY", 
    "AUTO_VERIFY_VERIFIERS",
    "analysis",
    "database",
    "dataset_sampler",
    "epsilon_value_estimator",
    "util",
    "verification_module",
    "DatasetSampler",
    "EpsilonValueEstimator",
    "VerificationModule",
    "list_auto_verify_verifiers",
    "create_auto_verify_verifier",
]
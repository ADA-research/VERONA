"""
Auto-verify plugin for ada-verona.

This module provides automatic detection and integration with the auto-verify
framework when it's available in the same environment.
"""

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ada_verona.robustness_experiment_box.database.verification_context import VerificationContext
from ada_verona.robustness_experiment_box.database.verification_result import CompleteVerificationData
from ada_verona.robustness_experiment_box.verification_module.verification_module import VerificationModule

logger = logging.getLogger(__name__)


def detect_auto_verify() -> bool:
    """
    Detect if auto-verify is available in the current environment.
    
    Returns:
        bool: True if auto-verify is available, False otherwise.
    """
    try:
        spec = importlib.util.find_spec("autoverify")
        return spec is not None
    except ImportError:
        return False


class AutoVerifyPlugin:
    """
    Plugin for integrating auto-verify verifiers with ada-verona.
    
    This class handles the detection, loading, and management of auto-verify
    verifiers, making them available through ada-verona's verification interface.
    """
    
    def __init__(self):
        self.available = detect_auto_verify()
        self._autoverify = None
        self._verifiers = {}
        
        if self.available:
            self._load_auto_verify()
    
    def _load_auto_verify(self):
        """Load the auto-verify module and discover available verifiers."""
        try:
            self._autoverify = importlib.import_module("autoverify")
            self._discover_verifiers()
            logger.info("Auto-verify plugin loaded successfully")
        except ImportError as e:
            logger.warning(f"Failed to load auto-verify: {e}")
            self.available = False
    
    def _discover_verifiers(self):
        """Discover available auto-verify verifiers."""
        if not self._autoverify:
            return
            
        try:
            # Import verifier utility functions
            verifiers_module = importlib.import_module("autoverify.util.verifiers")
            
            # Get all available verifiers
            verifier_names = verifiers_module.get_all_complete_verifier_names()
            
            for name in verifier_names:
                try:
                    verifier_class = verifiers_module.verifier_from_name(name)
                    self._verifiers[name] = verifier_class
                    logger.debug(f"Discovered auto-verify verifier: {name}")
                except Exception as e:
                    logger.warning(f"Failed to load verifier {name}: {e}")
                    
        except ImportError as e:
            logger.warning(f"Failed to discover auto-verify verifiers: {e}")
    
    def get_available_verifiers(self) -> List[str]:
        """
        Get list of available auto-verify verifiers.
        
        Returns:
            List[str]: Names of available verifiers.
        """
        return list(self._verifiers.keys()) if self.available else []
    
    def create_verifier_module(self, verifier_name: str, timeout: float = 600, 
                             config: Optional[Path] = None, **kwargs) -> Optional['AutoVerifyVerifierModule']:
        """
        Create an ada-verona compatible verification module for an auto-verify verifier.
        
        Args:
            verifier_name: Name of the auto-verify verifier to use
            timeout: Verification timeout in seconds
            config: Optional configuration file path
            **kwargs: Additional arguments passed to the verifier
            
        Returns:
            AutoVerifyVerifierModule or None if verifier not available
        """
        if not self.available or verifier_name not in self._verifiers:
            logger.warning(f"Auto-verify verifier '{verifier_name}' not available")
            return None
            
        verifier_class = self._verifiers[verifier_name]
        
        try:
            # Create the auto-verify verifier instance
            verifier_instance = verifier_class(**kwargs)
            
            # Wrap it in ada-verona compatible module
            return AutoVerifyVerifierModule(verifier_instance, timeout, config)
            
        except Exception as e:
            logger.error(f"Failed to create verifier module for {verifier_name}: {e}")
            return None


class AutoVerifyVerifierModule(VerificationModule):
    """
    Ada-verona verification module that wraps auto-verify verifiers.
    
    This adapter class bridges the gap between auto-verify's verifier interface
    and ada-verona's VerificationModule interface.
    """
    
    def __init__(self, auto_verify_verifier: Any, timeout: float, config: Optional[Path] = None):
        """
        Initialize the adapter module.
        
        Args:
            auto_verify_verifier: An instance of an auto-verify verifier
            timeout: Verification timeout in seconds
            config: Optional configuration file path
        """
        self.auto_verify_verifier = auto_verify_verifier
        self.timeout = timeout
        self.config = config
        self.name = f"AutoVerify-{getattr(auto_verify_verifier, 'name', type(auto_verify_verifier).__name__)}"
    
    def verify(self, verification_context: VerificationContext, epsilon: float) -> Union[str, CompleteVerificationData]:
        """
        Verify the robustness using the auto-verify verifier.
        
        Args:
            verification_context: The verification context containing network, data point, etc.
            epsilon: The perturbation magnitude
            
        Returns:
            Verification result (SAT, UNSAT, TIMEOUT, ERR) or CompleteVerificationData
        """
        try:
            # Extract image and create VNNLib property
            image = verification_context.data_point.data.reshape(-1).detach().numpy()
            vnnlib_property = verification_context.property_generator.create_vnnlib_property(
                image, verification_context.data_point.label, epsilon
            )
            
            # Save the property file
            verification_context.save_vnnlib_property(vnnlib_property)
            
            # Create verification instance for auto-verify
            from autoverify.util.verification_instance import VerificationInstance
            
            verification_instance = VerificationInstance(
                network=verification_context.network.path,
                property=vnnlib_property.path,
                timeout=int(self.timeout)
            )
            
            # Run verification using auto-verify
            if self.config:
                result = self.auto_verify_verifier.verify_instance(
                    verification_instance, config=self.config
                )
            else:
                result = self.auto_verify_verifier.verify_instance(verification_instance)
            
            # Process the result
            if result is None:
                logger.error("Auto-verify verifier returned None")
                return "ERR"
                
            # Handle Result[Ok, Err] types from auto-verify
            if hasattr(result, 'is_ok') and hasattr(result, 'is_err'):
                if result.is_ok():
                    outcome = result.unwrap()
                    return CompleteVerificationData(
                        result=outcome.result,
                        took=outcome.took,
                        counter_example=outcome.counter_example,
                        err=outcome.err,
                        stdout=outcome.stdout
                    )
                elif result.is_err():
                    error_data = result.unwrap_err()
                    logger.info(f"Auto-verify verification error: {error_data}")
                    return CompleteVerificationData(
                        result="ERR",
                        took=getattr(error_data, 'took', 0.0),
                        err=str(error_data)
                    )
            
            # Fallback for unexpected result types
            logger.warning(f"Unexpected result type from auto-verify: {type(result)}")
            return "ERR"
            
        except Exception as e:
            logger.error(f"Error during auto-verify verification: {e}")
            return CompleteVerificationData(
                result="ERR",
                took=0.0,
                err=str(e)
            )


# Global plugin instance
_auto_verify_plugin = None


def get_auto_verify_plugin() -> AutoVerifyPlugin:
    """
    Get the global auto-verify plugin instance.
    
    Returns:
        AutoVerifyPlugin: The plugin instance
    """
    global _auto_verify_plugin
    if _auto_verify_plugin is None:
        _auto_verify_plugin = AutoVerifyPlugin()
    return _auto_verify_plugin


def list_auto_verify_verifiers() -> List[str]:
    """
    List all available auto-verify verifiers.
    
    Returns:
        List[str]: Names of available verifiers
    """
    plugin = get_auto_verify_plugin()
    return plugin.get_available_verifiers()


def create_auto_verify_verifier(verifier_name: str, timeout: float = 600, 
                               config: Optional[Path] = None, **kwargs) -> Optional[AutoVerifyVerifierModule]:
    """
    Create an auto-verify verifier module.
    
    Args:
        verifier_name: Name of the verifier
        timeout: Verification timeout
        config: Optional configuration file
        **kwargs: Additional verifier arguments
        
    Returns:
        AutoVerifyVerifierModule or None if not available
    """
    plugin = get_auto_verify_plugin()
    return plugin.create_verifier_module(verifier_name, timeout, config, **kwargs) 
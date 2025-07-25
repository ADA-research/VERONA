"""
Plugin system for integrating external verification tools.

This module provides a plugin architecture that allows ada-verona to automatically
detect and integrate with external verification frameworks like auto-verify.
"""

from .auto_verify_plugin import AutoVerifyPlugin, detect_auto_verify

__all__ = ["AutoVerifyPlugin", "detect_auto_verify"] 
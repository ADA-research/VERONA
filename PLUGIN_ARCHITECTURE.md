# ADA-VERONA Plugin Architecture

## Overview

This document describes the plugin architecture that enables **ada-verona** to work independently while optionally integrating with **auto-verify** when it is detected in the virtual environment. The system provides:

- **Independent Operation**: ada-verona works without auto-verify
- **Automatic Detection**: Integration when auto-verify is present
- **Dual Environment Support**: Choose between conda and Python venv + uv
- **Plugin System**: Extensible architecture for future integrations

## Architecture

### Plugin Detection & Loading

```python
import ada_verona

# Check availability
print(f"Auto-verify available: {ada_verona.HAS_AUTO_VERIFY}")
print(f"Available verifiers: {ada_verona.AUTO_VERIFY_VERIFIERS}")

# Create verifier through plugin
verifier = ada_verona.create_auto_verify_verifier(
    verifier_name="nnenum",
    timeout=600
)
```

### Environment Management Strategy

The system supports **dual environment management**:

1. **Traditional conda-based** (existing auto-verify approach)
2. **Modern venv + uv based** (new simplified approach)

## Getting Started

### 1. Basic Usage (ada-verona only)

```python
from ada_verona.robustness_experiment_box.verification_module.attack_estimation_module import AttackEstimationModule
from ada_verona.robustness_experiment_box.verification_module.attacks.pgd_attack import PGDAttack

# Use attack-based verification
verifier = AttackEstimationModule(attack=PGDAttack(number_iterations=10, step_size=0.01))
```

### 2. With Auto-Verify Integration

```python
import ada_verona

# Check if auto-verify is available
if ada_verona.HAS_AUTO_VERIFY:
    # Use formal verification
    verifier = ada_verona.create_auto_verify_verifier("nnenum", timeout=300)
else:
    # Fallback to attacks
    from ada_verona.robustness_experiment_box.verification_module.attack_estimation_module import AttackEstimationModule
    from ada_verona.robustness_experiment_box.verification_module.attacks.pgd_attack import PGDAttack
    verifier = AttackEstimationModule(attack=PGDAttack(number_iterations=10, step_size=0.01))
```

### 3. Complete Example

```python
import ada_verona
from ada_verona.robustness_experiment_box.epsilon_value_estimator.binary_search_epsilon_value_estimator import BinarySearchEpsilonValueEstimator

# Create verifier (auto-detects what's available)
if ada_verona.HAS_AUTO_VERIFY and "nnenum" in ada_verona.AUTO_VERIFY_VERIFIERS:
    verifier = ada_verona.create_auto_verify_verifier("nnenum", timeout=300)
    print(f"Using formal verification: {verifier.name}")
else:
    # Use attack-based approach
    from ada_verona.robustness_experiment_box.verification_module.attack_estimation_module import AttackEstimationModule
    from ada_verona.robustness_experiment_box.verification_module.attacks.pgd_attack import PGDAttack
    verifier = AttackEstimationModule(attack=PGDAttack(number_iterations=10, step_size=0.01))
    print(f"Using attack-based verification: {verifier.name}")

# Use in epsilon value estimator
epsilon_estimator = BinarySearchEpsilonValueEstimator(
    epsilon_value_list=[0.001, 0.005, 0.01],
    verifier=verifier
)
```

## Configuration System

### Auto-Verify Configuration

Auto-verify now supports configuration for environment management:

```bash
# Show current configuration
auto-verify config show

# Set environment strategy  
auto-verify config set-env venv    # Use Python venv + uv
auto-verify config set-env conda   # Use conda (traditional)
auto-verify config set-env auto    # Auto-detect (prefer venv if uv available)

# Create example config file
auto-verify config example
```

### Configuration File

Create `~/.config/autoverify/autoverify.toml`:

```toml
# Environment management strategy
env_strategy = "venv"  # or "conda" or "auto"

# Runtime preferences
prefer_gpu = true
default_timeout = 600

# Advanced options
allow_conda_fallback = true
require_uv = false
```

## Environment Management

### Python venv + uv (Recommended)

**Benefits:**
- Simpler path management
- Faster package installation with uv
- More predictable dependencies
- Better integration with modern Python tooling
- User-controllable virtual environments

**Installation:**
```bash
# Install with venv (requires uv for best experience)
pip install uv
auto-verify config set-env venv
auto-verify install nnenum abcrown verinet
```

**Directory Structure:**
```
~/.local/share/autoverify-venv/
├── verifiers/
│   ├── nnenum/
│   │   ├── venv/          # Python virtual environment
│   │   ├── tool/          # Git repository
│   │   └── activate.sh    # Activation script
│   ├── abcrown/
│   └── verinet/
└── config.toml
```

### Conda (Traditional)

**Benefits:**
- Handles complex dependencies (e.g., CUDA, system libraries)
- Proven stability for scientific computing
- Better GPU support for some verifiers

**Installation:**
```bash
# Install with conda (default for existing users)
auto-verify config set-env conda
auto-verify install nnenum abcrown
```

## Technical Details

### Plugin Architecture Components

1. **Detection Module** (`auto_verify_plugin.py`)
   - Automatically detects auto-verify availability
   - Discovers installed verifiers
   - Handles version compatibility

2. **Adapter Layer** (`AutoVerifyVerifierModule`)
   - Bridges auto-verify and ada-verona interfaces
   - Handles result format conversion
   - Manages error handling and timeouts

3. **Environment Managers**
   - **Conda Manager**: Traditional conda-based installations
   - **Venv Manager**: Modern venv + uv based installations
   - **Configuration System**: User choice and auto-detection

### Adapter Implementation

The `AutoVerifyVerifierModule` wraps auto-verify verifiers to work with ada-verona:

```python
class AutoVerifyVerifierModule(VerificationModule):
    def verify(self, verification_context: VerificationContext, epsilon: float):
        # Convert ada-verona format to auto-verify format
        image = verification_context.data_point.data.reshape(-1).detach().numpy()
        vnnlib_property = verification_context.property_generator.create_vnnlib_property(
            image, verification_context.data_point.label, epsilon
        )
        
        # Create auto-verify verification instance
        verification_instance = VerificationInstance(
            network=verification_context.network.path,
            property=vnnlib_property.path,
            timeout=int(self.timeout)
        )
        
        # Run auto-verify verifier
        result = self.auto_verify_verifier.verify_instance(verification_instance)
        
        # Convert result back to ada-verona format
        return self._convert_result(result)
```

## Installation Methods

### Option 1: Independent ada-verona

```bash
pip install ada-verona
# Only attack-based verification available
```

### Option 2: Full Integration

```bash
pip install ada-verona
pip install auto-verify  # or your fork
# Both attack and formal verification available
```

### Option 3: Development Setup

```bash
# Clone both repositories
git clone <ada-verona-repo>
git clone <auto-verify-fork>

# Install in development mode
cd ada-verona && pip install -e .
cd ../auto-verify && pip install -e .

# Configure environment management
auto-verify config set-env venv
```

## Testing the Integration

### Test Script

```python
"""Test the plugin integration."""

import ada_verona

def test_plugin_system():
    print("Plugin System Status:")
    print(f"  Auto-verify available: {ada_verona.HAS_AUTO_VERIFY}")
    print(f"  PyAutoAttack available: {ada_verona.HAS_AUTOATTACK}")
    
    if ada_verona.HAS_AUTO_VERIFY:
        print(f"  Available verifiers: {ada_verona.AUTO_VERIFY_VERIFIERS}")
        
        # Try creating a verifier
        if ada_verona.AUTO_VERIFY_VERIFIERS:
            verifier_name = ada_verona.AUTO_VERIFY_VERIFIERS[0]
            verifier = ada_verona.create_auto_verify_verifier(verifier_name, timeout=60)
            print(f"  Successfully created: {verifier.name if verifier else 'Failed'}")
    
    print("\nReady to run robustness experiments!")

if __name__ == "__main__":
    test_plugin_system()
```

### Example Execution

```bash
cd scripts
python create_robustness_dist_with_auto_verify.py
```

Expected output:
```
ADA-VERONA Plugin System Demo
==================================================
Auto-verify available: True
PyAutoAttack available: True

Available auto-verify verifiers:
  • nnenum
  • abcrown
  • verinet

Usage example:
  verifier = ada_verona.create_auto_verify_verifier('nnenum', timeout=600)
  # Use verifier in BinarySearchEpsilonValueEstimator or other components

Running example with nnenum...
=== ADA-VERONA + Auto-Verify Integration Example ===
Auto-verify detected! Available verifiers: ['nnenum', 'abcrown', 'verinet']
Creating auto-verify verifier: nnenum
Successfully created verifier module: AutoVerify-nnenum
Starting robustness distribution computation with formal verification...
```

## Migration Guide

### For Existing VERONA Users

1. **Immediate**: Everything continues to work as before
2. **Optional**: Install auto-verify for formal verification
3. **Recommended**: Configure environment strategy

```bash
# Check current setup
python -c "import ada_verona; print(f'Auto-verify: {ada_verona.HAS_AUTO_VERIFY}')"

# Install auto-verify (optional)
pip install auto-verify  # or your preferred fork

# Configure environment (recommended)
auto-verify config set-env venv
```

### For Existing Auto-Verify Users

1. **Install ada-verona**: Adds robust experiment framework
2. **Choose environment**: Keep conda or migrate to venv
3. **Enjoy integration**: Verifier access

```bash
# Install ada-verona
pip install ada-verona

# Check integration
python -c "import ada_verona; print(ada_verona.AUTO_VERIFY_VERIFIERS)"

# Configure environment strategy
auto-verify config show
auto-verify config set-env auto  # or venv/conda
```

## Benefits Summary

### For Ada-Verona Users
- **Independence**: Works without external dependencies
- **Enhanced Capabilities**: Optional formal verification
- **Integration**: Automatic detection and setup
- **Unified Interface**: Same API regardless of backend

### For Auto-Verify Users  
- **Rich Experiment Framework**: Comprehensive robustness analysis
- **Modern Environment Management**: Python venv + uv option
- **Simplified Setup**: Easier installation and management
- **Better Path Management**: Cleaner, more predictable structure

### For Developers
- **Plugin Architecture**: Easy to extend with new verifiers
- **Clean Separation**: Independent codebases with optional integration
- **Dual Environment Support**: Choose the best tool for each use case
- **Configuration System**: User-controllable behavior

## Future Extensions

The plugin architecture enables future integrations:

- **Other Verification Tools**: Easy to add new formal verifiers
- **Cloud Verifiers**: Remote verification services
- **Distributed Computing**: Cluster-based verification
- **Custom Verifiers**: User-defined verification methods

---

**Ready to start? Run the example script and explore the integration!** 
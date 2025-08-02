# VERification Of Neural Architectures (VERONA)

The ada-verona package simplifies your experiment pipeline for performing local robustness verification on your networks and datasets. 
The entire package is class-based, which means that extending the existing configurations is accessible and easy. 
With one script it is possible to run an entire experiment with various networks, images and perturbation magnitudes (epsilons). 

The package can be used to create robustness distributions [Bosman, Berger, Hoos and van Rijn, 2025](https://jair.org/index.php/jair/article/view/18403), as well as empirically measuring robustness of networks using adversarial attacks and verify networks with formal verification tools using the [auto-verify](https://github.com/ADA-research/auto-verify) plugin that currently supports [nnenum](https://github.com/stanleybak/nnenum), [AB-Crown](https://github.com/Verified-Intelligence/alpha-beta-CROWN), [VeriNet](https://github.com/vas-group-imperial/VeriNet), and [Oval-Bab](https://github.com/oval-group/oval-bab).



We plan to add more verifiers to the plugin in the future.

## Important Notice: ada-auto-verify Plugin System Update

**The ada-auto-verify package needs to be updated to be fully usable as a plugin system.** We are currently coordinating with the maintainers to implement the necessary changes. The updated release is projected to happen within **1-2 weeks** (approx. until 2025-08-15).

Until the update is available, you can still use ada-verona for:
- Empirical robustness measurement using adversarial attacks (FGSM, PGD, AutoAttack)
- Upper bound estimation for robustness distributions
- All core functionality without formal verification

For formal verification capabilities, please check back for the updated ada-auto-verify package release.

## Table of Contents
- [VERification Of Neural Architectures (VERONA)](#verification-of-neural-architectures-verona)
  - [Important Notice: ada-auto-verify Plugin System Update](#important-notice-ada-auto-verify-plugin-system-update)
  - [Table of Contents](#table-of-contents)
  - [Authors](#authors)
  - [Installation](#installation)
    - [Installation Options](#installation-options)
    - [Installation Variants Explained](#installation-variants-explained)
  - [Documentation](#documentation)
  - [Getting Started: Guide and Example Scripts](#getting-started-guide-and-example-scripts)
  - [Command-Line Interface](#command-line-interface)
    - [Running Experiments](#running-experiments)
    - [Using Auto-Verify Verifiers](#using-auto-verify-verifiers)
    - [Listing Available Components](#listing-available-components)
  - [Experiment Folder and CLI Usage](#experiment-folder-and-cli-usage)
    - [Directory Structure Options](#directory-structure-options)
    - [Command-Line Usage](#command-line-usage)
    - [Examples](#examples)
  - [Verification](#verification)
    - [Auto-Verify Plugin System](#auto-verify-plugin-system)
      - [Using the Plugin System](#using-the-plugin-system)
      - [Installation](#installation-1)
    - [Custom Verifiers](#custom-verifiers)
  - [How to Add Your Own Verifier](#how-to-add-your-own-verifier)
  - [Available Attacks](#available-attacks)
    - [Custom Attacks](#custom-attacks)
  - [Datasets](#datasets)
  - [Related Papers and Citation](#related-papers-and-citation)
    - [Robustness distributions](#robustness-distributions)
    - [Upper bounds to robustness distributions](#upper-bounds-to-robustness-distributions)
    - [Per-class robustness distributions](#per-class-robustness-distributions)
  - [Acknowledgements](#acknowledgements)
  - [Contributing](#contributing)

## Authors

This package was created and is maintained by members the [ADA Research Group](https://adaresearch.wordpress.com/about/), which focuses on the development of AI techniques that complement human intelligence and automated algorithm design. The current core team includes:

- **Annelot Bosman** (LIACS, Leiden University)
- **Aaron Berger** (TU Delft)
- **Hendrik S. Baacke** (AIM, RWTH Aachen University)
- **Holger H. Hoos** (AIM, RWTH Aachen University)
- **Jan van Rijn** (LIACS, Leiden University)

## Installation

### Installation Options

ADA-VERONA offers several installation options to meet different needs:

```bash
# Default installation (includes GPU support and AutoAttack) using uv for faster installation (recommended)
uv pip install ada-verona

# Default installation (includes GPU support and AutoAttack)
pip install ada-verona

# CPU-only installation without GPU or AutoAttack
pip install "ada-verona[cpu]"

# Development installation with testing tools
pip install "ada-verona[dev]"
```

> **Note:** When installing with extras (like `[cpu]` or `[dev]`), quotes are required around the package name to prevent shell expansion of the square brackets.

### Installation Variants Explained

| Variant | Description | Best For |
|---------|-------------|----------|
| `ada-verona` | Full installation with GPU support and AutoAttack | Most users, recommended default |
| `ada-verona[cpu]` | CPU-only version with minimal dependencies | Basic usage, restricted environments |
| `ada-verona[dev]` | Full installation plus development tools | Contributors, developers |

This package was tested only on Python version 3.10 and we cannot guarantee it working on any other Python version.

## Documentation
In case you have more questions, please refer to the [VERONA documentation](https://deepwiki.com/ADA-research/VERONA).

## Getting Started: Guide and Example Scripts

To help you get up and running with ada-verona, we provide a comprehensive tutorial notebook and a collection of practical example scripts:

- **Main Guide:**
  - The primary resource for learning how to use ada-verona is the Jupyter notebook found in the [`notebooks`](./notebooks/) folder. This tutorial notebook offers an overview of the package components, step-by-step instructions, and practical demonstrations of typical workflows. We highly recommend starting here to understand the core concepts and capabilities of the package.

- **Quick-Start Example Scripts:**
  - The [`scripts`](./scripts/) folder contains a variety of example scripts designed to help you get started quickly with ada-verona. These scripts cover common use cases and can be run directly (from within the `scripts` folder) to see how to perform tasks such as:
    - Running VERONA with a custom dataset and ab-crown ([`create_robustness_distribution_from_test_dataset.py`](./scripts/create_robustness_distribution_from_test_dataset.py)).
    - Loading a PyTorch dataset and running VERONA with one-to-any or one-to-one verification ([`create_robustness_dist_on_pytorch_dataset.py`](./scripts/create_robustness_dist_on_pytorch_dataset.py)).
    - Distributing jobs across multiple nodes using SLURM for large-scale experiments ([`multiple_jobs`](./scripts/multiple_jobs/) folder), including distributing tasks over CPU and GPU for different verifiers in the same experiment.
    - Using auto-verify integration ([`create_robustness_dist_with_auto_verify.py`](./scripts/create_robustness_dist_with_auto_verify.py)).

The notebook is your main entry point for learning and understanding the package, while the scripts serve as practical templates and quick-start resources for your own experiments.

## Command-Line Interface

ADA-VERONA provides a command-line interface (CLI) for running robustness experiments without writing Python code. The CLI allows you to:

- Run robustness experiments with various networks, datasets, and verification methods
- Use auto-verify verifiers through virtual environment names
- List available components (verifiers, auto-verify environments)

### Running Experiments

The basic command to run a robustness experiment is:

```bash
ada-verona run --networks <path-to-networks> [options]
```

Key options include:

| Option | Description | Default |
|--------|-------------|---------|
| `--name` | Experiment name | `robustness_experiment` |
| `--networks` | Directory with network files (.onnx) | (required) |
| `--dataset` | Dataset to use (mnist, cifar10) | `mnist` |
| `--verifier` | Verification method (pgd, fgsm, auto-verify) | `pgd` |
| `--property` | Property type (one2any, one2one) | `one2any` |
| `--epsilons` | List of epsilon values to search | `[0.001, 0.005, 0.01, 0.05]` |
| `--sample-size` | Number of samples per network | `10` |
| `--sample-correct` | Sample only correctly predicted inputs | `false` |

### Using Auto-Verify Verifiers

When using auto-verify verifiers, you can specify either the verifier name or the virtual environment name:

```bash
# Using verifier name
ada-verona run --networks <path> --verifier auto-verify --auto-verify-name nnenum

# Using virtual environment name (venv)
ada-verona run --networks <path> --verifier auto-verify --auto-verify-venv nnenum
```

The `--auto-verify-venv` option allows you to specify which auto-verify virtual environment to use, which is particularly useful when you have multiple versions of the same verifier installed (e.g., different commits).

### Listing Available Components

You can list available components with:

```bash
# List all components
ada-verona list

# List available verifiers
ada-verona list --verifiers

# List auto-verify status and available verifiers
ada-verona list --auto-verify

# List auto-verify virtual environments
ada-verona list --auto-verify-venvs
``
```

## Experiment Folder and CLI Usage

ADA-VERONA uses the following directory structure for experiments:

**Note**: You must provide ONNX or torch network files in the networks directory. ada-verona will create directories automatically, but you need to supply your own network models.
```
experiment/
|-- data/
|   |-- labels.csv
|   |-- images/
|       |-- mnist_0.npy
|       |-- mnist_1.npy
|       |-- ...
|-- networks/
|   |-- mnist-net_256x2.onnx
|   |-- mnist-net_256x4.onnx
|   |-- ...
```

### Directory Structure Options

You have two main options for organizing your experiment:

1. **Default Structure**: Place an `experiment` folder in your working directory with the above structure. The CLI will use this by default.

2. **Custom Paths**: Specify custom paths for networks, data, and output using command-line arguments.

### Command-Line Usage

Basic usage with default paths (expects networks in `./experiment/networks/`):
```bash
ada-verona run --name PGDattack --dataset mnist --verifier pgd --epsilons 0.05 --sample-correct
```

Specifying custom paths:
```bash
# relative paths
ada-verona run --name my_experiment --networks ./my_models --output ./results --data-dir ./datasets

# absolute paths
ada-verona run --name my_experiment --networks /path/to/models --output /path/to/results --data-dir /path/to/data
```

### Examples

Basic experiment with PGD attack (assumes existing experiment folder and networks in experiment/networks/):
```bash
ada-verona run --networks --name pgd_experiment --verifier pgd --epsilons 0.001 0.005 0.01
```

Using auto-verify with a specific virtual environment:
```bash
ada-verona run --networks ./models --name formal_verification --verifier auto-verify --auto-verify-venv abcrown --timeout 600
```

Customizing dataset and sampling:
```bash
ada-verona run --networks ./models --dataset cifar10 --sample-size 20 --sample-correct
```


## Verification

### Auto-Verify Plugin System

Ada-verona features a plugin architecture that allows seamless integration with [auto-verify](https://github.com/ADA-research/auto-verify) when it's available. This design provides several benefits:

1. **Independence**: Ada-verona works perfectly without auto-verify, using attack-based verification methods for empirical upper bounds.
2. **Automatic Detection**: When auto-verify is installed in the same environment, its verifiers become automatically available
3. **Interface**: The same API works regardless of which verification backend is used

#### Using the Plugin System

To use formal verification tools from auto-verify:

```python
import ada_verona

# Check if auto-verify is available
print(f"Auto-verify available: {ada_verona.HAS_AUTO_VERIFY}")
print(f"Available verifiers: {ada_verona.AUTO_VERIFY_VERIFIERS}")

# Create a verifier if auto-verify is available
if ada_verona.HAS_AUTO_VERIFY and "nnenum" in ada_verona.AUTO_VERIFY_VERIFIERS:
    verifier = ada_verona.create_auto_verify_verifier("nnenum", timeout=300)
    print(f"Using formal verification: {verifier.name}")
else:
    # Fallback to attack-based verification
    from ada_verona.robustness_experiment_box.verification_module.attack_estimation_module import AttackEstimationModule
    from ada_verona.robustness_experiment_box.verification_module.attacks.pgd_attack import PGDAttack
    verifier = AttackEstimationModule(attack=PGDAttack(number_iterations=10, step_size=0.01))
    print(f"Using attack-based verification: {verifier.name}")
```

#### Installation

To enable formal verification through auto-verify:

```bash
# First install ada-verona
pip install ada-verona

# Then install auto-verify
pip install auto-verify

# View the current auto-verify configuration
auto-verify config show

# Configure environment (optional, but recommended)
auto-verify config set-env venv  # Use Python venv + uv (recommended)
# or
auto-verify config set-env conda  # Use conda (traditional)

# Install verifiers
auto-verify install nnenum abcrown
```

For more details, see the [Plugin Architecture documentation](./PLUGIN_ARCHITECTURE.md) and the example script [`create_robustness_dist_with_auto_verify.py`](./scripts/create_robustness_dist_with_auto_verify.py).

### Custom Verifiers

Custom verifiers can be implemented by using the [`VerificationModule`](./src/ada_verona/robustness_experiment_box/verification_module/verification_module.py) interface.

## How to Add Your Own Verifier

You can easily add your own verifier by following these steps:

1. **Implement the `VerificationModule` interface:**
   - Create a new class that inherits from [`VerificationModule`](./src/ada_verona/robustness_experiment_box/verification_module/verification_module.py).
   - Implement the `verify(self, verification_context: VerificationContext, epsilon: float)` method. This method should return either a string (e.g., "SAT", "UNSAT", "ERR") or a `CompleteVerificationData` object.

   Example:
   ```python
   from ada_verona.robustness_experiment_box.verification_module.verification_module import VerificationModule

   class MyCustomVerifier(VerificationModule):
       def verify(self, verification_context, epsilon):
           # Your custom verification logic here
           # Return "SAT", "UNSAT", or a CompleteVerificationData object
           return "UNSAT"
   ```

2. **(Optional) If your verifier wraps an external tool:**
   - Implement the `Verifier` interface in [`verification_runner.py`](./src/ada_verona/robustness_experiment_box/verification_module/verification_runner.py).
   - Then, use the `GenericVerifierModule` to wrap your `Verifier` implementation, which will handle property file management and result parsing for you.

   Example:
   ```python
   from ada_verona.robustness_experiment_box.verification_module.verification_runner import Verifier, GenericVerifierModule

   class MyExternalToolVerifier(Verifier):
       def verify_property(self, network_path, property_path, timeout, config=None):
           # Call your external tool here
           # Return Ok(result) or Err(error_message)
           pass

   my_verifier = MyExternalToolVerifier()
   module = GenericVerifierModule(my_verifier, timeout=60)
   ```

3. **Register and use your verifier in your experiment scripts.**

| Interface/Class         | Purpose                                      | Where to Implement/Use                |
|------------------------|----------------------------------------------|---------------------------------------|
| `VerificationModule`   | Main interface for all verifiers             | Subclass for custom logic             |
| `Verifier`             | For verifiers wrapping external tools        | Subclass if using external binaries   |
| `GenericVerifierModule`| Wraps a `Verifier` for property/result mgmt  | Use if you subclass `Verifier`        |

## Available Attacks

Currently the package implements the following adversarial attack methods:
- **Fast Gradient Sign Method (FGSM)** [Goodfellow et al., 2015](https://arxiv.org/abs/1412.6572)
- **Projected Gradient Descent (PGD)** [Madry et al., 2018](https://arxiv.org/abs/1706.06083)
- **AutoAttack** [Croce and Hein, 2020](https://github.com/fra31/auto-attack). For using AutoAttack the package has to be installed first as described in the AutoAttack repository.

### Custom Attacks

Custom attacks can be implemented by using the [`Attack`](./src/ada_verona/robustness_experiment_box/verification_module/attacks/attack.py) interface.

## Datasets

The package was tested on the MNIST and the CIFAR10 dataset. Example scripts for executing the package on MNIST or a custom dataset can be found in the [`scripts`](./scripts/) folder.

## Related Papers and Citation

This package was created to simplify reproducing and extending the results of two different lines of work of the ADA research group. 

### Robustness distributions 

**Please cite this paper when you have used VERONA in your experiments:**
```bibtex
@article{BosEtAl25,
author = {Annelot W Bosman, Aaron Berger, Holger H Hoos, Jan N van Rijn},
title = {Robustness Distributions in Neural Network Verification},
booktitle = {Journal of Artificial Intelligence Research}.
year = {2025}
}
```
- [Paper link](https://jair.org/index.php/jair/article/view/18403)

A short introduction to the concept of robustness distributions can be found in the following paper:
```bibtex
@article{BosEtAl23,
    author = "Bosman, Annelot W. and Hoos, Holger H. and van Rijn, Jan N.",
    title = "A Preliminary Study of Critical Robustness Distributions in Neural Network Verification",
    year = "2023",
    journal = "6th Workshop on Formal Methods for ML-Enabled Autonomous Systems (FoMLAS) co-located with the 35th International Conference on Computer Aided Verification (CAV 2023)"
}
```
  
- [Paper link](https://ada.liacs.leidenuniv.nl/papers/BosEtAl23.pdf)

### Upper bounds to robustness distributions

The concept of using adversarial attacks to compute upper bounds to robustness distributions is introduced in the following paper:
```bibtex
@inproceedings{bergerEmpiricalAnalysisUpper,
  title = {Empirical {{Analysis}} of {{Upper Bounds}} of {{Robustness Distributions}} Using {{Adversarial Attacks}}},
  booktitle = {{{THE 19TH LEARNING AND IN}}℡{{LIGENT OPTIMIZATION CONFERENCE}}},
  author = {Berger, Aaron and Eberhardt, Nils and Bosman, Annelot Willemijn and Duwe, Henning and family=Rijn, given=Jan N., prefix=van, useprefix=true and Hoos, Holger}
}
```
- [Paper link](https://openreview.net/forum?id=jsfqoRrsjy)

### Per-class robustness distributions

The concept of per-class robustness distributions is introduced in the following paper:
```bibtex
@inproceedings{BosEtAl24,
    author = {Bosman, Annelot W. and Münz, Anna L. and Hoos, Holger H. and van Rijn, Jan N.},
    title = {{A Preliminary Study to Examining Per-Class Performance Bias via Robustness Distributions}},
    year = {2024},
    booktitle = {The 7th International Symposium on AI Verification (SAIV) co-located with the 36th International Conference on Computer Aided Verification (CAV 2024)}
}
```
- [Paper link](https://ada.liacs.leidenuniv.nl/papers/BosEtAl24.pdf)

## Acknowledgements

This package makes use of the following tools and libraries:

- **AutoAttack** ([GitHub](https://github.com/fra31/auto-attack))
    - F. Croce and M. Hein, "Mind the box: l_1 -APGD for sparse adversarial attacks on image classifiers," in International Conference on Machine Learning, PMLR, 2021, pp. 2201–2211. [Online]. Available: http://proceedings.mlr.press/v139/croce21a.html
    - F. Croce and M. Hein, "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks," in International conference on machine learning, PMLR, 2020, pp. 2206–2216. [Online]. Available: https://proceedings.mlr.press/v119/croce20b.html

- **auto-verify** ([GitHub](https://github.com/ADA-research/auto-verify))
    - For integrating verifiers [nnenum](https://github.com/stanleybak/nnenum), [AB-Crown](https://github.com/Verified-Intelligence/alpha-beta-CROWN), [VeriNet](https://github.com/vas-group-imperial/VeriNet), and [Oval-Bab](https://github.com/oval-group/oval-bab). Please refer to the [auto-verify documentation](https://github.com/ADA-research/auto-verify) for details about auto-verify.

We thank the authors and maintainers of these projects, as well as the authors and maintainers of the verifiers for their contributions to the robustness research community.

## Contributing

We welcome contributions to the ada-verona package! If you find a bug, have a feature request, or want to contribute code, please follow these steps:

1. **Create an Issue:** Before starting work on a new feature or bug, please create an issue in the [GitHub repository](https://github.com/ADA-research/VERONA/issues) to discuss your plans. This helps us coordinate contributions and avoid duplicate work.

2. **Fork the Repository:** Create a personal copy of the repository on GitHub.

3. **Create a Branch:** Create a new branch for your feature or bug fix.

4. **Make Changes:** Implement your changes in the codebase.

5. **Test Your Changes:** Ensure that your changes do not break existing functionality and that new features work as intended.

6. **Commit Your Changes:** Write clear and concise commit messages describing your changes.

7. **Push to Your Fork:** Push your changes to your forked repository.

8. **Create a Pull Request:** Open a pull request against the main repository, describing your changes and why they are needed.

9. **Review Process:** Your pull request will be reviewed by at least one of the maintainers. They may request changes or provide feedback.

Thank you for contributing! 
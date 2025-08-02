# VERification Of Neural Architectures (VERONA)

The ada-verona package simplifies your experiment pipeline for performing local robustness verification on your networks and datasets. 
The entire package is class-based, which means that extending the existing configurations is accessible and easy. 
With one script it is possible to run an entire experiment with various networks, images and perturbation magnitudes (epsilons). 
The package can be used to create robustness distributions [Bosman, Berger, Hoos and van Rijn, 2023](https://ada.liacs.leidenuniv.nl/papers/BosEtAl25.pdf) and per-class robustness distributions [Bosman et al., 2024](https://ada.liacs.leidenuniv.nl/papers/BosEtAl24.pdf).

## Table of Contents
- [VERification Of Neural Architectures (VERONA)](#verification-of-neural-architectures-verona)
  - [Table of Contents](#table-of-contents)
  - [Authors](#authors)
  - [Installation](#installation)
    - [Installation Options](#installation-options)
    - [Installation Variants Explained](#installation-variants-explained)
  - [Documentation](#documentation)
  - [Getting Started: Guide and Example Scripts](#getting-started-guide-and-example-scripts)
  - [Experiment Folder](#experiment-folder)
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
    - [Per-class robustness distributions](#per-class-robustness-distributions)
    - [Upper bounds to robustness distributions](#upper-bounds-to-robustness-distributions)
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

## Experiment Folder
The following structure for an experiment folder is currently supported:

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

The images can be placed in it optionally, if one wants to execute the experiments using custom data. Otherwise, PyTorch Datasets can be used too and no image folder has to be created.

## Verification

### Auto-Verify Plugin System

Ada-verona features a plugin architecture that allows seamless integration with [auto-verify](https://github.com/ADA-research/auto-verify) when it's available. This design provides several benefits:

1. **Independence**: Ada-verona works perfectly without auto-verify, using attack-based verification methods
2. **Automatic Detection**: When auto-verify is installed in the same environment, its verifiers become automatically available
3. **Unified Interface**: The same API works regardless of which verification backend is used

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

This package was created to simplify reproducing and extending the results of two different lines of work of the ADA research group. Please consider citing these works when using this package for your research. 

### Robustness distributions 
Please cite this paper when you have used VERONA in your experiments. 
```bibtex
@article{BosEtAl25,
author = {Annelot W Bosman, Aaron Berger, Holger H Hoos, Jan N van Rijn},
title = {Robustness Distributions in Neural Network Verification},
booktitle = {Journal of Artificial Intelligence Research}.
year = {2025}
}
```
  
- [Paper link](https://ada.liacs.leidenuniv.nl/papers/BosEtAl23.pdf)
- An extended version of this work is currently under review. 


### Per-class robustness distributions
```bibtex
@inproceedings{BosEtAl24,
    author = {Bosman, Annelot W. and Münz, Anna L. and Hoos, Holger H. and van Rijn, Jan N.},
    title = {{A Preliminary Study to Examining Per-Class Performance Bias via Robustness Distributions}},
    year = {2024},
    booktitle = {The 7th International Symposium on AI Verification (SAIV) co-located with the 36th International Conference on Computer Aided Verification (CAV 2024)},
    url = {https://ada.liacs.leidenuniv.nl/papers/BosEtAl24.pdf}
}
```

### Upper bounds to robustness distributions
```bibtex
@inproceedings{bergerEmpiricalAnalysisUpper,
  title = {Empirical {{Analysis}} of {{Upper Bounds}} of {{Robustness Distributions}} Using {{Adversarial Attacks}}},
  booktitle = {{{THE 19TH LEARNING AND IN}}℡{{LIGENT OPTIMIZATION CONFERENCE}}},
  author = {Berger, Aaron and Eberhardt, Nils and Bosman, Annelot Willemijn and Duwe, Henning and family=Rijn, given=Jan N., prefix=van, useprefix=true and Hoos, Holger},
  url = {https://openreview.net/forum?id=jsfqoRrsjy}
}
```

## Acknowledgements

This package makes use of the following tools and libraries:

- **AutoAttack** ([GitHub](https://github.com/fra31/auto-attack))
    - F. Croce and M. Hein, "Mind the box: l_1 -APGD for sparse adversarial attacks on image classifiers," in International Conference on Machine Learning, PMLR, 2021, pp. 2201–2211. [Online]. Available: http://proceedings.mlr.press/v139/croce21a.html
    - F. Croce and M. Hein, "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks," in International conference on machine learning, PMLR, 2020, pp. 2206–2216. [Online]. Available: https://proceedings.mlr.press/v119/croce20b.html

- **auto-verify** ([GitHub](https://github.com/ADA-research/auto-verify))
    - For integrating additional verifiers such as nnenum, AB-Crown, VeriNet, and Oval-Bab. Please refer to the [auto-verify documentation](https://deepwiki.com/ADA-research/auto-verify/6.2-vnn-comp-format) for installation and usage details.

We thank the authors and maintainers of these projects for their contributions to the robustness research community.

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



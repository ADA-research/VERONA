# VERification Of Neural Architectures (VERONA)

The ada-verona package simplifies your experiment pipeline for performing local robustness verification on your networks and datasets. 
The entire package is class-based, which means that extending the existing configurations is accessible and easy. 
With one script it is possible to run an entire experiment with various networks, images and perturbation magnitudes (epsilons). 
The package can be used to create robustness distributions [Bosman,Berger, Hoos and van Rijn, 2023](https://ada.liacs.leidenuniv.nl/papers/BosEtAl25.pdf) and per-class robustness distributions [Bosman et al., 2024](https://ada.liacs.leidenuniv.nl/papers/BosEtAl24.pdf).


## Authors

This package was created and is maintained by members the [ADA Research Group](https://adaresearch.wordpress.com/about/), which focuses on the development of AI techniques that complement human intelligence and automated algorithm design. The current core team includes:

- **Annelot Bosman** (LIACS, Leiden University)
- **Aaron Berger** (TU Delft)
- **Hendrik Baacke** (AIM, RWTH Aachen University)
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


This package was tested only on python version 3.10 and we cannot guarentee it working on any other python version. 
## Documentation
In case you have more questions, please refer to the [VERONA documentation](https://deepwiki.com/ADA-research/VERONA).

## Contributing
We welcome contributions to the ada-verona package! If you find a bug, have a feature request, or want to contribute code, please follow these steps:
0. **Create an Issue:** Before starting work on a new feature or bug fix, please create an issue in the GitHub repository to discuss your plans. This helps us coordinate contributions and avoid duplicate work.
1. **Fork the Repository:** Create a personal copy of the repository on GitHub.
2. **Create a Branch:** Create a new branch for your feature or bug fix.
3. **Make Changes:** Implement your changes in the codebase.
4. **Test Your Changes:** Ensure that your changes do not break existing functionality and that new features work as intended.
5. **Commit Your Changes:** Write clear and concise commit messages describing your changes.
6. **Push to Your Fork:** Push your changes to your forked repository.
7. **Create a Pull Request:** Open a pull request against the main repository, describing your changes and why they are needed.
We appreciate all contributions, whether they are bug fixes, new features, or improvements to the documentation.
8. **Review Process:** Your pull request will be reviewed by at least one of the maintainers. They may request changes or provide feedback.


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

The images can be placed in it optionally, if one wants to execute the experiments using custom data. Otherwise, Pytorch Datasets can be used too and no image folder has to be created.

### Custom Verifiers
The ada-verona does not entail the [auto-verify](https://github.com/ADA-research/auto-verify) integration in order to provide a leaner package composition. If you want to use the verifiers supported by [auto-verify](https://github.com/ADA-research/auto-verify) and do not wish to tinker with integrating your own verifiers, then we can refer to our ada-auto-verona package.

<!-- **#TODO**: once package is up, add the link here. Prob. needs a catchier name also.  -->

Custom verifiers can be implemented too, by using the [VerificationModule](robustness_experiment_box/verification_module/verification_module.py) interface.

#### How to Add Your Own Verifier

You can easily add your own verifier by following these steps:

1. **Implement the `VerificationModule` interface:**
   - Create a new class that inherits from `VerificationModule` (see `robustness_experiment_box/verification_module/verification_module.py`).
   - Implement the `verify(self, verification_context: VerificationContext, epsilon: float)` method. This method should return either a string (e.g., "SAT", "UNSAT", "ERR") or a `CompleteVerificationData` object.

   Example:
   ```python
   from robustness_experiment_box.verification_module.verification_module import VerificationModule

   class MyCustomVerifier(VerificationModule):
       def verify(self, verification_context, epsilon):
           # Your custom verification logic here
           # Return "SAT", "UNSAT", or a CompleteVerificationData object
           return "UNSAT"
   ```

2. **(Optional) If your verifier wraps an external tool:**
   - Implement the `Verifier` interface in `verification_runner.py`.
   - Then, use the `GenericVerifierModule` to wrap your `Verifier` implementation, which will handle property file management and result parsing for you.

   Example:
   ```python
   from robustness_experiment_box.verification_module.verification_runner import Verifier, GenericVerifierModule

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
- Fast Gradient Sign Method (FGSM) [Goodfellow et al., 2015](https://arxiv.org/abs/1412.6572)
- Projected Gradient Descent (PGD) [Madry et al., 2018](https://arxiv.org/abs/1706.06083)
- AutoAttack https://github.com/fra31/auto-attack. For using AutoAttack the package has to be installed first as described in the AutoAttack repository.

### Custom Attacks
Custom attacks can be implemented too, by using the [Attack](robustness_experiment_box/verification_module/attacks/attack.py) interface.

### Datasets
- The package was tested on the MNIST and the CIFAR10 dataset. Example scripts for executing the package on mnist or a custom dataset can be found in the ```scripts``` folder

### Getting Started: Guide and Example Scripts

To help you get up and running with ada-verona, we provide a comprehensive tutorial notebook and a collection of practical example scripts:

- **Main Guide:**
  - The primary resource for learning how to use ada-verona is the Jupyter notebook found in the `notebooks` folder. This tutorial notebook offers an overview of the package components, step-by-step instructions, and practical demonstrations of typical workflows. We highly recommend starting here to understand the core concepts and capabilities of the package.

- **Quick-Start Example Scripts:**
  - The `scripts` folder contains a variety of example scripts designed to help you get started quickly with ada-verona. These scripts cover common use cases and can be run directly (from within the `scripts` folder) to see how to perform tasks such as:
    - Running VERONA with a custom dataset and ab-crown (`create_robustness_distribution_from_test_dataset.py`).
    - Loading a PyTorch dataset and running VERONA with one-to-any or one-to-one verification (`create_robustness_dist_on_pytorch_dataset.py`).
    - Distributing jobs across multiple nodes using SLURM for large-scale experiments (`multiple_jobs` folder), including distributing tasks over CPU and GPU for different verifiers in the same experiment.

The notebook is your main entry point for learning and understanding the package, while the scripts serve as practical templates and quick-start resources for your own experiments.

## Related Papers and Citation
This package was created to simplify reproducing and extending the results of two different lines of work of the ADA research group. Please consider citing these works when using this package for your research. 

### Robustness distributions 
- @article{BosEtAl23,
    author = "Bosman, Annelot W. and Hoos, Holger H. and van Rijn, Jan N.",
    title = "A Preliminary Study of Critical Robustness Distributions in Neural Network Verification",
    year = "2023",
    journal = "6th Workshop on Formal Methods for ML-Enabled Autonomous Systems (FoMLAS) co-located with the 35th International Conference on Computer Aided Verification (CAV 2023)",
    booktitle = "to appear"
}
  
- https://ada.liacs.leidenuniv.nl/papers/BosEtAl23.pdf
- An extended version of this work is currently under review. 


### Per-class robustness distributions
- @inproceedings{BosEtAl24,
    author = {Bosman, Annelot W. and Münz, Anna L. and Hoos, Holger H. and van Rijn, Jan N.},
    title = {{A Preliminary Study to Examining Per-Class Performance Bias via Robustness Distributions}},
    year = {2024},
    booktitle = {The 7th International Symposium on AI Verification (SAIV) co-located with the 36th International Conference on Computer Aided Verification (CAV 2024)},
    url = {https://ada.liacs.leidenuniv.nl/papers/BosEtAl24.pdf}
}

### Upper bounds to robustness distributions
- @inproceedings{bergerEmpiricalAnalysisUpper,
  title = {Empirical {{Analysis}} of {{Upper Bounds}} of {{Robustness Distributions}} Using {{Adversarial Attacks}}},
  booktitle = {{{THE 19TH LEARNING AND IN}}℡{{LIGENT OPTIMIZATION CONFERENCE}}},
  author = {Berger, Aaron and Eberhardt, Nils and Bosman, Annelot Willemijn and Duwe, Henning and family=Rijn, given=Jan N., prefix=van, useprefix=true and Hoos, Holger},
  url = {https://openreview.net/forum?id=jsfqoRrsjy}
}

## Acknowledgements

This package makes use of the following tools and libraries:

- **AutoAttack** ([GitHub](https://github.com/fra31/auto-attack))
    - F. Croce and M. Hein, “Mind the box: l_1 -APGD for sparse adversarial attacks on image classifiers,” in International Conference on Machine Learning, PMLR, 2021, pp. 2201–2211. [Online]. Available: http://proceedings.mlr.press/v139/croce21a.html
    - F. Croce and M. Hein, “Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks,” in International conference on machine learning, PMLR, 2020, pp. 2206–2216. [Online]. Available: https://proceedings.mlr.press/v119/croce20b.html

- **auto-verify** ([GitHub](https://github.com/ADA-research/auto-verify))
    - For integrating additional verifiers such as nnenum, AB-Crown, VeriNet, and Oval-Bab. Please refer to the auto-verify documentation for installation and usage details.

We thank the authors and maintainers of these projects for their contributions to the robustness research community.


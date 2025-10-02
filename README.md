[![Pytests](https://github.com/ADA-research/VERONA/actions/workflows/run_pytests_when_PR_opened.yml/badge.svg)](https://github.com/ADA-research/VERONA/actions/workflows/run_pytests_when_PR_opened.yml)
[![Lint](https://github.com/ADA-research/VERONA/actions/workflows/lint.yml/badge.svg)](https://github.com/ADA-research/VERONA/actions/workflows/lint.yml)
[![Release to PyPI](https://github.com/ADA-research/VERONA/actions/workflows/pypi_release.yml/badge.svg)](https://github.com/ADA-research/VERONA/actions/workflows/pypi_release.yml)

# VERification Of Neural Architectures (VERONA)

VERONA simplifies your experiment pipeline for performing local robustness verification on your networks and datasets. 
VERONA is class-based, which means that extending the existing configurations is accessible and easy. 
With one script it is possible to run an entire experiment with various networks, images and perturbation magnitudes (epsilons). 

Example use cases of the VERONA package include creating robustness distributions [Bosman, Berger, Hoos and van Rijn, 2025](https://jair.org/index.php/jair/article/view/18403), empirically measuring robustness of networks using adversarial attacks and verifying networks with formal verification tools using the external [auto-verify](https://github.com/ADA-research/auto-verify) package.

## Authors

This package was created and is maintained by members the [ADA Research Group](https://ada.liacs.nl/), which focuses on the development of AI techniques that complement human intelligence and automated algorithm design. The current core team includes:

- **Annelot Bosman** (LIACS, Leiden University)
- **Aaron Berger** (TU Delft)
- **Hendrik S. Baacke** (AIM, RWTH Aachen University)
- **Holger H. Hoos** (AIM, RWTH Aachen University)
- **Jan van Rijn** (LIACS, Leiden University)

## Installation and Environment Setup

We recommend to have a look at the [Documentation](https://ada-research.github.io/VERONA).

### Create Virtual Environment and install ada-verona

The python package for VERONA is called `ada-verona`, as our research group is called ADA.

To run ada-verona, we recommend to set up a conda environment. We also recommend using [miniforge](https://github.com/conda-forge/miniforge) as the package manager.

**Create a new conda environment named `verona_env`:**
```bash
conda create -n verona_env python=3.10
conda activate verona_env
```
### Installing the package

Inside the conda environment, install the ada-verona package preferably using uv (fast Python package installer and resolver). Alternatively, you can install the package using pip only.
```bash
uv pip install ada-verona
```

### Local installation for e.g. development purposes

If you want to install ada-verona locally using git:

```bash
git clone https://github.com/ADA-research/VERONA.git
cd VERONA
uv pip install -e .
```

To help you get up and running with ada-verona, we provide a tutorial notebook and a collection of example scripts:
- **Main Guide:**
  - The primary resource for learning how to use VERONA is the Jupyter notebook found in the [`notebooks`](./notebooks/) folder. This tutorial notebook offers an overview of the package components, step-by-step instructions, and practical demonstrations of typical workflows. We highly recommend starting here to understand the core concepts and capabilities of the package.

- **Quick-Start Example Scripts:**
  - The [`scripts`](./scripts/) folder contains a variety of example scripts designed to help you get started quickly with ada-verona. These scripts cover common use cases and can be run directly (from within the `scripts` folder) to see how to perform tasks such as:
    - Running VERONA with a custom dataset and ab-crown ([`create_robustness_distribution_from_test_dataset.py`](./scripts/create_robustness_distribution_from_test_dataset.py)).
    - Loading a PyTorch dataset and running VERONA with one-to-any or one-to-one verification ([`create_robustness_dist_on_pytorch_dataset.py`](./scripts/create_robustness_dist_on_pytorch_dataset.py)).
    - Distributing jobs across multiple nodes using SLURM for large-scale experiments ([`multiple_jobs`](./scripts/multiple_jobs/) folder), including distributing tasks over CPU and GPU for different verifiers in the same experiment.
    - Using auto-verify integration ([`create_robustness_dist_with_auto_verify.py`](./scripts/create_robustness_dist_with_auto_verify.py)).

The notebook is your main entry point for learning and understanding the package, while the scripts serve as practical templates and quick-start resources for your own experiments.

## Datasets

The package was tested on the MNIST and the CIFAR10 dataset. Example scripts for executing the package on MNIST or a custom dataset can be found in the [`scripts`](./scripts/) folder.

## Related Papers and Citation

This package was created to simplify reproducing and extending the results of different lines of work of the ADA research group. For more information about the ADA research group, please refer to the [official website](https://ada.liacs.nl/).

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

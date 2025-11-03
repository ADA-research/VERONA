[![codecov](https://codecov.io/gh/ADA-research/VERONA/graph/badge.svg?token=O0J6S4TSF2)](https://codecov.io/gh/ADA-research/VERONA)
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
- **Hendrik Baacke** (AIM, RWTH Aachen University)
- **Holger H. Hoos** (AIM, RWTH Aachen University)
- **Jan van Rijn** (LIACS, Leiden University)

## Installation and Environment Setup

We recommend to have a look at the [Documentation](https://ada-research.github.io/VERONA).

### Create Virtual Environment and install ada-verona

The python package for VERONA is called `ada-verona`, as our research group is called ADA.

To run ada-verona, we recommend to set up a conda environment. We also recommend using [miniforge](https://github.com/conda-forge/miniforge) as the package manager and using [uv](https://docs.astral.sh/uv/) for dependency management.

**Create a new conda environment named `verona_env`:**
```bash
conda create -n verona_env python=3.10
conda activate verona_env
```
### Installing the package

Inside the conda environment, install the ada-verona package:


```bash
uv pip install ada-verona
```

### GPU-version Installation

Note that the default installation is CPU-only, and that we recommend to install the GPU version for full functionality, as, e.g. [AB-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN), heavily relies on GPU parallelization for practical performance. The package resolver will automatically resolve the correct version of the package for your system, depending on whether you have a GPU available, but you can also explicitly install the GPU version with the following command:

```bash
uv pip install ada-verona[gpu]
```

### Local installation for e.g. development purposes

If you want to install ada-verona locally using git:

```bash
git clone https://github.com/ADA-research/VERONA.git
cd VERONA
uv sync --dev  #or uv sync --extra gpu --dev for GPU-version installation

```
### Optional: AutoAttack Installation

To use the AutoAttack adversarial attack wrapper ([`AutoAttackWrapper`](./ada_verona/verification_module/attacks/auto_attack_wrapper.py)), you need to install AutoAttack separately from its GitHub repository:

```bash
uv pip install git+https://github.com/fra31/auto-attack
```

This package provides ensemble-based adversarial attacks for robustness evaluation, as described in the paper by [Croce and Hein (2020)](https://proceedings.mlr.press/v139/croce21a.html).

### Optional: AutoVerify Installation

To use the auto-verify verifiers, you need to install auto-verify separately:

```bash
uv pip install auto-verify>=0.1.4
```

This package provides a framework for integrating verifiers. Please refer to the [auto-verify documentation](https://ada-research.github.io/auto-verify/) for details about auto-verify.

## Guides
To help you get up and running with ada-verona, we provide a tutorial notebook and a collection of example scripts in the folder [`examples`](./examples/) :
- **Main Guide:**
  - The primary resource for learning how to use VERONA is the Jupyter notebook found in the [`notebooks`](./examples/notebooks/) folder. This tutorial notebook offers an overview of the package components, step-by-step instructions, and practical demonstrations of typical workflows. We highly recommend starting here to understand the core concepts and capabilities of the package.

- **Quick-Start Example Scripts:**
  - The [`scripts`](./examples/scripts/) folder contains a variety of example scripts designed to help you get started quickly with ada-verona. These scripts cover common use cases and can be run directly (from within the `scripts` folder) to see how to perform tasks such as:
    - Running VERONA with a custom dataset and ab-crown ([`create_robustness_distribution_from_test_dataset.py`](./examples/scripts/create_robustness_distribution_from_test_dataset.py)).
    - Loading a PyTorch dataset and running VERONA with one-to-any or one-to-one verification ([`create_robustness_dist_on_pytorch_dataset.py`](./examples/scripts/create_robustness_dist_on_pytorch_dataset.py)).
    - Distributing jobs across multiple nodes using SLURM for large-scale experiments ([`multiple_jobs`](./examples/scripts/multiple_jobs/) folder), including distributing tasks over CPU and GPU for different verifiers in the same experiment.
    - Using auto-verify integration ([`create_robustness_dist_with_auto_verify.py`](./examples/scripts/create_robustness_dist_with_auto_verify.py)).

The notebook is your main entry point for learning and understanding the package, while the scripts serve as practical templates and quick-start resources for your own experiments.

## Datasets

The package was tested on the MNIST, GTRSB and the CIFAR-10 datasets. Example scripts for executing the package on MNIST or a custom dataset can be found in the [`scripts`](./examples/scripts/) folder.

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


## Acknowledgements

This package makes use of the following tools and libraries:

- **AutoAttack** ([GitHub](https://github.com/fra31/auto-attack))
    - F. Croce and M. Hein, "Mind the box: l_1 -APGD for sparse adversarial attacks on image classifiers," in International Conference on Machine Learning, PMLR, 2021, pp. 2201–2211. [Online]. Available: https://proceedings.mlr.press/v139/croce21a.html
    - F. Croce and M. Hein, "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks," in International conference on machine learning, PMLR, 2020, pp. 2206–2216. [Online]. Available: https://proceedings.mlr.press/v119/croce20b.html

- **auto-verify** ([GitHub](https://github.com/ADA-research/auto-verify))
    - For integrating verifiers [nnenum](https://github.com/stanleybak/nnenum), [AB-Crown](https://github.com/Verified-Intelligence/alpha-beta-CROWN), [VeriNet](https://github.com/vas-group-imperial/VeriNet), and [Oval-Bab](https://github.com/oval-group/oval-bab). Please refer to the [auto-verify documentation](https://github.com/ADA-research/auto-verify) for details about auto-verify.

We thank the authors and maintainers of these projects, as well as the authors and maintainers of the verifiers for their contributions to the robustness research community.


# VERONA

VERONA is a lightweight Python package for setting up adversarial robustness experiments and to compute robustness distributions. The package implements adversarial attacks which can be extended with the auto-verify plugin to enable complete verification. 

## Installation and Environment Setup

### Create Virtual Environment and install ada-verona

The python package for VERONA is called **ada-verona**, as our research group is called ADA.

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

**Note:** On macOS (and sometimes on Linux), you may need to install `swig` first with `conda install -c conda-forge swig`.

```bash
uv pip install auto-verify>=0.1.4
```

This package provides a framework for integrating verifiers. Please refer to the [auto-verify documentation](https://ada-research.github.io/auto-verify/) for details about auto-verify.
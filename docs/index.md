# VERONA

VERONA is a lightweight Python package for setting up adversarial robustness experiments and to compute robustness distributions. The package implements adversarial attacks which can be extended with the auto-verify plugin to enable complete verification. 

## Installation and Environment Setup

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

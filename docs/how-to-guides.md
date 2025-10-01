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


## Setting up the Experiment Directory

The experiment directory structure by default is expected as follows:

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

## Verification

### Auto-Verify

VERONA features a plugin architecture through the [`AutoVerifyModule`](./ada_verona/verification_module/auto_verify_module.py) which allows integration with [auto-verify](https://github.com/ADA-research/auto-verify) when it's installed in the same environment. This design provides several benefits:

1. **Independence**: VERONA works perfectly without auto-verify, using attack-based verification methods for empirical upper bounds.
2. **Automatic Detection**: When auto-verify is installed in the same environment, its verifiers become automatically available
3. **Interface**: The same API works regardless of which verification backend is used

### Available Verifiers

Currently, auto-verify supports [nnenum](https://github.com/stanleybak/nnenum), [AB-Crown](https://github.com/Verified-Intelligence/alpha-beta-CROWN), [VeriNet](https://github.com/vas-group-imperial/VeriNet), and [Oval-Bab](https://github.com/oval-group/oval-bab). We thank the authors and maintainers of these projects for their contributions to the robustness research community.

We plan to add more verifiers to auto-verify in the future. For additional information about auto-verify, please refer to the [official GitHub repository](https://github.com/ADA-research/auto-verify) and [documentation](https://ada-research.github.io/auto-verify/).
Verifiers can be installed using the `auto-verify` command, e.g. to install nnenum and abcrown:

```bash
auto-verify install nnenum abcrown
```
To see the current configuration of auto-verify, you can use the `auto-verify config show` command.

```bash
auto-verify config show
```
### Possible Extension: How to Add Your Own Verifier

Custom verifiers can be added to VERONA by using the [`VerificationModule`](./ada_verona/verification_module/verification_module.py) interface.

**Implement new verifiers using the `VerificationModule` class:**

- Create a new class that inherits from [`VerificationModule`](./ada_verona/verification_module/verification_module.py).
- Implement the `verify(self, verification_context: VerificationContext, epsilon: float)` method. This method should return either a string (e.g., "SAT", "UNSAT", "ERR") or a `CompleteVerificationData` object.

Example:
```python
from ada_verona.verification_module.verification_module import VerificationModule

class MyCustomVerifier(VerificationModule):
    def verify(self, verification_context, epsilon):
        # Your custom verification logic here
        # Return "SAT", "UNSAT", or a CompleteVerificationData object
        return "UNSAT"
```

## Adversarial Attacks

Currently VERONA implements the following adversarial attack methods:

- **Fast Gradient Sign Method (FGSM)** [Goodfellow et al., 2015](https://arxiv.org/abs/1412.6572)
- **Projected Gradient Descent (PGD)** [Madry et al., 2018](https://arxiv.org/abs/1706.06083)
- **AutoAttack** [Croce and Hein, 2020](https://pypi.org/project/pyautoattack/).

### Possible Extension: Custom Attacks

Custom attacks can be implemented by using the [`Attack`](./ada_verona/verification_module/attacks/attack.py) interface.


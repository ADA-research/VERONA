# Copyright 2025 ADA Reseach Group and VERONA council. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tutorial: robustness distributions with Foolbox attacks across norms & datasets.

This walk-through showcases several pieces of VERONA at once:

  * Part 1 -- run three *L-inf* Foolbox attacks (FGSM, PGD, DeepFool) on the
    bundled ImageFileDataset and overlay their robustness distributions to see
    how the choice of attack changes the estimated robustness.
  * Part 2 -- run an *L2* attack to show that the pipeline is norm-agnostic: only
    the attack class and the epsilon scale change (L2 budgets are much larger
    than L-inf ones). This part also loads its data via PytorchExperimentDataset
    (a torchvision dataset) instead of ImageFileDataset, to demonstrate that
    loader.

Background -- what is a "robustness distribution"?
--------------------------------------------------
For every (correctly classified) input we binary-search for the *critical
epsilon*: the smallest perturbation budget at which the attack flips the
prediction. Collecting that critical epsilon over many inputs gives a
distribution; its empirical CDF (ECDF) is the robustness distribution.

How to read the comparison
---------------------------
A *stronger* attack finds adversarial examples with *smaller* perturbations, so
its ECDF sits further to the *left / higher*; a *weaker* attack pushes its curve
to the right. Since attacks only upper-bound the true robustness, the left-most
curve is the most informative.

NOTE -- plotting dependency
---------------------------
The comparison plots use the grouping-agnostic ReportCreator (``group_by=...``),
which is added in branch ``henba1-enhance-report-creator`` (commit
53f7c65093adcdaa7b7b639628d1247c14dbeb1e). Until that change is merged, run this
script with that version of ``ada_verona.analysis.report_creator`` on the path.

How to run
----------
    cd examples/scripts
    python create_robustness_dist_foolbox.py

Requires ``foolbox`` (``pip install foolbox``). Outputs (CSVs + PNG plots) are
written next to the example experiment under ``results_foolbox_comparison/``.
"""

import importlib.util

# ── Step 0: make sure the optional dependency is available ─────────────────────
# Foolbox is an *optional* extra for VERONA, so fail early with a helpful message
# rather than with a cryptic ImportError halfway through the run.
if importlib.util.find_spec("foolbox") is None:
    raise ImportError("Foolbox not found. This package is required for this script. To install: pip install foolbox")

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from foolbox.attacks import L2PGD, LinfDeepFoolAttack, LinfFastGradientAttack, LinfPGD

import ada_verona.util.logger as logger
from ada_verona.analysis.report_creator import ReportCreator
from ada_verona.database.dataset.image_file_dataset import ImageFileDataset
from ada_verona.database.dataset.pytorch_experiment_dataset import PytorchExperimentDataset
from ada_verona.database.experiment_repository import ExperimentRepository
from ada_verona.dataset_sampler.predictions_based_sampler import PredictionsBasedSampler
from ada_verona.epsilon_value_estimator.binary_search_epsilon_value_estimator import (
    BinarySearchEpsilonValueEstimator,
)
from ada_verona.verification_module.attack_estimation_module import AttackEstimationModule
from ada_verona.verification_module.attacks.foolbox_attack import FoolboxAttack
from ada_verona.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)

logger.setup_logging(level=logging.INFO)
torch.manual_seed(0)  # PGD uses a random start; seed for reproducible curves

# ── Step 1: the three L-inf attacks to compare ─────────────────────────────────
# Each entry is (label, foolbox_attack_class, attack_kwargs). We deliberately span
# the spectrum from "cheap & weak" to "strong" to "minimal-perturbation":
#   * FGSM      -- a single gradient step; weakest here, so it over-estimates
#                  robustness (curve furthest right).
#   * PGD       -- iterated FGSM projected back into the L-inf ball; a strong,
#                  widely used baseline (usually the left-most curve).
#   * DeepFool  -- walks toward the nearest decision boundary (minimal perturbation).
# Any untargeted L-inf Foolbox attack works -- just add it to this list.
LINF_ATTACKS = [
    ("FGSM", LinfFastGradientAttack, {}),
    ("PGD", LinfPGD, {"steps": 40}),
    ("DeepFool", LinfDeepFoolAttack, {"steps": 50}),
]

# ── Step 2: experiment configuration ───────────────────────────────────────────
# Anchor every path to *this file* so the script runs from any working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = SCRIPT_DIR.parent / "example_experiment"

results_root = EXAMPLE_DIR / "results_foolbox_comparison"
network_folder = EXAMPLE_DIR / "data" / "networks"
image_folder = EXAMPLE_DIR / "data" / "images"
image_label_file = EXAMPLE_DIR / "data" / "image_labels.csv"

timeout = 600
# Epsilon grids the binary search snaps to. NOTE the very different scales: an
# L-inf budget bounds the *per-pixel* change (small), whereas an L2 budget bounds
# the *whole-image* Euclidean norm (much larger). Using an L-inf grid for an L2
# attack (or vice-versa) would make every input look trivially (non-)robust.
linf_epsilon_list = np.arange(0.00, 0.4, 0.0039)
l2_epsilon_list = np.arange(0.00, 4.0, 0.04)

# The sampler and property generator are shared everywhere. We only attack inputs
# the network already classifies correctly (flipping a wrong prediction is moot).
dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=True)
property_generator = One2AnyPropertyGenerator()


def run_attack_experiment(label, attack_cls, attack_kwargs, dataset, epsilon_list) -> pd.DataFrame:
    """Run the full VERONA pipeline for a single attack/dataset and return results.

    Args:
        label: Human-readable name used for the experiment folder and plot legend.
        attack_cls: The Foolbox attack class to wrap.
        attack_kwargs: Keyword arguments forwarded to the attack constructor.
        dataset: The ExperimentDataset to evaluate on.
        epsilon_list: The (norm-appropriate) epsilon grid for the binary search.

    Returns:
        The per-input result DataFrame, tagged with an ``attack`` column.
    """
    print(f"\n{'=' * 70}\nRunning attack: {label} ({attack_cls.__name__}) with kwargs={attack_kwargs}\n{'=' * 70}")

    # Each attack gets its own experiment folder so results never clobber each
    # other; re-running the script overwrites the previous run cleanly.
    experiment_name = f"foolbox_{label}"
    file_database = ExperimentRepository(base_path=results_root, network_folder=network_folder)
    file_database.initialize_new_experiment(experiment_name)
    file_database.save_configuration(
        dict(
            experiment_name=experiment_name,
            attack=attack_cls.__name__,
            attack_kwargs=attack_kwargs,
            network_folder=str(network_folder),
            dataset=str(dataset),
            timeout=timeout,
            epsilon_list=[str(x) for x in epsilon_list],
        )
    )

    # Wrap the Foolbox attack and plug it into the estimation pipeline. The binary
    # search calls the attack repeatedly to home in on the critical epsilon.
    verifier = AttackEstimationModule(attack=FoolboxAttack(attack_cls, bounds=(0, 1), **attack_kwargs))
    epsilon_value_estimator = BinarySearchEpsilonValueEstimator(
        epsilon_value_list=epsilon_list.copy(), verifier=verifier
    )

    network_list = file_database.get_network_list()
    print(f"Found {len(network_list)} network(s).")

    for network in network_list:
        print(f"  Processing network: {network.name}")
        sampled_data = dataset_sampler.sample(network, dataset)
        print(f"  Sampled {len(sampled_data)} data point(s).")

        for i, data_point in enumerate(sampled_data):
            verification_context = file_database.create_verification_context(network, data_point, property_generator)
            epsilon_value_result = epsilon_value_estimator.compute_epsilon_value(verification_context)
            print(f"    [{label}] data point {i}: critical epsilon = {epsilon_value_result.epsilon}")
            file_database.save_result(epsilon_value_result)

    # Native per-experiment plots (hist/box/kde/ecdf) written alongside the CSV.
    file_database.save_plots()

    result_df = file_database.get_result_df()
    result_df["attack"] = label
    return result_df


# ── Step 3: Part 1 -- L-inf comparison on the bundled ImageFileDataset ──────────
# The example image dataset is small, so we evaluate every sampled input (no
# early stopping) to get the smoothest possible curves.
image_dataset = ImageFileDataset(image_folder=image_folder, label_file=image_label_file)
linf_results = [
    run_attack_experiment(label, cls, kwargs, image_dataset, linf_epsilon_list) for label, cls, kwargs in LINF_ATTACKS
]

combined_df = pd.concat(linf_results, ignore_index=True)
results_root.mkdir(parents=True, exist_ok=True)
combined_csv = results_root / "combined_results.csv"
combined_df.to_csv(combined_csv, index=False)
print(f"\nWrote combined L-inf results to {combined_csv}")

# Overlay the three L-inf attacks on one robustness distribution (grouped by attack).
report = ReportCreator(
    combined_df,
    group_by="attack",
    value_label="Critical epsilon (L-inf perturbation budget)",
)
ecdf_path = results_root / "comparison_ecdf.png"
box_path = results_root / "comparison_boxplot.png"
report.create_ecdf_figure().savefig(ecdf_path, bbox_inches="tight")
report.create_box_figure().savefig(box_path, bbox_inches="tight")
print(f"Saved L-inf comparison plots:\n  {ecdf_path}\n  {box_path}")

print("\nMean critical epsilon per L-inf attack (smaller = stronger attack):")
for attack_name, mean_eps in combined_df.groupby("attack")["epsilon_value"].mean().sort_values().items():
    print(f"  {attack_name:<10} {mean_eps:.4f}")

# ── Step 4: Part 2 -- an L2 attack on a torchvision dataset ─────────────────────
# Exactly the same pipeline; only two things change:
#   1. the attack is an L2 attack (so we use the larger L2 epsilon grid), and
#   2. the data comes from a torchvision dataset wrapped in PytorchExperimentDataset
#      instead of ImageFileDataset.
# MNIST is large, so we take a small subset (via get_subset) to keep the demo quick.
print(f"\n{'#' * 70}\nPart 2: L2 attack on a torchvision (PytorchExperimentDataset) MNIST subset\n{'#' * 70}")
torch_dataset = torchvision.datasets.MNIST(
    root=str(EXAMPLE_DIR / "data" / "pytorch_mnist"),
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)
pytorch_dataset = PytorchExperimentDataset(dataset=torch_dataset).get_subset(list(range(50)))

run_attack_experiment("L2PGD", L2PGD, {"steps": 40}, pytorch_dataset, l2_epsilon_list)

print("\nDone. Open the PNGs under results_foolbox_comparison/ to inspect the distributions.")

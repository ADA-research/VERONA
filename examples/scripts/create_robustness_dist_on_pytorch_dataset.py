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

import importlib.util
import logging
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

if importlib.util.find_spec("autoverify") is None:
    raise ImportError(
        "AutoVerify not found. This package is required for this script. "
        "To install: pip install auto-verify"
    )

from autoverify.verifier import AbCrown, Nnenum

import ada_verona.util.logger as logger
from ada_verona.database.dataset.experiment_dataset import ExperimentDataset
from ada_verona.database.dataset.pytorch_experiment_dataset import PytorchExperimentDataset
from ada_verona.database.experiment_repository import ExperimentRepository
from ada_verona.dataset_sampler.dataset_sampler import DatasetSampler
from ada_verona.dataset_sampler.predictions_based_sampler import PredictionsBasedSampler
from ada_verona.epsilon_value_estimator.binary_search_epsilon_value_estimator import (
    BinarySearchEpsilonValueEstimator,
)
from ada_verona.epsilon_value_estimator.epsilon_value_estimator import EpsilonValueEstimator
from ada_verona.verification_module.auto_verify_module import AutoVerifyModule
from ada_verona.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)
from ada_verona.verification_module.property_generator.one2one_property_generator import (
    One2OnePropertyGenerator,
)
from ada_verona.verification_module.property_generator.property_generator import PropertyGenerator

logger.setup_logging(level=logging.INFO)

torch.manual_seed(0)

def create_distribution(
    experiment_repository: ExperimentRepository,
    dataset: ExperimentDataset,
    dataset_sampler: DatasetSampler,
    epsilon_value_estimator: EpsilonValueEstimator,
    property_generator: PropertyGenerator,
):
    network_list = experiment_repository.get_network_list()
    failed_networks = []
    for network in network_list:
        try:
            sampled_data = dataset_sampler.sample(network, dataset)
        except Exception as e:
            logging.info(f"failed for network: {network} with error: {e}")
            failed_networks.append(network)
            continue
        for data_point in sampled_data:
            verification_context = experiment_repository.create_verification_context(
                network, data_point, property_generator
            )

            epsilon_value_result = epsilon_value_estimator.compute_epsilon_value(verification_context)

            experiment_repository.save_result(epsilon_value_result)

    experiment_repository.save_plots()
    logging.info(f"Failed for networks: {failed_networks}")


def main():
    timeout = 600
    epsilon_list = [0.001, 0.005, 0.05, 0.08]
    experiment_repository_path = Path("../example_experiment/results")
    network_folder = Path("../example_experiment/data/networks")
    torch_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
    )

    dataset = PytorchExperimentDataset(dataset=torch_dataset)

    experiment_repository = ExperimentRepository(base_path=experiment_repository_path, network_folder=network_folder)

    # Create distribution using one-to-one verification with nnenum
    experiment_name = "nnenum_one2one"
    property_generator = One2OnePropertyGenerator(target_class=1)

    # Nnenum has to be installed using auto-verify
    verifier = AutoVerifyModule(verifier=Nnenum(), timeout=timeout)

    epsilon_value_estimator = BinarySearchEpsilonValueEstimator(
        epsilon_value_list=epsilon_list.copy(), verifier=verifier
    )
    dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=True)
    experiment_repository.initialize_new_experiment(experiment_name)
    experiment_repository.save_configuration(
        dict(
            experiment_name=experiment_name,
            experiment_repository_path=str(experiment_repository_path),
            network_folder=str(network_folder),
            dataset=str(dataset),
            timeout=timeout,
            epsilon_list=[str(x) for x in epsilon_list],
        )
    )
    create_distribution(experiment_repository, dataset, dataset_sampler, epsilon_value_estimator, property_generator)

    # Create distribution using AB-Crown verifier
    experiment_name = "ab_crown_one2any"
    property_generator = One2AnyPropertyGenerator()
    verifier = AutoVerifyModule(verifier=AbCrown(), timeout=timeout)
    epsilon_value_estimator = BinarySearchEpsilonValueEstimator(
        epsilon_value_list=epsilon_list.copy(), verifier=verifier
    )
    dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=True)
    experiment_repository.initialize_new_experiment(experiment_name)
    experiment_repository.save_configuration(
        dict(
            experiment_name=experiment_name,
            experiment_repository_path=str(experiment_repository_path),
            network_folder=str(network_folder),
            dataset=str(dataset),
            timeout=timeout,
            epsilon_list=[str(x) for x in epsilon_list],
        )
    )

    create_distribution(experiment_repository, dataset, dataset_sampler, epsilon_value_estimator, property_generator)


if __name__ == "__main__":
    main()

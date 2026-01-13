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

import logging
from pathlib import Path

import numpy as np
from foolbox.attacks import L2CarliniWagnerAttack

import ada_verona.util.logger as logger
from ada_verona.database.dataset.image_file_dataset import ImageFileDataset
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

experiment_name = "foolbox_cw"
timeout = 600
experiment_repository_path = Path("../example_experiment/results_foolbox_cw")
network_folder = Path("../example_experiment/data/networks")
image_folder = Path("../example_experiment/data/images")
image_label_file = Path("../example_experiment/data/image_labels.csv")

epsilon_list = np.arange(0.00, 0.4, 0.0039)

dataset = ImageFileDataset(image_folder=image_folder, label_file=image_label_file)

file_database = ExperimentRepository(base_path=experiment_repository_path, network_folder=network_folder)

file_database.initialize_new_experiment(experiment_name)

file_database.save_configuration(
    dict(
        experiment_name=experiment_name,
        experiment_repository_path=str(experiment_repository_path),
        network_folder=str(network_folder),
        dataset=str(dataset),
        timeout=timeout,
        epsilon_list=[str(x) for x in epsilon_list],
    )
)

property_generator = One2AnyPropertyGenerator()
verifier = AttackEstimationModule(attack=FoolboxAttack(L2CarliniWagnerAttack, bounds=(0, 1), steps=100))

epsilon_value_estimator = BinarySearchEpsilonValueEstimator(epsilon_value_list=epsilon_list.copy(), verifier=verifier)
dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=True)

network_list = file_database.get_network_list()

print(f"Found {len(network_list)} networks.")

for network in network_list:
    print(f"Processing network: {network.name}")
    sampled_data = dataset_sampler.sample(network, dataset)
    print(f"Sampled {len(sampled_data)} data points.")

    for i, data_point in enumerate(sampled_data):
        if i >= 1:
            break

        print(f"Verifying data point {i}...")
        verification_context = file_database.create_verification_context(network, data_point, property_generator)

        epsilon_value_result = epsilon_value_estimator.compute_epsilon_value(verification_context)

        print(f"Result: {epsilon_value_result}")
        file_database.save_result(epsilon_value_result)

print("Done.")

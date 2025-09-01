import logging
from pathlib import Path

import ada_verona
from ada_verona.robustness_experiment_box.database.dataset.image_file_dataset import ImageFileDataset
from ada_verona.robustness_experiment_box.database.experiment_repository import ExperimentRepository
from ada_verona.robustness_experiment_box.dataset_sampler.predictions_based_sampler import PredictionsBasedSampler
from ada_verona.robustness_experiment_box.epsilon_value_estimator.binary_search_epsilon_value_estimator import (
    BinarySearchEpsilonValueEstimator,
)
from ada_verona.robustness_experiment_box.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

experiment_name = "auto_verify_nnenum_test"
timeout = 10
experiment_repository_path = Path("../experiment")
network_folder = Path("../experiment/networks")
image_folder = Path("../experiment/data/images")
image_label_file = Path("../experiment/data/image_labels.csv")
epsilon_list = [0.001, 0.005]

# Check if auto-verify is available and Nnenum verifier exists
if not ada_verona.HAS_AUTO_VERIFY:
    raise RuntimeError("Auto-verify is not available. Please install auto-verify package.")

if "nnenum" not in ada_verona.AUTO_VERIFY_VERIFIERS:
    raise RuntimeError(
        f"Nnenum verifier is not available. Available verifiers: {ada_verona.AUTO_VERIFY_VERIFIERS}"
    )

# Create verifier using plugin architecture
verifier = ada_verona.create_auto_verify_verifier("nnenum", timeout=timeout)
if verifier is None:
    raise RuntimeError("Failed to create Nnenum verifier through plugin system.")

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

epsilon_value_estimator = BinarySearchEpsilonValueEstimator(epsilon_value_list=epsilon_list.copy(), verifier=verifier)
dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=False)

network_list = file_database.get_network_list()

for network in network_list:
    sampled_data = dataset_sampler.sample(network, dataset)

    for data_point in sampled_data:
        verification_context = file_database.create_verification_context(network, data_point, property_generator)

        epsilon_value_result = epsilon_value_estimator.compute_epsilon_value(verification_context)

        file_database.save_result(epsilon_value_result)

file_database.save_plots()

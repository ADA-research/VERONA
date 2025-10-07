import logging
from pathlib import Path

from autoverify.verifier import AbCrown

from ada_verona.database.dataset.image_file_dataset import ImageFileDataset
from ada_verona.database.experiment_repository import ExperimentRepository
from ada_verona.dataset_sampler.predictions_based_sampler import PredictionsBasedSampler
from ada_verona.epsilon_value_estimator.binary_search_epsilon_value_estimator import (
    BinarySearchEpsilonValueEstimator,
)
from ada_verona.verification_module.auto_verify_module import AutoVerifyModule
from ada_verona.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

experiment_name = "auto_verify"
timeout = 600
experiment_repository_path = Path("../example_experiment")
network_folder = Path("../example_experiment/data/networks")
image_folder = Path("../example_experiment/data/images")
image_label_file = Path("../example_experiment/data/image_labels.csv")
epsilon_list = [0.001, 0.005, 0.01, 0.02, 0.05, 0.08]

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
verifier = AutoVerifyModule(verifier=AbCrown(), timeout=timeout)

epsilon_value_estimator = BinarySearchEpsilonValueEstimator(epsilon_value_list=epsilon_list.copy(), verifier=verifier)
dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=True)

network_list = file_database.get_network_list()

for network in network_list:
    sampled_data = dataset_sampler.sample(network, dataset)

    for data_point in sampled_data:
        verification_context = file_database.create_verification_context(network, data_point, property_generator)

        epsilon_value_result = epsilon_value_estimator.compute_epsilon_value(verification_context)

        file_database.save_result(epsilon_value_result)

file_database.save_plots()

import logging
import os
import stat
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ada_verona.database.dataset.experiment_dataset import ExperimentDataset
from ada_verona.database.dataset.image_file_dataset import ImageFileDataset
from ada_verona.database.experiment_repository import ExperimentRepository
from ada_verona.dataset_sampler.dataset_sampler import DatasetSampler
from ada_verona.dataset_sampler.predictions_based_sampler import PredictionsBasedSampler
from ada_verona.verification_module.property_generator.one2any_property_generator import (
    One2AnyPropertyGenerator,
)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
torch.manual_seed(0)

def write_slurm_script(
    slurm_script_template: str,
    slurm_scripts_path: Path,
    file_verification_context: Path,
    base_path_experiment_repository: Path,
    network_folder: Path,
    experiment_name: Path,
    epsilon_list: np.ndarray,
    temp_slurm_script: Path,
):
    epsilon_list_str = " ".join(map(str, epsilon_list))

    slurm_script_content = slurm_script_template.format(
        slurm_scripts_path=slurm_scripts_path,
        file_verification_context=file_verification_context,
        base_path_experiment_repository=base_path_experiment_repository,
        network_folder=network_folder,
        experiment_name=experiment_name,
        epsilon_list=epsilon_list_str,
    )

    # create slurmscript and run it.
    with open(temp_slurm_script, "w") as f:
        f.write(slurm_script_content)


def run_slurm_script(temp_slurm_script: Path):
    os.chmod(temp_slurm_script, stat.S_IRWXU)
    os.system(f"sbatch {temp_slurm_script}")


def write_yaml_file(yaml_scripts_path: Path) -> Path:
    # Example of giving unique yaml names.
    # One could also use same format as the slurm scripts for consistency.
    now = datetime.now()
    now_string = now.strftime("%d-%m-%Y+%H_%M_%S_%f")[:-3]

    return Path(f"{yaml_scripts_path}/verification_context_{now_string}.yaml")


def get_network_name(network_path: Path) -> Path:
    root, _ = os.path.splitext(network_path)
    return os.path.basename(root)


def create_distribution(
    experiment_repository: ExperimentRepository,
    dataset: ExperimentDataset,
    dataset_sampler: DatasetSampler,
    epsilon_list: np.ndarray,
):
    # Example of slurmscript template
    slurm_script_template = (
        "#!/bin/sh \n"
        "#SBATCH --job-name=ada_verona\n"
        "#SBATCH --partition=graceGPU \n"
        "#SBATCH --exclude=ethnode[07] \n"
        "#SBATCH --output={slurm_scripts_path}/slurm_output_%A_%a.out \n"
        "python multiple_jobs/one_multiple_jobs.py --file_verification_context {file_verification_context}"
        "--base_path_experiment_repository "
        "{base_path_experiment_repository} --network_folder {network_folder}"
        "--experiment_name {experiment_name} --epsilon_list {epsilon_list} "
    )

    network_list = experiment_repository.get_network_list()
    failed_networks = []

    experiment_path = experiment_repository.get_act_experiment_path()

    # create extra directories for temporary files such as slurm scripts and yaml files.
    yaml_scripts_path = Path(experiment_path / "yaml")
    os.makedirs(yaml_scripts_path)

    slurm_scripts_path = Path(experiment_path / "slurm")
    os.makedirs(slurm_scripts_path)

    for network in network_list:
        try:
            sampled_data = dataset_sampler.sample(network, dataset)
        except Exception as e:
            logging.info(f"failed for network: {network} with error: {e}")
            failed_networks.append(network)
            continue

        for data_point in sampled_data:
            # make the verification context and save it to a temporary yaml file
            property_generator = One2AnyPropertyGenerator(number_classes=10, data_lb=0, data_ub=1)
            verification_context = experiment_repository.create_verification_context(
                network, data_point, property_generator
            )
            file_path = write_yaml_file(yaml_scripts_path)
            file_verification_context = experiment_repository.save_verification_context_to_yaml(
                file_path, verification_context
            )

            # ceate slurmscript and fill it based on this verification context.
            network_name = get_network_name(network.path)
            temp_slurm_script = Path(f"{slurm_scripts_path}/slurmscript_{data_point.id}_{network_name}.sh")

            write_slurm_script(
                slurm_script_template,
                slurm_scripts_path,
                file_verification_context,
                experiment_repository.base_path,
                experiment_repository.network_folder,
                experiment_path,
                epsilon_list,
                temp_slurm_script,
            )
            run_slurm_script(temp_slurm_script)

    logging.info(f"Failed for networks: {failed_networks}")


def main():
    timeout = 360
    epsilon_list = np.arange(0.0039, 0.2, 0.0039)

    experiment_repository_path = Path("../example_experiment")
    network_folder = Path("../example_experiment/data/networks")
    image_folder = Path("../example_experiment/data/images")
    image_label_file = Path("../example_experiment/data/image_labels.csv")

    dataset = ImageFileDataset(image_folder=image_folder, label_file=image_label_file)
    experiment_repository = ExperimentRepository(base_path=experiment_repository_path, network_folder=network_folder)

    # Create sampler for correct predictions or incorrect predictions
    dataset_sampler = PredictionsBasedSampler(sample_correct_predictions=True)

    # create the experiment based on a name you choose
    experiment_name = "multiple_jobs_example"
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

    create_distribution(experiment_repository, dataset, dataset_sampler, epsilon_list)


if __name__ == "__main__":
    main()

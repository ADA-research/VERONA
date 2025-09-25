from pathlib import Path

from robustness_experiment_box.database.experiment_repository import ExperimentRepository

experiment_repo = ExperimentRepository(base_path=Path('..'), network_folder = Path('tests/test_experiment/networks'))
networks = experiment_repo.get_network_list(csv_name = 'networks.csv')


pytorch = networks[1]
print(pytorch.get_input_shape())
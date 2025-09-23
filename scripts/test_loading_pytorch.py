from pathlib import Path
from robustness_experiment_box.database.experiment_repository import ExperimentRepository

experiment_repo = ExperimentRepository(base_path=Path('/home/abosman/dev/test_loading_pytorch.py'), network_folder = Path('/home/abosman/dev/VERONA/tests/test_experiment/networks'))
networks = experiment_repo.get_network_list(csv_name = 'networks.csv')



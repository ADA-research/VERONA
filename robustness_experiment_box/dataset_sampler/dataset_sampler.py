from abc import ABC, abstractmethod

from robustness_experiment_box.database.network import Network
from robustness_experiment_box.database.dataset.experiment_dataset import ExperimentDataset

class DatasetSampler(ABC):

    @abstractmethod
    def sample(network: Network, dataset: ExperimentDataset) -> ExperimentDataset:
        pass

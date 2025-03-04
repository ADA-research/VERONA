from abc import ABC, abstractmethod

from robustness_experiment_box.database.dataset.experiment_dataset import ExperimentDataset
from robustness_experiment_box.database.network import Network


class DatasetSampler(ABC):
    @abstractmethod
    def sample(network: Network, dataset: ExperimentDataset) -> ExperimentDataset:
        """Abstract method to sample a dataset.

        Args:
            network (Network): The Network object for which we are sampling the dataset.
            dataset (ExperimentDataset): The dataset object to sample.

        Returns:
            ExperimentDataset: The sampled dataset we are analysing for this Network object.
        """
        pass

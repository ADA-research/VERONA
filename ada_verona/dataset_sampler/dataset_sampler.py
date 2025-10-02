from abc import ABC, abstractmethod

from ada_verona.database.dataset.experiment_dataset import ExperimentDataset
from ada_verona.database.machine_learning_model.network import Network


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
        raise NotImplementedError("This is an abstract method and should be implemented in subclasses.")

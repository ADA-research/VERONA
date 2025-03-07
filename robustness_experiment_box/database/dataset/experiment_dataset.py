from abc import ABC, abstractmethod

from typing_extensions import Self

from robustness_experiment_box.database.dataset.data_point import DataPoint


class ExperimentDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> DataPoint:
        pass

    @abstractmethod
    def get_subset(self, indices: list[int]) -> Self:
        pass

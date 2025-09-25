from abc import ABC, abstractmethod

from typing_extensions import Self

from ada_verona.database.dataset.data_point import DataPoint


class ExperimentDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError("This is an abstract method and should be implemented in subclasses.")
       
    @abstractmethod
    def __getitem__(self, idx: int) -> DataPoint:
        raise NotImplementedError("This is an abstract method and should be implemented in subclasses.")
  

    @abstractmethod
    def get_subset(self, indices: list[int]) -> Self:
        raise NotImplementedError("This is an abstract method and should be implemented in subclasses.")
       

from torch.utils.data import Dataset
import torch
from typing_extensions import Self
from robustness_experiment_box.database.dataset.data_point import DataPoint

class PytorchExperimentDataset():

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self._indices = [x for x in range(0, len(dataset))]

    def __len__(self) -> int:
        return len(self._indices)
    
    def __getitem__(self, idx) -> DataPoint:
        
        index = self._indices[idx]

        data, label = self.dataset[index]

        return DataPoint(index, label, data)
    
    def get_subset(self, indices: list[int]) -> Self:
        new_instance = PytorchExperimentDataset(self.dataset)

        new_instance._indices = indices

        return new_instance
    
    def __str__(self) -> str:
        return str(self._indices)
# Copyright 2025 ADA Reseach Group and VERONA council. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from torch.utils.data import Dataset
from typing_extensions import Self

from ada_verona.database.dataset.data_point import DataPoint


class PytorchExperimentDataset:
    """
    A dataset class for wrapping a PyTorch dataset for experiments.
    """

    def __init__(self, dataset: Dataset) -> None:
        """
        Initialize the PytorchExperimentDataset with a PyTorch dataset.

        Args:
            dataset (Dataset): The PyTorch dataset to wrap.
        """
        self.dataset = dataset
        self._indices = [x for x in range(0, len(dataset))]

    def __len__(self) -> int:
        """
        Get the number of data points in the dataset.

        Returns:
            int: The number of data points in the dataset.
        """
        return len(self._indices)

    def __getitem__(self, idx) -> DataPoint:
        """
        Get the data point at the specified index.

        Args:
            idx (int): The index of the data point.

        Returns:
            DataPoint: The data point at the specified index.
        """
        index = self._indices[idx]

        data, label = self.dataset[index]

        return DataPoint(index, label, data)

    def get_subset(self, indices: list[int]) -> Self:
        """
        Get a subset of the underlying pytorch dataset for 
        the specified indices.

        Args:
            indices (list[int]): The list of indices to get the subset for.

        Returns:
            Self: The subset of the dataset.
        """
        new_instance = PytorchExperimentDataset(self.dataset)

        new_instance._indices = indices

        return new_instance

    def __str__(self) -> str:
        """
        Get the string representation of the dataset.

        Returns:
            str: The string representation of the dataset.
        """
        return str(self._indices)

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

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

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DataPoint:
    """
    A class to represent a data point in a dataset.

    Attributes:
        id (str): The identifier of the data point.
        label (int): The label of the data point.
        data (torch.Tensor): The data associated with the data point.
    """

    id: str
    label: int
    data: torch.Tensor

    def to_dict(self) -> dict:
        """
        Convert the DataPoint to a dictionary.

        Returns:
            dict: The dictionary representation of the DataPoint.
        """
        return {"id": self.id, "label": self.label, "data": self.data.numpy().tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "DataPoint":
        """
        Create a DataPoint from a dictionary.

        Args:
            data (dict): The dictionary containing the DataPoint attributes.

        Returns:
            DataPoint: The created DataPoint.
        """
        return cls(id=data["id"], label=data["label"], data=torch.tensor(np.array(data["data"]).astype(np.float32)))

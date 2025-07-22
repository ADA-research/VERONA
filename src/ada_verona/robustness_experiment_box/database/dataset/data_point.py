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

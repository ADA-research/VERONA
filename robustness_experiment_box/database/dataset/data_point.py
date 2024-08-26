from dataclasses import dataclass
import torch
import numpy as np

@dataclass
class DataPoint:

    id: str
    label: int
    data: torch.Tensor

    def to_dict(self):
        return {
            'id': self.id, 
            'label': self.label, 
            'data' : self.data.numpy().tolist()
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(id=data['id'], label=data['label'], data=torch.tensor(np.array(data['data']).astype(np.float32)))

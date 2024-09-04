from abc import ABC, abstractmethod
from robustness_experiment_box.database.vnnlib_property import VNNLibProperty
import numpy as np
class PropertyGenerator(ABC):
    @abstractmethod
    def create_vnnlib_property(self, image: np.array, image_class: int, epsilon: float)-> VNNLibProperty:
        pass
    
    @abstractmethod
    def get_dict_for_epsilon_result(self) -> dict:
        pass

    def to_dict(self):
        pass
    
    @classmethod
    def from_dict(cls, data: dict):
        pass


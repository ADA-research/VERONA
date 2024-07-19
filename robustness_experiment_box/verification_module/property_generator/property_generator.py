from abc import ABC, abstractmethod
from robustness_experiment_box.database.vnnlib_property import VNNLibProperty
import numpy as np
class PropertyGenerator(ABC):
    @abstractmethod
    def create_vnnlib_property(self, image: np.array, image_class: int, epsilon: float)-> VNNLibProperty:
        pass


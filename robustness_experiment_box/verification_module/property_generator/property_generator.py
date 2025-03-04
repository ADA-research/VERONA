from abc import ABC, abstractmethod
from robustness_experiment_box.database.vnnlib_property import VNNLibProperty
import numpy as np


class PropertyGenerator(ABC):
    """
    Abstract base class for generating properties for verification.
    This class provides an interface for creating VNNLib properties and
    retrieving results in dictionary form.
    Methods:
        create_vnnlib_property(image: np.array, image_class: int, epsilon: float) -> VNNLibProperty:
            Abstract method to create a VNNLib property based on the given image, its class, and epsilon value.
        get_dict_for_epsilon_result() -> dict:
            Abstract method to get a dictionary representation of the epsilon result.
        to_dict():
            Converts the property generator instance to a dictionary.
        from_dict(cls, data: dict):
            Creates an instance of the property generator from a dictionary.
        ABC (type): Abstract base class from which this class inherits."""

    @abstractmethod
    def create_vnnlib_property(self, image: np.array, image_class: int, epsilon: float) -> VNNLibProperty:
        pass

    @abstractmethod
    def get_dict_for_epsilon_result(self) -> dict:
        pass

    def to_dict(self):
        pass

    @classmethod
    def from_dict(cls, data: dict):
        pass

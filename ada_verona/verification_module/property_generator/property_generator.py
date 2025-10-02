import importlib
from abc import ABC, abstractmethod

import numpy as np

from ada_verona.database.vnnlib_property import VNNLibProperty


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
        raise NotImplementedError("This is an abstract method and should be implemented in subclasses.")
      

    @abstractmethod
    def get_dict_for_epsilon_result(self) -> dict:
        raise NotImplementedError("This is an abstract method and should be implemented in subclasses.")


    @abstractmethod
    def to_dict(self):
        raise NotImplementedError("This is an abstract method and should be implemented in subclasses.")
    

    @classmethod
    def from_dict(cls, data: dict):
        class_name = data.pop("type", None)
        module_name = data.pop("module", None)  # Get module info

        if not class_name or not module_name:
            raise ValueError("Missing 'class' or 'module' key in dictionary")

        try:
            module = importlib.import_module(module_name)  # Dynamically import module
            subclass = getattr(module, class_name)  # Get class from module
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Could not import {class_name} from {module_name}: {e}") from e

        return subclass.from_dict(data)  # Call subclass's `from_dict`

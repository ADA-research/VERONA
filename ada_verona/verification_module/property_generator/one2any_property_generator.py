import numpy as np

from ada_verona.database.vnnlib_property import VNNLibProperty
from ada_verona.verification_module.property_generator.property_generator import PropertyGenerator


class One2AnyPropertyGenerator(PropertyGenerator):
    """One2AnyPropertyGenerator generates properties for untargeted verification of neural networks.
    This means the property is violated if we can find any class other than the target class 
    that has a higher output value.
    """

    def __init__(self, number_classes: int = 10, data_lb: int = 0, data_ub: int = 1):
        """
        Initialize the One2AnyPropertyGenerator with the number of classes, data lower bound, and data upper bound.
        Args:
            number_classes (int, optional): The number of classes. Defaults to 10.
            data_lb (int, optional): The lower bound of the input features. Defaults to 0.
            data_ub (int, optional): The upper bound of the input features. Defaults to 1.
        """
        super().__init__()
        self.number_classes = number_classes
        self.data_lb = data_lb
        self.data_ub = data_ub

    def create_vnnlib_property(self, image: np.array, image_class: int, epsilon: float) -> VNNLibProperty:
        """Creates a VNNLib property for a given image, its class, and a perturbation epsilon.
        Args:
            image (np.array): The input image as a numpy array.
            image_class (int): The class of the input image.
            epsilon (float): The perturbation value to create the property.

        Returns:
            VNNLibProperty: An object containing the name and content of the VNNLib property.
        """

        x = image
        mean = 0
        std = 1.0
        x_lb = np.clip(x - epsilon, self.data_lb, self.data_ub)
        x_lb = ((x_lb - mean) / std).reshape(-1)
        x_ub = np.clip(x + epsilon, self.data_lb, self.data_ub)
        x_ub = ((x_ub - mean) / std).reshape(-1)

        result = ""

        result += f"; Spec for image and epsilon {epsilon:.5f}\n"

        result += "\n; Definition of input variables\n"
        for i in range(len(x)):
            result += f"(declare-const X_{i} Real)\n"

        result += "\n; Definition of output variables\n"
        for i in range(self.number_classes):
            result += f"(declare-const Y_{i} Real)\n"

        result += "\n; Definition of input constraints\n"
        for i in range(len(x_ub)):
            result += f"(assert (<= X_{i} {x_ub[i]:.8f}))\n"
            result += f"(assert (>= X_{i} {x_lb[i]:.8f}))\n"

        result += "\n; Definition of output constraints\n"
    
        result += "(assert (or\n"
        for i in range(self.number_classes):
            if i == image_class:
                continue
            result += f"\t(and (>= Y_{i} Y_{image_class}))\n"
        result += "))\n"

        property_name = f"property_{image_class}_{str(epsilon).replace('.', '_')}"

        return VNNLibProperty(name=property_name, content=result)

    def get_dict_for_epsilon_result(self) -> dict:
        """
        Get a dictionary representation of the epsilon result.

        Returns:
            dict: The dictionary representation of the epsilon result.
        """
        return dict()

    def to_dict(self) -> dict:
        """
        Convert the One2AnyPropertyGenerator to a dictionary.

        Returns:
            dict: The dictionary representation of the One2AnyPropertyGenerator.
        """
        return dict(
            number_classes=self.number_classes,
            data_lb=self.data_lb,
            data_ub=self.data_ub,
            type=self.__class__.__name__,
            module=self.__class__.__module__,
        )
    
    
    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a One2AnyPropertyGenerator from a dictionary.
        Args:
            data (dict): The dictionary containing the One2AnyPropertyGenerator attributes.

        Returns:
            One2AnyPropertyGenerator: The created One2AnyPropertyGenerator.
        """
        return cls(number_classes=data["number_classes"], data_lb=data["data_lb"], data_ub=data["data_ub"])

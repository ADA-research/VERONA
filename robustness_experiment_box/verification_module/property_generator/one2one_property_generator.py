from robustness_experiment_box.verification_module.property_generator.property_generator import PropertyGenerator
from robustness_experiment_box.database.vnnlib_property import VNNLibProperty
import numpy as np

class One2OnePropertyGenerator(PropertyGenerator):
    def __init__(self, target_class: int, number_classes: int=10, data_lb: int =0, data_ub: int=1):
        super().__init__()
        self.number_classes = number_classes
        self.data_lb = data_lb
        self.data_ub = data_ub
        self.target_class = target_class


    def create_vnnlib_property(self, image: np.array, image_class: int, epsilon: float) -> VNNLibProperty:
    
        x = image
        mean = 0
        std = 1.0
        x_lb = np.clip(x - epsilon, self.data_lb, self.data_ub)
        x_lb = ((x_lb-mean)/std).reshape(-1)
        x_ub = np.clip(x + epsilon, self.data_lb, self.data_ub)
        x_ub = ((x_ub - mean) / std).reshape(-1)

        result = ""

        result += f"; Spec for image and epsilon {epsilon:.5f}\n"

        result +=f"\n; Definition of input variables\n"
        for i in range(len(x)):
            result += f"(declare-const X_{i} Real)\n"

        result +=f"\n; Definition of output variables\n"
        for i in range(self.number_classes):
            result +=f"(declare-const Y_{i} Real)\n"

        result += f"\n; Definition of input constraints\n"
        for i in range(len(x_ub)):
            result += f"(assert (<= X_{i} {x_ub[i]:.8f}))\n"
            result += f"(assert (>= X_{i} {x_lb[i]:.8f}))\n"

        result += f"\n; Definition of output constraints\n"
       
        result += f"(assert (or\n"
        result += f"\t(and (>= Y_{self.target_class} Y_{image_class}))\n"
        result += f"))\n"
        
        property_name = f"property_{image_class}_{str(epsilon).replace('.', '_')}"

        return VNNLibProperty(name=property_name, content=result)
    
    def get_dict_for_epsilon_result(self) -> dict:
        return dict(target_class=self.target_class)
    
    def to_dict(self):
        return dict(target_class=self.target_class, number_classes=self.number_classes, data_lb=self.data_lb, data_ub=self.data_ub)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(target_class=data["target_class"], number_classes=data["number_classes"], data_lb=data["data_lb"], data_ub=data["data_ub"])


from pathlib import Path

import pytest
from autoverify.verifier.verification_result import CompleteVerificationData

from robustness_experiment_box.database.dataset.data_point import DataPoint
from robustness_experiment_box.database.network import Network
from robustness_experiment_box.database.verification_context import VerificationContext
from robustness_experiment_box.verification_module.verification_module import VerificationModule


@pytest.fixture
def data_point():
    return DataPoint(0, Path("test_image.pt"), "0")


@pytest.fixture
def network():
    return Network(Path("test_network.onnx"))


@pytest.fixture
def tmp_path():
    return Path("tests/test_experiment/tmp")


@pytest.fixture
def verification_context(network, data_point, tmp_path):
    return VerificationContext(network, data_point, tmp_path, save_epsilon_results=False, property_generator=None)


class MockVerificationModule(VerificationModule):
    def __init__(self, result_dict: dict):
        self.result_dict = result_dict
        self.name = "MockVerificationModule"

    def verify(self, verification_context: VerificationContext, epsilon: float) -> str | CompleteVerificationData:
        return CompleteVerificationData(self.result_dict[epsilon], took=10.0)

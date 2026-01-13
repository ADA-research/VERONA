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

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from result import Ok

from ada_verona.database.dataset.data_point import DataPoint
from ada_verona.database.machine_learning_model.onnx_network import ONNXNetwork
from ada_verona.database.verification_context import VerificationContext
from ada_verona.verification_module.auto_verify_module import AutoVerifyModule
from ada_verona.verification_module.property_generator.one2any_property_generator import One2AnyPropertyGenerator


class _CapturingVerifier:
    def __init__(self, name: str):
        self.name = name
        self.seen_network: Path | None = None
        self.seen_property: Path | None = None

    def verify_property(self, network: Path, property: Path, **kwargs):  # noqa: A002
        self.seen_network = network
        self.seen_property = property
        return Ok("UNSAT")


def _parse_verona_header(vnnlib_text: str) -> tuple[float, int, np.ndarray]:
    epsilon = None
    image_class = None
    image_csv = None

    for raw_line in vnnlib_text.splitlines():
        line = raw_line.strip()
        if not line.startswith(";"):
            break
        if line.startswith("; verona_epsilon:"):
            epsilon = float(line.split(":", 1)[1].strip())
        elif line.startswith("; verona_image_class:"):
            image_class = int(line.split(":", 1)[1].strip())
        elif line.startswith("; verona_image:"):
            image_csv = line.split(":", 1)[1].strip()

    assert epsilon is not None
    assert image_class is not None
    assert image_csv is not None

    return epsilon, image_class, np.fromstring(image_csv, sep=",", dtype=np.float32)


def test_sdpcrown_injects_verona_metadata_header(tmp_path: Path):
    network_path = tmp_path / "network.onnx"
    network_path.write_text("", encoding="utf-8")

    verification_context = VerificationContext(
        network=ONNXNetwork(network_path),
        data_point=DataPoint(id="1", label=2, data=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)),
        tmp_path=tmp_path,
        property_generator=One2AnyPropertyGenerator(number_classes=10, data_lb=0, data_ub=1),
    )

    verifier = _CapturingVerifier(name="sdpcrown")
    module = AutoVerifyModule(verifier=verifier, timeout=1.0)

    epsilon = 0.1
    result = module.verify(verification_context, epsilon=epsilon)
    assert result == "UNSAT"

    assert verifier.seen_property is not None
    vnnlib_text = verifier.seen_property.read_text(encoding="utf-8")

    assert vnnlib_text.startswith("; verona_metadata_version: 1\n")

    parsed_epsilon, parsed_label, parsed_image = _parse_verona_header(vnnlib_text)
    assert parsed_epsilon == pytest.approx(epsilon)
    assert parsed_label == 2
    assert np.allclose(parsed_image, verification_context.data_point.data.detach().cpu().numpy().reshape(-1))


def test_non_sdpcrown_does_not_inject_metadata(tmp_path: Path):
    network_path = tmp_path / "network.onnx"
    network_path.write_text("", encoding="utf-8")

    verification_context = VerificationContext(
        network=ONNXNetwork(network_path),
        data_point=DataPoint(id="1", label=2, data=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)),
        tmp_path=tmp_path,
        property_generator=One2AnyPropertyGenerator(number_classes=10, data_lb=0, data_ub=1),
    )

    verifier = _CapturingVerifier(name="not_sdpcrown")
    module = AutoVerifyModule(verifier=verifier, timeout=1.0)

    result = module.verify(verification_context, epsilon=0.1)
    assert result == "UNSAT"

    assert verifier.seen_property is not None
    vnnlib_text = verifier.seen_property.read_text(encoding="utf-8")
    assert "; verona_metadata_version:" not in vnnlib_text

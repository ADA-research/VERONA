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

import torch

from ada_verona.database.dataset.data_point import DataPoint


def test_to_dict():
    data_point = DataPoint(
        id="dp1",
        label=1,
        data=torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    )
    expected_dict = {
        "id": "dp1",
        "label": 1,
        "data": [[1.0, 2.0], [3.0, 4.0]]
    }

    result = data_point.to_dict()

    assert result == expected_dict


def test_from_dict():
    data_dict = {
        "id": "dp1",
        "label": 1,
        "data": [[1.0, 2.0], [3.0, 4.0]]
    }
    expected_data_point = DataPoint(
        id="dp1",
        label=1,
        data=torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    )

    result = DataPoint.from_dict(data_dict)

    assert result.id == expected_data_point.id
    assert result.label == expected_data_point.label
    assert torch.equal(result.data, expected_data_point.data)
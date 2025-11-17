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

import pytest

from ada_verona.database.dataset.experiment_dataset import ExperimentDataset


def test_cannot_instantiate_dataset():
    """Ensure ExperimentDataset cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ExperimentDataset()


def test_abstract_methods_raise_not_implemented_error():
    # Call the abstract methods on the class itself (unbound)
    with pytest.raises(NotImplementedError):
        ExperimentDataset.__len__(ExperimentDataset)
        
    with pytest.raises(NotImplementedError):
        ExperimentDataset.__getitem__(ExperimentDataset, 0)
        
    with pytest.raises(NotImplementedError):
        ExperimentDataset.get_subset(ExperimentDataset, [0])
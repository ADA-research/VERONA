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

from ada_verona.verification_module.attacks.attack import Attack


def test_cannot_instantiate_property_generator():
    """Ensure PropertyGenerator cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Attack()

def test_abstract_methods_raise_not_implemented_error():
    # Call the abstract methods on the class itself (unbound)
    with pytest.raises(NotImplementedError):
        Attack.execute(Attack,None,None,None,None)
    
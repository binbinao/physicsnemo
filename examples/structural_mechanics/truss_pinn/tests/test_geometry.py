# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

# tests/test_geometry.py
import torch
from helpers.geometry import (
    NODES_XY, ELEMENTS, SUPPORT_DOFS, FREE_DOFS, E, RHO, AREA,
    element_lengths, element_directions, FREE_NODES,
)

def test_connectivity_is_triangulated():
    # every member connects distinct nodes, positive length
    L = element_lengths()
    assert L.shape == (11,)
    assert torch.all(L > 1.4)  # shortest member is the sqrt(2) ≈ 1.414 m diagonals
    assert torch.all(ELEMENTS[:, 0] != ELEMENTS[:, 1])

def test_support_and_free_dofs_partition():
    # 14 total DOFs partition into 3 support + 11... no: 3 fixed + 10 free + 1 free (roller x)
    # pin node0: dofs 0,1 ; roller node3: dof 7 (y). Free dofs = the rest.
    assert set(SUPPORT_DOFS) == {0, 1, 7}
    assert len(FREE_DOFS) == 11
    assert set(SUPPORT_DOFS).isdisjoint(FREE_DOFS)
    assert set(SUPPORT_DOFS) | set(FREE_DOFS) == set(range(14))

def test_directions_are_unit():
    c, s = element_directions()
    assert torch.allclose(c**2 + s**2, torch.ones(11))

def test_material_constants():
    assert E == 3e9 and RHO == 1150.0 and AREA == 1e-4


def test_geometry_tables_pinned():
    assert torch.equal(NODES_XY, torch.tensor([
        [0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0],
        [1.0, 1.0], [3.0, 1.0], [5.0, 1.0]]))
    assert torch.equal(ELEMENTS, torch.tensor([
        [0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [0, 4], [3, 6],
        [1, 4], [2, 5], [1, 5], [2, 6]]))
    assert FREE_NODES == (1, 2, 4, 5, 6)

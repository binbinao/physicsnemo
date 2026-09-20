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

"""Pratt truss geometry: single source of truth for nodes, members, supports, material."""
import torch

E = 3.0e9          # Pa, nylon 6/6
RHO = 1150.0       # kg/m^3
AREA = 1.0e-4      # m^2, 32x32 mm solid square
INERTIA = AREA**2 / 12.0  # m^4

P_MAX = 2000.0     # N, provisional; refrozen after stability scan (Task 4)

# node index: bottom 0..3 left→right, top 4..6 left→right
NODES_XY = torch.tensor([
    [0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0],
    [1.0, 1.0], [3.0, 1.0], [5.0, 1.0],
])
ELEMENTS = torch.tensor([
    [0, 1], [1, 2], [2, 3],      # bottom chord
    [4, 5], [5, 6],              # top chord
    [0, 4], [3, 6],              # end diagonals
    [1, 4], [2, 5],              # verticals
    [1, 5], [2, 6],              # inner diagonals (Pratt: slope down toward center)
])
SUPPORT_DOFS = (0, 1, 7)          # pin node 0 (x,y), roller node 3 (y only)
FREE_DOFS = tuple(d for d in range(14) if d not in SUPPORT_DOFS)  # 11 dofs
FREE_NODES = (1, 2, 4, 5, 6)
LOADABLE_NODES = (1, 2, 4, 5, 6)

def element_lengths() -> torch.Tensor:  # (11,)
    p = NODES_XY[ELEMENTS[:, 0]]
    q = NODES_XY[ELEMENTS[:, 1]]
    return (q - p).norm(dim=-1)

def element_directions():
    """Return (cos, sin) unit direction of each member, i→j."""
    p = NODES_XY[ELEMENTS[:, 0]]
    q = NODES_XY[ELEMENTS[:, 1]]
    d = q - p
    L = d.norm(dim=-1, keepdim=True)
    return d[..., 0] / L[..., 0], d[..., 1] / L[..., 0]

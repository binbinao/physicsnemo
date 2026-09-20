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

"""FEM assembly (global stiffness/mass) and batched static solve for the Pratt truss.

Matrices are assembled once in float64 on CPU (reference quality) and returned
as fresh clones so callers may mutate them freely.
"""
import math

import torch

from helpers import geometry

_DTYPE = torch.float64
_N_DOF = 2 * geometry.NODES_XY.shape[0]  # 14

# Lazily built global matrices (14, 14) float64.
_K = None
_M = None


def _member_matrices():
    """Per-member global stiffness and consistent mass matrices, (11, 4, 4) each.

    For member e with unit direction (c, s), length L, axial stiffness k=EA/L:
    G = [[cc, cs], [cs, ss]], K_e = k [[G, -G], [-G, G]].
    Consistent mass of a 2D bar element (axial from int N1 N2 dx > 0, plus the
    transverse block that carries the rotational/transverse inertia):
    M_e = (rho*A*L/6) [[2 I, I], [I, 2 I]] with I the 2x2 identity —
    orientation-independent, so rigid translation gives u'Mu = rho*A*L exactly.
    """
    L = geometry.element_lengths().to(_DTYPE)                    # (11,)
    c, s = geometry.element_directions()
    d = torch.stack([c, s], dim=-1).to(_DTYPE)                   # (11, 2)
    G = d.unsqueeze(-1) * d.unsqueeze(-2)                        # (11, 2, 2)
    k = geometry.E * geometry.AREA / L                           # (11,)
    K_e = k.view(-1, 1, 1) * torch.cat([                         # (11, 4, 4)
        torch.cat([G, -G], dim=-1),
        torch.cat([-G, G], dim=-1), ], dim=-2)
    m = geometry.RHO * geometry.AREA * L / 6.0                   # (11,)
    I2 = torch.eye(2, dtype=_DTYPE)
    M_e = m.view(-1, 1, 1) * torch.cat([                         # (11, 4, 4)
        torch.cat([2 * I2, I2], dim=-1),
        torch.cat([I2, 2 * I2], dim=-1), ], dim=-2).expand(11, 4, 4)
    return K_e, M_e


def _dof_map():
    """DOF indices [2i, 2i+1, 2j, 2j+1] per member, (11, 4)."""
    i = geometry.ELEMENTS[:, 0]
    j = geometry.ELEMENTS[:, 1]
    return torch.stack([2 * i, 2 * i + 1, 2 * j, 2 * j + 1], dim=-1)


def _assemble():
    """Build and cache the global 14x14 stiffness and consistent mass matrices."""
    global _K, _M
    if _K is None or _M is None:
        K_e, M_e = _member_matrices()
        dofs = _dof_map()                                        # (11, 4)
        n_mem, blk = dofs.shape
        # Flat (row, col, val) triples: entry (a, b) of member e lands at
        # K[dofs[e, a], dofs[e, b]]. Duplicate (row, col) pairs across members
        # MUST accumulate (advanced-index += would silently overwrite).
        rows = dofs.unsqueeze(-1).expand(n_mem, blk, blk).reshape(-1)
        cols = dofs.unsqueeze(-2).expand(n_mem, blk, blk).reshape(-1)
        _K = torch.zeros(_N_DOF, _N_DOF, dtype=_DTYPE)
        _K.index_put_((rows, cols), K_e.reshape(-1), accumulate=True)
        _M = torch.zeros(_N_DOF, _N_DOF, dtype=_DTYPE)
        _M.index_put_((rows, cols), M_e.reshape(-1), accumulate=True)
    return _K, _M


def assemble_K():
    """Global stiffness matrix (14, 14) float64; fresh clone per call."""
    return _assemble()[0].clone()


def assemble_M():
    """Global consistent mass matrix (14, 14) float64; fresh clone per call."""
    return _assemble()[1].clone()


def reduce_matrix(K):
    """Restrict a (..., 14, 14) matrix to the free DOFs -> (..., 11, 11)."""
    idx = torch.tensor(geometry.FREE_DOFS)
    return K.index_select(-2, idx).index_select(-1, idx)


def solve_static(K_red, f_red):
    """Solve K_red u = f_red. Batched: K_red (..., 11, 11), f_red (..., 11)."""
    if f_red.ndim == 1 and K_red.ndim == 2:
        return torch.linalg.solve(K_red, f_red.unsqueeze(-1)).squeeze(-1)
    batch = torch.broadcast_shapes(K_red.shape[:-2], f_red.shape[:-1])
    K = K_red.expand(batch + K_red.shape[-2:])
    f = f_red.unsqueeze(-1).expand(batch + f_red.shape[-1:] + (1,))
    return torch.linalg.solve(K, f).squeeze(-1)


def nodal_load(n, theta, P):
    """Point load P at node n, direction theta from +x axis -> (14,) float64.

    f[2n] = P*cos(theta), f[2n+1] = P*sin(theta). Accepts python scalars or
    0-dim tensors for theta/P (converted via float()).
    """
    n = int(n)
    theta = float(theta)
    P = float(P)
    f = torch.zeros(_N_DOF, dtype=_DTYPE)
    f[2 * n] = P * math.cos(theta)
    f[2 * n + 1] = P * math.sin(theta)
    return f


def axial_forces(u_full):
    """Member axial forces from full nodal displacement, tension positive.

    u_full: (..., 14) -> (..., 11). N_e = (EA/L) * (c*dux + s*duy) with the
    stretch measured along the member axis (i -> j).
    """
    i = geometry.ELEMENTS[:, 0]
    j = geometry.ELEMENTS[:, 1]
    dux = u_full[..., 2 * j] - u_full[..., 2 * i]
    duy = u_full[..., 2 * j + 1] - u_full[..., 2 * i + 1]
    L = geometry.element_lengths().to(_DTYPE)
    c, s = geometry.element_directions()
    c = c.to(_DTYPE)
    s = s.to(_DTYPE)
    k = geometry.E * geometry.AREA / L                             # (11,)
    return (k * (c * dux + s * duy)).to(u_full.dtype)

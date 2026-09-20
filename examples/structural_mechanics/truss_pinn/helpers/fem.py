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

"""FEM assembly (global stiffness/mass/geometric stiffness), batched static
solve, and the prestressed modal solver for the Pratt truss.

Matrices are assembled once in float64 on CPU (reference quality) and returned
as fresh clones so callers may mutate them freely. The modal solver reduces
the generalized eigenproblem (K + K_G) phi = w^2 M phi on the free DOFs to a
symmetric standard problem via a Cholesky transform of M; geometric
stiffness is rotated to global coordinates along each member's transverse
direction t_e = (-s_e, c_e).
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


_KG_PATTERN = None
_K_RED = None
_M_CHOL = None


def _kg_pattern():
    """Per-member geometric-stiffness matrices in GLOBAL coords, (11, 14, 14).

    The member-transverse direction is t_e = (-s_e, c_e); the geometric
    stiffness resists only differential motion along t_e:
    K_G_e = [[G_perp, -G_perp], [-G_perp, G_perp]] with G_perp = outer(t, t).
    For a horizontal member t = (0, 1) this reduces exactly to the classic
    local pattern ([[0,0],[0,1]] blocks, no axial terms); the rotation is
    what makes inclined members see the full N/L in their true transverse
    direction. Coefficient N_e/L_e is factored out and applied by callers;
    member overlap is summed there (within a member the dofs are distinct,
    so the per-member scatter needs no accumulation).
    """
    global _KG_PATTERN
    if _KG_PATTERN is None:
        c, s = geometry.element_directions()
        t = torch.stack([-s, c], dim=-1).to(_DTYPE)         # (11, 2)
        G_perp = t.unsqueeze(-1) * t.unsqueeze(-2)          # (11, 2, 2)
        kg = torch.cat([                                   # (11, 4, 4)
            torch.cat([G_perp, -G_perp], dim=-1),
            torch.cat([-G_perp, G_perp], dim=-1),
        ], dim=-2)
        dmap = _dof_map()                                   # (11, 4)
        e_idx = torch.arange(11).view(11, 1, 1).expand(11, 4, 4)
        rows = dmap[:, :, None].expand(11, 4, 4)
        cols = dmap[:, None, :].expand(11, 4, 4)
        G = torch.zeros(11, _N_DOF, _N_DOF, dtype=_DTYPE)
        G[e_idx, rows, cols] = kg
        _KG_PATTERN = G
    return _KG_PATTERN


def _reduced_K_and_M_chol():
    """Cache the reduced stiffness (11, 11) and Cholesky factor of reduced mass."""
    global _K_RED, _M_CHOL
    if _K_RED is None:
        K, M = _assemble()
        idx = torch.tensor(geometry.FREE_DOFS)
        _K_RED = K.index_select(-2, idx).index_select(-1, idx)
        M_red = M.index_select(-2, idx).index_select(-1, idx)
        _M_CHOL = torch.linalg.cholesky(M_red)              # lower triangular
    return _K_RED, _M_CHOL


def assemble_KG(u_full):
    """Geometric stiffness from member axial forces, (..., 14) -> (..., 14, 14).

    K_G = sum_e (N_e / L_e) * G_e with G_e the member-transverse pattern
    [[G_perp, -G_perp], [-G_perp, G_perp]], G_perp = outer(t_e, t_e),
    t_e = (-s_e, c_e), in global coordinates: tension (N > 0) stiffens
    transverse motion, compression softens it. Batched over the leading
    dims of u_full via einsum.
    """
    coef = axial_forces(u_full).to(_DTYPE) / geometry.element_lengths().to(_DTYPE)
    return torch.einsum("...e,ekl->...kl", coef, _kg_pattern())


def modal_solve(u_full, n_modes=3):
    """Solve the prestressed eigenproblem (K + K_G) phi = w^2 M phi.

    u_full: (..., 14) nodal displacement (prestress state) -> per-batch
    (omegas (..., n_modes), modes (..., 11, n_modes)), frequencies ascending,
    modes M-orthonormal. Reduction to a symmetric standard problem:
    L_M = chol(M_red); A = L_M^-1 (K + K_G)_red L_M^-T; eigh(A);
    phi = L_M^-T @ eigvecs. All arithmetic stays float64.
    """
    K_red, L_M = _reduced_K_and_M_chol()
    S = K_red + reduce_matrix(assemble_KG(u_full))
    # A = L^-1 S L^-T, done with two triangular solves (both broadcast):
    #   Y1 = L^-1 S;  A = (L^-1 Y1^T)^T
    Y1 = torch.linalg.solve_triangular(L_M, S, upper=False)
    A = torch.linalg.solve_triangular(L_M, Y1.transpose(-1, -2), upper=False).transpose(-1, -2)
    A = 0.5 * (A + A.transpose(-1, -2))                     # kill roundoff asymmetry
    eigvals, eigvecs = torch.linalg.eigh(A)                 # ascending, orthonormal
    omegas = torch.sqrt(eigvals[..., :n_modes])
    modes = torch.linalg.solve_triangular(
        L_M.transpose(-1, -2), eigvecs[..., :n_modes], upper=True
    )
    return omegas, modes


def phase_fixed(phi):
    """Flip mode signs so each mode's largest-magnitude entry is positive.

    phi: (..., 11, n) with modes as columns -> same shape. Batched; a
    globally negated mode maps to the identical fixed mode.
    """
    lead_idx = phi.abs().argmax(dim=-2)                     # (..., n)
    lead = phi.gather(-2, lead_idx.unsqueeze(-2))           # (..., 1, n)
    sign = lead.sign()
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return phi * sign

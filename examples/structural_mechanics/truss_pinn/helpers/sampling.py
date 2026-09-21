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

"""Load-parameter sampling, anchor generation, and the frozen test set.

Design decisions:

- sample_batch draws (n_idx, theta, P) from ONE generator in a fixed order,
  so a seeded torch.Generator reproduces a batch exactly: n_idx is a
  discrete uniform over geometry.LOADABLE_NODES via randint, theta ~
  U[0, 2pi), P ~ U[-P_MAX, +P_MAX]. generator=None falls through to the
  default global RNG (torch's generator=None semantics). Determinism is
  process-stable and pinned to the environment (torch 2.5.1 CPU) -- it is
  not an absolute cross-version guarantee (RNG stream layout may change
  between torch releases).
- Load construction always goes through fem.nodal_load -- the single source
  of truth for the load direction -- so the LOAD VECTORS are bit-identical
  to what the documented single-case pipeline (solve_case) builds. The
  labels computed from them are batched: all N static solves run as one
  batched torch.linalg.solve and all N eigenproblems as one batched eigh
  inside fem.modal_solve. Batched LAPACK blocking differs from the
  single-case path, so anchor/test labels agree with solve_case only to
  roundoff (~1e-14 relative; measured worst: omega 5e-12 abs) -- downstream
  comparisons against recomputed single-case values need a tolerance
  >= ~1e-12 relative, never bitwise equality.
- Anchors form the explicit grid 5 loadable nodes x 4 directions x
  n_per_node |P| levels (linspace(0.1, 1.0) * P_MAX) x 2 signs = 120 cases
  at the defaults. The spec's 64 anchors could not cover all 5 loadable
  nodes; 120 is the smallest grid that does.
- The test set draws from the sample_batch distribution with a FROZEN seed
  (part of the contract: training and evaluation rebuild the SAME set).
  Disjointness from the anchor grid holds by construction on the continuous
  parameters: theta and P are drawn from continuous distributions against a
  finite grid, so collisions are measure-zero.
- Everything is float64 (fem discipline); anchors store f64 tensors.
"""

import math

import torch

from helpers import geometry
from helpers.fem import (
    assemble_K,
    axial_forces,
    modal_solve,
    nodal_load,
    phase_fixed,
    reduce_matrix,
    solve_static,
)

_DTYPE = torch.float64
_FREE = torch.tensor(geometry.FREE_DOFS)  # (11,) free-DOF indices
_DEFAULT_ANCHOR_THETAS = (-0.5 * math.pi, 0.0, 0.5 * math.pi, math.pi)


def sample_batch(n_cases, generator=None):
    """Draw `n_cases` load cases uniformly at random.

    n_idx: uniform over geometry.LOADABLE_NODES (discrete uniform, randint).
    theta: U[0, 2pi).  P: U[-P_MAX, +P_MAX]. All three draws consume
    `generator` (None = default global RNG) in the fixed order
    (n_idx, theta, P), so a seeded generator reproduces the batch.

    Returns dict with keys 'n_idx' ((n,) int64), 'theta', 'P' ((n,) f64).
    """
    nodes = torch.tensor(geometry.LOADABLE_NODES, dtype=torch.int64)
    n_idx = nodes[torch.randint(nodes.numel(), (n_cases,), generator=generator)]
    theta = torch.rand(n_cases, dtype=_DTYPE, generator=generator) * (2.0 * math.pi)
    P = (
        torch.rand(n_cases, dtype=_DTYPE, generator=generator) * 2.0 - 1.0
    ) * geometry.P_MAX
    return {"n_idx": n_idx, "theta": theta, "P": P}


def _fem_labels(n_idx, theta, P):
    """Batched FEM labels for aligned (N,) parameter tensors.

    Loads are built per case with fem.nodal_load (single source of truth
    for the load direction), then all N static solves and all N modal
    solves run as single batched calls. Returns dict with keys 'u_red'
    (N, 11), 'omega' (N, 3), 'phi' (N, 11, 3), all f64, modes phase-fixed.
    """
    f_red = torch.stack(
        [
            nodal_load(int(n), float(t), float(p))[_FREE]
            for n, t, p in zip(n_idx.tolist(), theta.tolist(), P.tolist())
        ]
    )  # (N, 11)
    K_red = reduce_matrix(assemble_K())  # (11, 11)
    u_red = solve_static(K_red, f_red)  # (N, 11)
    u14 = torch.zeros((u_red.shape[0], 2 * geometry.NODES_XY.shape[0]), dtype=_DTYPE)
    u14[:, _FREE] = u_red
    omegas, modes = modal_solve(u14)  # (N, 3), (N, 11, 3)
    return {"u_red": u_red, "omega": omegas, "phi": phase_fixed(modes)}


def make_anchors(n_per_node=3, thetas=_DEFAULT_ANCHOR_THETAS):
    """Anchor grid over the load box, every case labeled by the FEM.

    Grid: for each loadable node, each direction in `thetas`, each |P| in
    linspace(0.1, 1.0, n_per_node) * P_MAX, both signs -- N =
    len(LOADABLE_NODES) * len(thetas) * n_per_node * 2 = 120 at the
    defaults (5 nodes x 4 directions x 3 levels x 2 signs).

    Returns dict with keys 'params' ({'n_idx', 'theta', 'P'}), 'u_red'
    (N, 11), 'omega' (N, 3), 'phi' (N, 11, 3), all f64.
    """
    levels = torch.linspace(0.1, 1.0, n_per_node, dtype=_DTYPE) * geometry.P_MAX
    cases = [
        (n, float(th), sign * p)
        for n in geometry.LOADABLE_NODES
        for th in thetas
        for p in levels.tolist()
        for sign in (1.0, -1.0)
    ]
    params = {
        "n_idx": torch.tensor([c[0] for c in cases], dtype=torch.int64),
        "theta": torch.tensor([c[1] for c in cases], dtype=_DTYPE),
        "P": torch.tensor([c[2] for c in cases], dtype=_DTYPE),
    }
    return {
        "params": params,
        **_fem_labels(params["n_idx"], params["theta"], params["P"]),
    }


def make_test_set(n_cases=200, seed=20260920):
    """Frozen evaluation set: seeded sample_batch draws + FEM labels.

    The default seed is part of the contract -- training and evaluation
    rebuild the SAME test set deterministically. The draws follow the
    sample_batch distribution; collisions with the finite anchor grid on
    the continuous parameters (theta, P) are measure-zero by construction.

    Returns dict with keys 'n_idx' ((n,) int64), 'theta', 'P', 'u_red'
    (n, 11), 'omega' (n, 3), 'phi' (n, 11, 3), all tensors f64 except n_idx.
    """
    g = torch.Generator().manual_seed(seed)
    params = sample_batch(n_cases, generator=g)
    return {**params, **_fem_labels(params["n_idx"], params["theta"], params["P"])}


def solve_case(n_idx, theta, P):
    """Single-case FEM pipeline: nodal_load -> solve_static -> modal_solve.

    Accepts python scalars or 0/1-dim single-element tensors (promoted via
    int()/float()). Returns dict with keys 'u14' (14,), 'u_red' (11,),
    'N' (11,), 'omegas' (3,), 'modes' (11, 3), all f64, modes phase-fixed.
    """
    f = nodal_load(int(n_idx), float(theta), float(P))
    u_red = solve_static(reduce_matrix(assemble_K()), f[_FREE])  # (11,)
    u14 = torch.zeros(2 * geometry.NODES_XY.shape[0], dtype=_DTYPE)
    u14[_FREE] = u_red
    N = axial_forces(u14)
    omegas, modes = modal_solve(u14)
    return {
        "u14": u14,
        "u_red": u_red,
        "N": N,
        "omegas": omegas,
        "modes": phase_fixed(modes),
    }

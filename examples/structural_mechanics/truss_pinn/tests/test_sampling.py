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

# tests/test_sampling.py
"""Sampling / anchor / test-set tests: draw distribution, anchor grid
coverage + disjointness, FEM-label correctness, deterministic rebuild.

Contracts under test (spec sections 4.2 items 2-3 and 4.3):
- sample_batch draws n_idx uniform over geometry.LOADABLE_NODES, theta ~
  U[0, 2pi), P ~ U[-P_MAX, P_MAX]; all draws come from the caller's
  generator (None = global default RNG).
- make_anchors builds the 5 loadable nodes x 4 directions x 3 |P| levels
  x 2 signs = 120 anchor grid, every anchor labeled by the batched FEM
  pipeline in float64.
- make_test_set draws the SAME distribution from a FIXED seed so training
  and evaluation rebuild bit-identical tensors. Disjointness from the
  anchors holds by construction on the continuous parameters: theta and P
  are continuous draws against a finite grid, so collisions are
  measure-zero -- the frozen-seed set is checked to keep min distance > 0.
- solve_case is the single-case FEM pipeline and must reproduce the manual
  nodal_load -> solve_static -> modal_solve -> phase_fixed chain exactly.
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

_FREE = torch.tensor(geometry.FREE_DOFS)
_K_RED = reduce_matrix(assemble_K())


def _manual_pipeline(n, theta, P):
    """Reference single-case FEM chain, returns (u14, u_red, N, omegas, modes)."""
    f = nodal_load(n, theta, P)
    u_red = solve_static(_K_RED, f[_FREE])
    u14 = torch.zeros(14, dtype=torch.float64)
    u14[_FREE] = u_red
    N = axial_forces(u14)
    omegas, modes = modal_solve(u14)
    return u14, u_red, N, omegas, phase_fixed(modes)


def test_anchor_grid_covers_and_disjoint_from_test():
    from helpers.sampling import make_anchors, make_test_set

    a = make_anchors()
    p = a["params"]
    # grid shape: 5 loadable nodes x 4 thetas x 3 |P| levels x 2 signs
    assert p["n_idx"].shape == (120,) and p["n_idx"].dtype == torch.int64
    assert a["u_red"].shape == (120, 11) and a["u_red"].dtype == torch.float64
    assert a["omega"].shape == (120, 3) and a["omega"].dtype == torch.float64
    assert a["phi"].shape == (120, 11, 3) and a["phi"].dtype == torch.float64
    assert torch.isfinite(a["u_red"]).all()
    assert torch.isfinite(a["omega"]).all() and (a["omega"] > 0).all()
    assert torch.isfinite(a["phi"]).all()
    # coverage: every loadable node, all 4 directions, the 3 |P| levels, both signs
    assert set(p["n_idx"].tolist()) == set(geometry.LOADABLE_NODES)
    assert sorted(set(p["theta"].tolist())) == sorted(
        [-math.pi / 2, 0.0, math.pi / 2, math.pi])
    expected_p = sorted(set(
        (torch.linspace(0.1, 1.0, 3, dtype=torch.float64) * geometry.P_MAX).tolist()))
    assert sorted(set(p["P"].abs().tolist())) == expected_p
    assert (p["P"] > 0).any() and (p["P"] < 0).any()

    t = make_test_set()
    for key in ("n_idx", "theta", "P", "u_red", "omega", "phi"):
        assert key in t
    assert t["n_idx"].shape == (200,) and t["n_idx"].dtype == torch.int64
    assert t["theta"].shape == (200,) and t["P"].shape == (200,)
    assert t["u_red"].shape == (200, 11)
    assert t["omega"].shape == (200, 3) and t["phi"].shape == (200, 11, 3)
    for key in ("u_red", "omega", "phi"):
        assert torch.isfinite(t[key]).all()
    assert (t["omega"] > 0).all()
    # disjointness on the continuous parameters: no test draw hits the grid
    dist = ((t["theta"][:, None] - p["theta"][None, :]).abs()
            + (t["P"][:, None] - p["P"][None, :]).abs() / geometry.P_MAX)
    assert dist.min() > 0.0


def test_solve_case_matches_pipeline():
    from helpers.sampling import solve_case

    out = solve_case(2, -1.0, 300.0)
    assert out["u14"].shape == (14,) and out["u_red"].shape == (11,)
    assert out["N"].shape == (11,)
    assert out["omegas"].shape == (3,) and out["modes"].shape == (11, 3)
    assert all(v.dtype == torch.float64 for v in out.values())
    assert (out["omegas"] > 0).all()
    assert torch.isfinite(out["modes"]).all()

    u14, u_red, N, omegas, modes = _manual_pipeline(2, -1.0, 300.0)
    for key, ref in (("u14", u14), ("u_red", u_red), ("N", N),
                     ("omegas", omegas), ("modes", modes)):
        assert torch.allclose(out[key], ref, rtol=0.0, atol=1e-12)

    # robust promotion: 0-dim and 1-dim tensor inputs land on the same case
    for wrap in (torch.tensor, lambda x: torch.tensor([x])):
        out_t = solve_case(wrap(2), wrap(-1.0), wrap(300.0))
        assert out_t["omegas"].shape == (3,) and out_t["modes"].shape == (11, 3)
        assert torch.equal(out_t["omegas"], omegas)
        assert torch.equal(out_t["modes"], modes)
        assert torch.equal(out_t["N"], N)


def test_sample_batch_distribution():
    from helpers.sampling import sample_batch

    n = 5000
    g = torch.Generator().manual_seed(7)
    b = sample_batch(n, generator=g)
    assert b["n_idx"].shape == (n,) and b["n_idx"].dtype == torch.int64
    assert b["theta"].shape == (n,) and b["theta"].dtype == torch.float64
    assert b["P"].shape == (n,) and b["P"].dtype == torch.float64
    # discrete uniform over the loadable nodes, each within +-20% of expected
    assert set(b["n_idx"].tolist()) <= set(geometry.LOADABLE_NODES)
    counts = torch.bincount(b["n_idx"], minlength=7)
    expected = n / len(geometry.LOADABLE_NODES)
    for node in geometry.LOADABLE_NODES:
        assert abs(counts[node].item() - expected) < 0.2 * expected
    # theta ~ U[0, 2pi), P ~ U[-P_MAX, P_MAX]
    assert (b["theta"] >= 0.0).all() and (b["theta"] < 2.0 * math.pi).all()
    assert (b["P"] >= -geometry.P_MAX).all() and (b["P"] <= geometry.P_MAX).all()
    # observed range covers >90% of each interval (min/max near the bounds)
    assert (b["theta"].max() - b["theta"].min()).item() > 0.9 * 2.0 * math.pi
    assert (b["P"].max() - b["P"].min()).item() > 0.9 * 2.0 * geometry.P_MAX
    # generator=None falls back to the default RNG without changing shapes
    b0 = sample_batch(8)
    assert b0["n_idx"].shape == (8,) and b0["theta"].shape == (8,)


def test_test_set_deterministic():
    from helpers.sampling import make_test_set

    t1 = make_test_set()
    t2 = make_test_set(n_cases=200, seed=20260920)
    for key in ("n_idx", "theta", "P", "u_red", "omega", "phi"):
        assert torch.equal(t1[key], t2[key])

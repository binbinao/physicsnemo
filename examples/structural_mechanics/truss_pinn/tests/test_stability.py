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

# tests/test_stability.py
"""Stability-scan tests: critical load per load path, and the frozen P_MAX.

Physics contract (established in the Task 3 fix round): stability is decided
ONLY by positivity of the reduced tangent stiffness K_red + K_G_red.
Frequency trends are NOT a stability indicator for this truss -- a downward
load stiffens omega_1 through the tension diagonals while the compressed
chords soften, so the scan tracks min eigvalsh(tangent), never omega.

Sufficiency of the edge checks: statics is linear, so along a fixed ray
(n, theta) the tangent is AFFINE in P: u(P) = P u(1), hence
tangent(P) = K_red + P K_G(u(1)). lambda_min of an affine pencil is a
pointwise minimum of affine functions of P, i.e. concave; a concave function
positive at both segment ends is positive throughout, so checking the
+-P_MAX edges (plus P = 0, where K_red is PD) covers the whole load box.
"""

import math

import torch

from helpers import geometry
from helpers.fem import (
    assemble_K,
    assemble_KG,
    nodal_load,
    reduce_matrix,
    solve_static,
)

_FREE = torch.tensor(geometry.FREE_DOFS)
_K_RED = reduce_matrix(assemble_K())


def _tangent_min_eig(P, n, theta):
    """Smallest eigenvalue of the reduced tangent stiffness at (n, theta, P)."""
    u_red = solve_static(_K_RED, nodal_load(n, theta, P)[_FREE])
    u14 = torch.zeros(14, dtype=torch.float64)
    u14[_FREE] = u_red
    return float(torch.linalg.eigvalsh(_K_RED + reduce_matrix(assemble_KG(u14)))[0])


def test_p_crit_positive_and_finite():
    """The default scan yields a finite P_crit of plausible magnitude (1e4 to 2e5 N)."""
    # default path: P downward at node 1. Measured P_crit ~ 8.44e4 N, so the
    # brief's pre-scan guess "p < 5e4" is superseded by the measured range
    # (the default scan ceiling p_hi = 2e5 was raised accordingly).
    from helpers.stability import scan_p_crit

    p = scan_p_crit()
    assert p == p  # NaN guard: exhausted scan -> NaN
    assert 1.0e4 < p < 2.0e5  # plausible magnitude, inside scan range


def test_p_crit_brackets_tangent_sign_change():
    """The tangent min eigenvalue is positive just below P_crit and nonpositive just above."""
    # P_crit is BY DEFINITION where the tangent stiffness loses positivity:
    # just below it the min eigenvalue must be positive, just above <= 0.
    from helpers.stability import scan_p_crit

    p = scan_p_crit()
    assert _tangent_min_eig(0.98 * p, 1, -math.pi / 2) > 0.0
    assert _tangent_min_eig(1.02 * p, 1, -math.pi / 2) <= 0.0


def test_p_crit_exhausted_scan_returns_nan():
    """An exhausted scan (ceiling below the transition) returns NaN, never a wrong number."""
    # a scan ceiling far below the transition must return NaN (documented
    # contract), never a silently wrong number.
    from helpers.stability import scan_p_crit

    p = scan_p_crit(n=1, theta=-math.pi / 2, p_hi=1.0e3)
    assert p != p  # NaN


def test_frozen_p_max_stays_stable():
    """The tangent stiffness stays PD across the full frozen load envelope, both signs."""
    # self-test: (K + K_G) min eig > 0 across the FULL frozen load range.
    # Runtime envelope exactly as train.py would freeze it for this path.
    from helpers.stability import scan_p_crit

    default = geometry.P_MAX
    try:
        p_crit = scan_p_crit()
        geometry.P_MAX = 0.3 * p_crit
        # worst load path of the scan itself
        assert _tangent_min_eig(geometry.P_MAX, 1, -math.pi / 2) > 0.0
        # spot-check other nodes/directions at the envelope edges (+-P_MAX)
        for n in (2, 4, 6):
            for theta in (-math.pi / 2, 0.0, math.pi / 2, math.pi):
                assert _tangent_min_eig(geometry.P_MAX, n, theta) > 0.0
                assert _tangent_min_eig(-geometry.P_MAX, n, theta) > 0.0
    finally:
        geometry.P_MAX = default


def test_p_max_frozen_matches_scan():
    """The shipped geometry.P_MAX reproduces freeze_p_max() within 1% and assigns at runtime."""
    # the hardcoded geometry.P_MAX default must reproduce freeze_p_max()
    # (the node x direction grid min times safety) within 1%, and the
    # freeze must assign geometry.P_MAX at runtime.
    from helpers.stability import freeze_p_max

    default = geometry.P_MAX
    try:
        p = freeze_p_max()
        assert geometry.P_MAX == p  # runtime assignment happened
        assert abs(p - default) <= 0.01 * abs(default)  # refrozen default within 1%
        # the shipped envelope keeps the tangent PD on the grid's own worst
        # load path (node 6, theta = pi) at both edges
        assert _tangent_min_eig(default, 6, math.pi) > 0.0
        assert _tangent_min_eig(-default, 6, math.pi) > 0.0
    finally:
        geometry.P_MAX = default

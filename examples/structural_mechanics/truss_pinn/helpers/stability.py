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

"""Linear-stability scan: critical load per load path, and the P_MAX freeze.

Stability criterion (the ONLY one used here): the reduced tangent stiffness
K_red + K_G_red must stay positive definite, i.e. min eigvalsh > 0. Frequency
trends are NOT a stability indicator for this truss -- a downward load can
raise omega_1 through geometric stiffening of the tension diagonals while the
compressed chords soften, so the scan never looks at omega.

Load-path dependence: P_crit depends on (node, direction). Along a fixed ray
the statics is linear, u(P) = P u(1), so the tangent is an affine pencil
K_red + P K_G(u(1)); its smallest eigenvalue is a pointwise minimum of
affine functions of P, hence concave. Starting positive at P = 0 (K_red is
PD), a concave lambda_min crosses zero exactly once, so "first sign change"
is the unique critical load and the tangent stays indefinite beyond it.

Scan contract: scan_p_crit linearly scans [0, p_hi] in n_steps steps,
bisects 40 iterations between the last stable and first unstable load, and
returns that midpoint as P_crit. If no load in [0, p_hi] destabilizes the
truss the scan is exhausted and float('nan') is returned so the caller can
detect it (freeze_p_max skips such paths; they are more stable than any
finite grid P_crit). The default ceiling p_hi = 2e5 N is set above the
measured worst path (node 1, theta = -pi/2: P_crit ~ 8.44e4 N; grid minimum
node 6, theta = pi: P_crit ~ 3.96e4 N) -- an earlier provisional 5e4 ceiling
from the pre-scan plan brackets nothing on the default path.

Frozen envelope (this module, safety = 0.3, 5 loadable nodes x 8 evenly
spaced directions over [0, 2pi); a grid over the full circle covers both
load signs): P_MAX = 0.3 * min(P_crit) ~= 11871.275 N. geometry.P_MAX ships
with this value; freeze_p_max re-derives it at runtime (train.py startup
when cfg.freeze_p_max = true) and assigns geometry.P_MAX in place.
"""
import math

import torch

from helpers import geometry
from helpers.fem import assemble_K, assemble_KG, reduce_matrix, solve_static

_FREE = torch.tensor(geometry.FREE_DOFS)
_N_DOF = 2 * geometry.NODES_XY.shape[0]  # 14
_BISECT_ITERS = 40

_K_RED = None


def _k_red():
    """Cached reduced linear stiffness (11, 11) float64."""
    global _K_RED
    if _K_RED is None:
        _K_RED = reduce_matrix(assemble_K())
    return _K_RED


def _tangent_min_eigs(P, n, theta):
    """Min eigenvalue of the reduced tangent stiffness at loads P -> (...,).

    P: scalar or (...,) tensor of load magnitudes; n: loaded node; theta:
    load direction from +x. Batched over the leading dims of P: static
    solve -> expand to full dofs -> geometric stiffness -> eigenvalues.
    """
    P = torch.as_tensor(P, dtype=torch.float64)
    n = int(n)
    theta = float(theta)
    f = torch.zeros(*P.shape, _N_DOF, dtype=torch.float64)
    f[..., 2 * n] = P * math.cos(theta)
    f[..., 2 * n + 1] = P * math.sin(theta)
    u = torch.zeros(*P.shape, _N_DOF, dtype=torch.float64)
    u[..., _FREE] = solve_static(_k_red(), f[..., _FREE])
    tangent = _k_red() + reduce_matrix(assemble_KG(u))
    return torch.linalg.eigvalsh(tangent)[..., 0]


def _stable(P, n, theta):
    """True iff the tangent stiffness is PD at (n, theta, P). Scalar."""
    lam = _tangent_min_eigs(P, n, theta)
    return bool(torch.isfinite(lam).all() and (lam > 0).all())


def scan_p_crit(theta: float = -math.pi / 2, n: int = 1, p_hi: float = 2.0e5,
                n_steps: int = 400) -> float:
    """Critical load of load path (node n, direction theta) -> float.

    Coarse-to-fine: linear scan of [0, p_hi] in n_steps steps tracking the
    sign of min eigvalsh(K_red + K_G_red); on the first non-positive (or
    non-finite) value, bisect _BISECT_ITERS times between the last stable
    and first unstable load. Returns the bracket midpoint. Exhausted scan
    (no destabilizing load within [0, p_hi]) -> float('nan').

    P_crit is load-path-dependent; see the module docstring for the
    conservativeness protocol used by freeze_p_max.
    """
    theta = float(theta)
    n = int(n)
    p_hi = float(p_hi)
    n_steps = int(n_steps)
    Ps = torch.linspace(0.0, p_hi, n_steps + 1, dtype=torch.float64)
    eigs = _tangent_min_eigs(Ps, n, theta)
    bad = torch.nonzero(~((eigs > 0) & torch.isfinite(eigs)))
    if bad.numel() == 0:
        return float("nan")                     # scan exhausted: caller detects
    i = int(bad[0, 0])
    if i == 0:
        return 0.0                              # unstable already at P = 0
    lo, hi = float(Ps[i - 1]), float(Ps[i])     # last stable / first unstable
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        if _stable(mid, n, theta):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def freeze_p_max(safety: float = 0.3, n_theta: int = 8, nodes=None) -> float:
    """Conservative P_MAX over the loadable-node x direction grid -> float.

    Scans each (node, theta) on nodes (default geometry.LOADABLE_NODES) x
    n_theta directions evenly spaced over [0, 2pi) -- a full circle, so both
    load signs are covered -- takes the minimum finite P_crit, multiplies by
    `safety`, assigns geometry.P_MAX in place, and returns the value. Paths
    whose scan is exhausted (NaN: no buckling within the scan range) are
    skipped; they are strictly more stable than the retained minimum.
    """
    if nodes is None:
        nodes = geometry.LOADABLE_NODES
    p_crits = []
    for n in nodes:
        for k in range(int(n_theta)):
            theta = 2.0 * math.pi * k / int(n_theta)
            p = scan_p_crit(theta=theta, n=n)
            if p == p:                           # skip NaN (exhausted scan)
                p_crits.append(p)
    if not p_crits:
        raise RuntimeError(
            "no load path lost tangent stiffness positivity inside the "
            "scan range; cannot freeze a stability envelope"
        )
    p_max = float(safety) * min(p_crits)
    geometry.P_MAX = p_max
    return p_max

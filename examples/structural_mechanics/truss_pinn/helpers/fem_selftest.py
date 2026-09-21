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

"""FEM physics gate for training startup (plain-assert port of the test suite).

run_fem_self_tests() bundles the load-bearing checks of tests/test_fem.py,
tests/test_stability.py and tests/test_truss_pde.py so train.py can refuse
to train on a broken FEM stack without shelling out to pytest:

1. K symmetric, PSD on the full space, PD after support elimination
2. M symmetric, positive diagonal, rigid-translation invariant
3. modal_solve modes M-orthonormal with strictly positive frequencies
4. first-order prestress shift of omega_1 odd in P
5. reduced tangent stiffness PD at the frozen P_MAX envelope edges
   (worst grid path node 6 / theta = pi, plus edge spot checks)
6. compiled sympy residual equals K_red @ u on a random batch

All random draws use device-local generators: the global torch RNG belongs
to train.py (model initialization) and must not be disturbed by this gate.
"""

import math

import torch

from helpers import geometry
from helpers.fem import (
    assemble_K,
    assemble_KG,
    assemble_M,
    modal_solve,
    nodal_load,
    reduce_matrix,
    solve_static,
)
from helpers.truss_pde import make_residual_fn

_N_DOF = 2 * geometry.NODES_XY.shape[0]  # 14
_FREE = torch.tensor(geometry.FREE_DOFS)  # (11,) free-DOF indices


def _check_stiffness():
    """K symmetric and PSD on the full space, PD after support elimination."""
    K = assemble_K()
    assert K.shape == (_N_DOF, _N_DOF) and K.dtype == torch.float64
    assert torch.allclose(K, K.T, atol=1e-9), "K is not symmetric"
    assert torch.linalg.eigvalsh(K).min() > -1e-6, "K is not PSD"
    K_red = reduce_matrix(K)
    assert K_red.shape == (11, 11), "wrong reduced stiffness shape"
    assert torch.linalg.eigvalsh(K_red).min() > 0.0, "K_red is not PD"


def _check_mass():
    """M symmetric with positive diagonal; rigid translation sees rho*A*sum(L)."""
    M = assemble_M()
    assert M.shape == (_N_DOF, _N_DOF) and M.dtype == torch.float64
    assert torch.allclose(M, M.T, atol=1e-9), "M is not symmetric"
    assert torch.diag(M).min() > 0.0, "M diagonal is not positive"
    L = geometry.element_lengths().to(torch.float64)
    total = geometry.RHO * geometry.AREA * L.sum()
    ones = torch.ones(_N_DOF, dtype=torch.float64)
    # ones = rigid translation at speed sqrt(2): 1^T M 1 = 2 * rho*A*sum(L)
    got = (ones @ M @ ones).item()
    want = 2.0 * total.item()
    assert abs(got - want) < 1e-9 * max(1.0, want), (
        f"M not rigid-translation invariant: 1^T M 1 = {got:.6e}, want {want:.6e}"
    )


def _check_modal_orthonormality():
    """modal_solve modes M-orthonormal with strictly positive frequencies."""
    u0 = torch.zeros(_N_DOF, dtype=torch.float64)
    omegas, modes = modal_solve(u0)
    assert omegas.shape == (3,) and modes.shape == (11, 3), "wrong modal shapes"
    M_red = reduce_matrix(assemble_M())
    gram = modes.T @ M_red @ modes
    assert torch.allclose(gram, torch.eye(3, dtype=torch.float64), atol=1e-8), (
        "modes are not M-orthonormal"
    )
    assert bool((omegas > 0).all()), "non-positive modal frequency"


def _check_prestress_odd_in_p():
    """First-order prestress shift of omega_1 flips sign when P flips sign."""
    K_red = reduce_matrix(assemble_K())

    def omega1(P):
        f14 = nodal_load(1, -0.5 * math.pi, P)
        u14 = torch.zeros(_N_DOF, dtype=torch.float64)
        u14[_FREE] = solve_static(K_red, f14[_FREE])
        w, _ = modal_solve(u14)
        return w[0]

    w0 = omega1(0.0)
    w_plus, w_minus = omega1(1000.0), omega1(-1000.0)
    assert (w_plus - w0) * (w_minus - w0) < 0.0, (
        "prestress frequency shift is not odd in P"
    )
    assert abs(w_plus - w0) > 1e-7 * w0 and abs(w_minus - w0) > 1e-7 * w0, (
        "prestress frequency shift vanished"
    )


def _tangent_min_eig(P, n, theta):
    """Min eigenvalue of the reduced tangent stiffness at (n, theta, P)."""
    K_red = reduce_matrix(assemble_K())
    f14 = nodal_load(n, theta, P)
    u14 = torch.zeros(_N_DOF, dtype=torch.float64)
    u14[_FREE] = solve_static(K_red, f14[_FREE])
    tangent = K_red + reduce_matrix(assemble_KG(u14))
    return float(torch.linalg.eigvalsh(tangent)[0])


def _check_tangent_pd_at_p_max():
    """Reduced tangent stiffness stays PD at the frozen envelope edges."""
    p_max = float(geometry.P_MAX)
    # worst path of the freeze grid (node 6, theta = pi): both load signs
    assert _tangent_min_eig(p_max, 6, math.pi) > 0.0, (
        "tangent not PD at +P_MAX on the worst grid path (node 6, theta=pi)"
    )
    assert _tangent_min_eig(-p_max, 6, math.pi) > 0.0, (
        "tangent not PD at -P_MAX on the worst grid path (node 6, theta=pi)"
    )
    # edge spot checks on the remaining loadable nodes
    for n in (1, 2, 4):
        for theta in (-0.5 * math.pi, 0.0, 0.5 * math.pi, math.pi):
            assert _tangent_min_eig(p_max, n, theta) > 0.0, (
                f"tangent not PD at +P_MAX (node {n}, theta={theta:.4f})"
            )
            assert _tangent_min_eig(-p_max, n, theta) > 0.0, (
                f"tangent not PD at -P_MAX (node {n}, theta={theta:.4f})"
            )


def _check_residual_matches_assembled_k():
    """Compiled sympy residual equals K_red @ u on a random batch."""
    residual = make_residual_fn()
    K_red = reduce_matrix(assemble_K())
    # Device-local seeded draws: never touch the global torch RNG.
    gen = torch.Generator(device=torch.empty(0).device).manual_seed(0)
    u = torch.randn(64, 11, generator=gen)  # float32 on device
    r = residual(u)
    expected = u.to(torch.float64) @ K_red.T
    assert r.shape == (64, 11), "wrong residual shape"
    assert r.dtype == u.dtype, "residual must preserve the input dtype"
    assert torch.allclose(
        r.to(torch.float64),
        expected,
        rtol=1e-5,
        atol=1e-4 * expected.abs().max(),
    ), "symbolic residual disagrees with K_red @ u"


def run_fem_self_tests() -> None:
    """Run the FEM physics gate; raise RuntimeError on any failure."""
    checks = (
        ("stiffness", _check_stiffness),
        ("mass", _check_mass),
        ("modal orthonormality", _check_modal_orthonormality),
        ("prestress odd in P", _check_prestress_odd_in_p),
        ("tangent PD at P_MAX", _check_tangent_pd_at_p_max),
        ("symbolic residual", _check_residual_matches_assembled_k),
    )
    try:
        for name, check in checks:
            check()
    except AssertionError as exc:
        raise RuntimeError(f"FEM self-test '{name}' failed: {exc}") from exc

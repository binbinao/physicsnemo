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

# tests/test_fem.py
import math

import torch

from helpers import geometry
from helpers.fem import (
    assemble_K,
    assemble_M,
    reduce_matrix,
    solve_static,
    nodal_load,
    axial_forces,
)

_FREE = torch.tensor(geometry.FREE_DOFS)


def test_K_symmetric_positive_semidefinite():
    """K is 14x14 float64, symmetric, PSD on the full space, PD after elimination."""
    K = assemble_K()
    assert K.shape == (14, 14)
    assert K.dtype == torch.float64
    assert torch.allclose(K, K.T, atol=1e-9)
    eig = torch.linalg.eigvalsh(K)
    assert eig.min() > -1e-6  # PSD on full space (3 constraints)
    K_red = reduce_matrix(K)
    assert K_red.shape == (11, 11)
    assert torch.linalg.eigvalsh(K_red).min() > 0.0  # PD after elimination


def test_M_positive_definite_symmetric():
    """M is 14x14 float64, symmetric, with strictly positive diagonal."""
    M = assemble_M()
    assert M.shape == (14, 14)
    assert M.dtype == torch.float64
    assert torch.allclose(M, M.T, atol=1e-9)
    assert torch.diag(M).min() > 0.0


def test_solve_static_identity_on_assembled_system():
    """A random free-DOF load is exactly reproduced by K_red @ u."""
    # identity check of solve_static against the assembled matrix: a random
    # load at the free DOFs must be reproduced by K_red @ u exactly
    K = assemble_K()
    K_red = reduce_matrix(K)
    f = torch.randn(11, dtype=torch.float64)
    u = solve_static(K_red, f)
    assert torch.allclose(K_red @ u, f, atol=1e-8)


def test_solve_static_batched():
    """Batched solve_static applies K_red elementwise across a (5, 7) batch."""
    K_red = reduce_matrix(assemble_K())
    f_batch = torch.randn(5, 7, 11, dtype=torch.float64)
    u_batch = solve_static(K_red.expand(5, 7, 11, 11), f_batch)
    assert u_batch.shape == (5, 7, 11)
    # broadcast the shared operator over the batch: (11,11) @ (5,7,11) is invalid
    assert torch.allclose(
        torch.einsum("ij,...j->...i", K_red, u_batch), f_batch, atol=1e-8
    )


def test_nodal_load_direction():
    """A downward load at node 2 lands on its y DOF and nowhere else."""
    f = nodal_load(2, -math.pi / 2, 100.0)
    assert f.shape == (14,)
    assert f.dtype == torch.float64
    assert abs(f[4].item() - 100.0 * math.cos(-math.pi / 2)) < 1e-9  # ~0
    assert abs(f[5].item() + 100.0) < 1e-9  # -P downward
    mask = torch.ones(14, dtype=torch.bool)
    mask[[4, 5]] = False
    assert torch.allclose(f[mask], torch.zeros(int(mask.sum()), dtype=f.dtype))


def test_static_equilibrium_and_axial_force_symmetry():
    """A downward load at node 1 balances: member forces + reactions = applied load."""
    # vertical load P down at node 1 (dofs 2,3 per f[2n], f[2n+1])
    K = assemble_K()
    K_red = reduce_matrix(K)
    f14 = nodal_load(1, -math.pi / 2, 100.0)
    f_red = f14[_FREE]
    u_red = solve_static(K_red, f_red)
    u14 = torch.zeros(14, dtype=torch.float64)
    u14[_FREE] = u_red
    N = axial_forces(u14)
    assert N.shape == (11,)
    # residual (K @ u - f) is nonzero only at support dofs
    reactions = (K @ u14) - f14
    assert torch.allclose(
        reactions[[2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]],
        torch.zeros(11, dtype=torch.float64),
        atol=1e-6,
    )
    # total vertical equilibrium: R_y(node0) + R_y(node3) + f_y = 0
    # (applied load at node 1: f_y = f14[3])
    assert abs(reactions[1] + reactions[7] + f14[3]) < 1e-6


def test_axial_force_closed_form_single_member():
    """Member (0,1) axial force equals EA/L * dx for a unit x displacement."""
    # hand check: displace node 1 by +dx in x only; member (0,1) axial force = EA/L * dx
    dx = 1e-4
    u14 = torch.zeros(14, dtype=torch.float64)
    u14[2] = dx  # node 1 x
    N = axial_forces(u14)
    L01 = 2.0
    k01 = geometry.E * geometry.AREA / L01
    assert abs(N[0].item() - k01 * dx) < 1e-6 * k01 * dx + 1e-9


def test_axial_forces_batched():
    """Batched axial_forces reproduces the single-member closed form per row."""
    dx = 1e-4
    u_batch = torch.zeros(3, 14, dtype=torch.float64)
    u_batch[:, 2] = dx
    N = axial_forces(u_batch)
    assert N.shape == (3, 11)
    k01 = geometry.E * geometry.AREA / 2.0
    assert torch.allclose(N[:, 0], torch.full((3,), k01 * dx, dtype=torch.float64))


def test_M_rigid_translation_invariance():
    """Mass energy of a rigid translation is isotropic: rho*A*sum(L) in any direction."""
    # consistent mass must be translation-invariant and isotropic:
    # rigid translation u gives u^T M u = rho*A*sum(L) for ANY direction,
    # and 1^T M 1 = 2 * rho*A*sum(L) (ones = rigid translation at speed sqrt(2)).
    M = assemble_M()
    L = geometry.element_lengths().to(torch.float64)
    total = geometry.RHO * geometry.AREA * L.sum()
    for vx, vy in [(1.0, 0.0), (0.0, 1.0), (0.6, 0.8)]:
        u = torch.zeros(14, dtype=torch.float64)
        u[0::2] = vx
        u[1::2] = vy
        assert abs((u @ M @ u).item() - total.item()) < 1e-9 * max(1.0, total.item())
    ones = torch.ones(14, dtype=torch.float64)
    # ones-vector is a rigid translation at speed sqrt(2): 1^T M 1 = 2 * total
    assert abs((ones @ M @ ones).item() - 2 * total.item()) < 1e-9 * max(
        1.0, total.item()
    )


def test_modal_M_orthonormality():
    """Modes are M-orthonormal with strictly positive frequencies."""
    # self-test 3: modes must be M-orthonormal with strictly positive freqs
    from helpers.fem import modal_solve

    u0 = torch.zeros(14, dtype=torch.float64)
    omegas, modes = modal_solve(u0)
    assert omegas.shape == (3,) and modes.shape == (11, 3)
    M_red = reduce_matrix(assemble_M())
    Gram = modes.T @ M_red @ modes  # (3, 3)
    assert torch.allclose(Gram, torch.eye(3, dtype=torch.float64), atol=1e-8)
    assert (omegas > 0).all()


def test_prestress_changes_frequency_symmetrically_opposite():
    """The first-order prestress shift of omega_1 flips sign when P flips (odd in P)."""
    # self-test 4 (robust form): member forces are odd in P and prestress
    # stiffening is linear in N_e, so the first-order shift of the
    # fundamental frequency must flip sign when P flips sign (odd in P).
    from helpers.fem import modal_solve

    K_red = reduce_matrix(assemble_K())

    def omega_at(P):
        f14 = nodal_load(1, -math.pi / 2, P)  # downward load at node 1
        f_red = f14[_FREE]
        u_red = solve_static(K_red, f_red)
        u14 = torch.zeros(14, dtype=torch.float64)
        u14[_FREE] = u_red
        w, _ = modal_solve(u14)
        return w[0]

    w0 = omega_at(0.0)
    w_plus, w_minus = omega_at(1000.0), omega_at(-1000.0)
    # linear prestress theory: first-order change is odd in P
    assert (w_plus - w0) * (w_minus - w0) < 0.0
    assert abs(w_plus - w0) > 1e-7 * w0 and abs(w_minus - w0) > 1e-7 * w0


def test_phase_fixed_makes_max_entry_positive():
    """phase_fixed makes each mode's largest-magnitude entry positive and idempotent."""
    # self-test 5: phase fixing makes each mode's largest-magnitude entry
    # positive and returns the identical mode for a globally negated input
    from helpers.fem import modal_solve, phase_fixed

    _, modes = modal_solve(torch.zeros(14, dtype=torch.float64))
    modes_fixed = phase_fixed(modes)
    modes_neg = phase_fixed(-modes)
    cols = torch.arange(3)
    assert (modes_fixed[modes_fixed.abs().argmax(dim=0), cols] > 0).all()
    assert (modes_neg[modes_neg.abs().argmax(dim=0), cols] > 0).all()
    # flipping the input mode must yield the same fixed mode
    assert torch.allclose(modes_fixed, modes_neg, rtol=0.0, atol=1e-12)


def test_modal_solve_batched_matches_single():
    """Batched modal_solve rows reproduce the single-case solve exactly."""
    # batched eigenpath: 4 identical rows must reproduce the single solve
    from helpers.fem import modal_solve

    K_red = reduce_matrix(assemble_K())
    f14 = nodal_load(1, -math.pi / 2, 750.0)
    u_red = solve_static(K_red, f14[_FREE])
    u14 = torch.zeros(14, dtype=torch.float64)
    u14[_FREE] = u_red
    omegas1, modes1 = modal_solve(u14)
    u_batch = u14.expand(4, 14).contiguous()
    omegas_b, modes_b = modal_solve(u_batch)
    assert omegas_b.shape == (4, 3) and modes_b.shape == (4, 11, 3)
    assert torch.allclose(omegas_b, omegas1.expand(4, -1), rtol=0.0, atol=1e-12)
    assert torch.allclose(modes_b, modes1.expand(4, -1, -1), rtol=0.0, atol=1e-12)


def _force_state(target_N):
    """Full displacement (14,) whose member forces match target_N (11,).

    The compatibility map u -> N is linear and the Pratt truss is
    statically determinate (11 members + 3 reactions = 14 = 2*7 dofs),
    so the map has full row rank and any force vector is reachable.
    Lets the K_G tests manufacture prestress states member by member.
    """
    from helpers.fem import axial_forces

    cols = []
    for d in range(14):
        e = torch.zeros(14, dtype=torch.float64)
        e[d] = 1.0
        cols.append(axial_forces(e))
    B = torch.stack(cols, dim=1)  # (11, 14)
    # lstsq on the underdetermined system is nondeterministic in this torch
    # build; B B^T is SPD and well-conditioned (cond ~45), so solve the
    # normal equations instead — deterministic and exact to ~3e-13.
    return B.T @ torch.linalg.solve(B @ B.T, target_N)


def test_KG_horizontal_member_matches_classic_pattern():
    """Member (0,1) K_G block matches the classic local-coordinate transverse pattern."""
    # regression: horizontal members must be untouched by the rotation to
    # global coordinates. Member (0,1) carrying tension N0 only: its K_G
    # block on dofs (0,1,2,3) is (N0/L) * [[0,0,0,0],[0,1,0,-1],[0,0,0,0],
    # [0,-1,0,1]] — the classic local-coordinate transverse pattern.
    from helpers.fem import assemble_KG

    N0 = 500.0
    target = torch.zeros(11, dtype=torch.float64)
    target[0] = N0
    KG = assemble_KG(_force_state(target))
    coef = N0 / geometry.element_lengths()[0].item()
    expected = coef * torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    idx = torch.tensor([0, 1, 2, 3])
    assert torch.allclose(KG[idx][:, idx], expected, rtol=0.0, atol=1e-9 * coef)


def test_KG_transverse_energy_is_rotation_invariant():
    """K_G transverse energy of a differential displacement is N_e/L_e for ANY orientation."""
    # the member K_G acts in the member-transverse direction t = (-s, c):
    # a differential transverse displacement of node j (u_i = 0, u_j = t)
    # must see exactly u' K_G u = N_e/L_e for ANY member orientation.
    # A global-y-only pattern would give c_e^2 * N_e/L_e — only 0.5*N_e/L_e
    # for the 45-degree diagonals.
    from helpers.fem import assemble_KG, axial_forces

    c, s = geometry.element_directions()
    L = geometry.element_lengths().to(torch.float64)
    t = torch.stack([-s, c], dim=-1).to(torch.float64)  # (11, 2)
    N0 = 500.0
    for e in range(11):
        target = torch.zeros(11, dtype=torch.float64)
        target[e] = N0
        u_star = _force_state(target)
        assert torch.allclose(axial_forces(u_star), target, rtol=1e-8, atol=1e-8)
        KG = assemble_KG(u_star)  # == (N0/L_e) * S_e, others ~ 0
        u = torch.zeros(14, dtype=torch.float64)
        j = geometry.ELEMENTS[e][1].item()
        u[2 * j : 2 * j + 2] = t[e]
        u_energy = u @ KG @ u
        # tolerance is float32 geometry quantization: element_directions()
        # is float32, so t_e is off-unit by ~1e-7; the 0.5x bug this test
        # guards against errs at the 5e-1 relative level.
        assert abs(u_energy.item() - N0 / L[e].item()) < 1e-6 * N0 / L[e].item()


def test_KG_rigid_translation_zero_energy():
    """Geometric stiffness sees exactly zero energy for a rigid translation."""
    # geometric stiffness must not resist rigid translation: each member
    # block [[G_perp, -G_perp], [-G_perp, G_perp]] annihilates u_i = u_j,
    # so a uniform nodal translation sees exactly zero K_G energy even
    # under nonzero prestress (manufactured uniform tension).
    from helpers.fem import assemble_KG

    target = torch.full((11,), 500.0, dtype=torch.float64)
    KG = assemble_KG(_force_state(target))
    for vx, vy in [(1.0, 0.0), (0.0, 1.0), (0.6, 0.8)]:
        u = torch.zeros(14, dtype=torch.float64)
        u[0::2] = vx
        u[1::2] = vy
        assert abs((u @ KG @ u).item()) < 1e-9

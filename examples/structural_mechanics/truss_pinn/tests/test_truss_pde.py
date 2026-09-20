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

# tests/test_truss_pde.py
"""Symbolic equilibrium residual tests (spec self-test 6 + guards).

Physics contract: the physicsnemo.sym-compiled residual must reproduce the
assembled reduced stiffness rows exactly. This holds because the residual is
built as dU/du_d with u substituted to 0 at the support DOFs BEFORE
differentiation -- zero-substitution removes the reaction-force terms, so
each equation is exactly one row of K_red (reaction-free by construction).
"""

import torch

from helpers import geometry
from helpers.fem import assemble_K, reduce_matrix
from helpers.truss_pde import TrussEquilibrium, make_residual_fn

_K_RED = reduce_matrix(assemble_K())


def _expected(u_red):
    """K_red @ u for any batch shape, promoted to the matrix dtype."""
    u = u_red.to(_K_RED.dtype)
    return u @ _K_RED.T if u.dim() >= 2 else _K_RED @ u


def test_symbolic_residual_equals_assembled_K_times_u():
    # self-test 6: elementwise agreement between sympy-compiled residual
    # and K_red @ u on a random batch (float32 input vs float64 reference).
    residual = make_residual_fn()
    torch.manual_seed(0)
    u = torch.randn(64, 11)
    r = residual(u)
    expected = _expected(u)
    assert r.shape == (64, 11)
    assert r.dtype == u.dtype
    assert torch.allclose(
        r.to(expected.dtype), expected, rtol=1e-5, atol=1e-4 * expected.abs().max()
    )


def test_residual_gradients_flow():
    # the compiled residual must be differentiable w.r.t. displacements:
    # autograd flows through the SympyToTorch ops back to the input.
    residual = make_residual_fn()
    u = torch.randn(1, 11, requires_grad=True)
    r = residual(u)
    r.sum().backward()
    assert u.grad is not None
    assert torch.isfinite(u.grad).all()
    assert (u.grad != 0).any()
    # analytic anchor (float64 avoids float32 cancellation in the sums):
    # r.sum() = ones^T K_red u, so the gradient is exactly the K_red column sums.
    u64 = torch.randn(1, 11, dtype=torch.float64, requires_grad=True)
    residual(u64).sum().backward()
    expected = _K_RED.sum(dim=0)
    assert torch.allclose(u64.grad.squeeze(0), expected, rtol=1e-8, atol=1e-6)


def test_residual_support_dofs_are_reaction_free():
    # guard against accidental support-term leakage: K_red @ 0 = 0 EXACTLY,
    # so the compiled residual of the zero state must be exactly zero.
    # If support symbols survived differentiation, reaction terms (constant
    # nonzero contributions) would appear here.
    residual = make_residual_fn()
    r = residual(torch.zeros(11))
    assert r.shape == (11,)
    assert torch.equal(r, torch.zeros(11))
    # and every equation is free of support symbols by construction
    pde = TrussEquilibrium()
    support_syms = {pde.u_symbols[d] for d in geometry.SUPPORT_DOFS}
    for name, expr in pde.equations.items():
        assert not (support_syms & expr.free_symbols), name


def test_residual_batch_broadcast_shapes():
    # broadcasting: any leading batch shape over the 11 free DOFs works and
    # stays consistent with K_red @ u.
    residual = make_residual_fn()
    torch.manual_seed(1)
    for shape in [(3, 11), (2, 3, 11), (11,)]:
        u = torch.randn(shape)
        r = residual(u)
        assert r.shape == shape
        expected = _expected(u)
        assert torch.allclose(
            r.to(expected.dtype),
            expected,
            rtol=1e-5,
            atol=1e-4 * expected.abs().max(),
        ), shape


def test_truss_equilibrium_pde_interface():
    # physicsnemo.sym PDE interface: dim set, 11 equations keyed by free DOF.
    pde = TrussEquilibrium()
    assert pde.dim == 1
    assert set(pde.equations) == {f"equilibrium_{d}" for d in geometry.FREE_DOFS}
    # each equation is linear in the free displacement symbols
    for d in geometry.FREE_DOFS:
        expr = pde.equations[f"equilibrium_{d}"]
        sym = pde.u_symbols[d]
        assert expr.diff(sym, 2) == 0

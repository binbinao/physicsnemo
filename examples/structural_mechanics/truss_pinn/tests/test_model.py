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
# Unless required by applicable law or agreed to in writing, distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# tests/test_model.py
"""PINN model + composite loss tests (spec sections 4.1 model and 4.2 loss).

Contracts under test:
- TrussPINN wraps physicsnemo FullyConnected(8 -> 44) and slices the output
  into u_hat (..., 11), log_omega_hat (..., 3), phi_hat (..., 11, 3).
- encode_params one-hot-encodes the loadable-node POSITION (n_idx holds
  node indices, mapped to their slot in geometry.LOADABLE_NODES) and
  appends (cos theta, sin theta, P / P_MAX) -> (..., 8) float32.
- pinn_loss returns (total, {'eq', 'sup', 'mode', 'eig_reg'}) where every
  component is a finite differentiable scalar and the total is the exact
  weighted sum total = w_eq*eq + w_sup*sup + w_mode*mode + eig_reg.
- Gradients reach the parameters through BOTH the equilibrium residual
  path and the eigen-residual regularizer path.
- The mode term phase-fixes the predicted shapes to the anchor sign
  convention: a perfectly predicted but globally flipped mode carries ~zero
  mode loss (a missing phase fix would instead score it O(1)).
"""
import torch

from helpers import geometry
from helpers.sampling import make_anchors, sample_batch
from helpers.truss_pde import make_residual_fn

from helpers.model import TrussPINN, encode_params, pinn_loss

# Anchors and the compiled residual are expensive (~1s compile + 120 FEM
# labels); every test uses the same module-level instances.
_ANCHORS = make_anchors()
_RESIDUAL_FN = make_residual_fn()


def _seeded_model(seed=0):
    torch.manual_seed(seed)
    return TrussPINN()


def _seeded_batch(n_cases=8, seed=1234):
    g = torch.Generator().manual_seed(seed)
    return sample_batch(n_cases, generator=g)


def _grads(model):
    return [p.grad for p in model.parameters() if p.grad is not None]


def test_shapes_and_normalization():
    model = _seeded_model()
    n = torch.tensor([1, 4])
    theta = torch.tensor([0.3, 1.2])
    P = torch.tensor([500.0, -800.0])
    x = encode_params(n, theta, P)
    assert x.shape == (2, 8)
    assert x.dtype == torch.float32
    # node 1 sits at position 0 of LOADABLE_NODES, node 4 at position 2.
    assert torch.allclose(x[0, :5], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(x[1, :5], torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(x[:, 5], torch.cos(theta))
    assert torch.allclose(x[:, 6], torch.sin(theta))
    assert torch.allclose(x[:, 7], P / geometry.P_MAX)

    u_hat, log_omega_hat, phi_hat = model(x)
    assert u_hat.shape == (2, 11)
    assert log_omega_hat.shape == (2, 3)
    assert phi_hat.shape == (2, 11, 3)
    assert u_hat.dtype == torch.float32


def test_loss_components_finite_and_weighted():
    model = _seeded_model()
    batch = _seeded_batch()
    total, comp = pinn_loss(model, batch, _RESIDUAL_FN, _ANCHORS,
                            weights=(1.0, 100.0, 10.0))
    for key in ("eq", "sup", "mode", "eig_reg"):
        assert isinstance(comp[key], torch.Tensor)
        assert comp[key].ndim == 0
        assert torch.isfinite(comp[key]).all(), key
    assert torch.isfinite(total).all()
    # exact float arithmetic: the total is built from these same tensors.
    rebuilt = (comp["eq"] + 100.0 * comp["sup"]
               + 10.0 * comp["mode"] + comp["eig_reg"])
    assert total == rebuilt


def test_loss_gradients_flow():
    model = _seeded_model()
    batch = _seeded_batch()
    total, comp = pinn_loss(model, batch, _RESIDUAL_FN, _ANCHORS,
                            weights=(1.0, 100.0, 10.0))

    model.zero_grad(set_to_none=True)
    total.backward(retain_graph=True)
    grads = _grads(model)
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.norm() > 0 for g in grads), "all parameter gradients vanished"

    # each physics path must deliver gradient on its own: a broken residual
    # graph would zero 'eq', a detached eig_reg would zero the regularizer.
    for key in ("eq", "eig_reg"):
        model.zero_grad(set_to_none=True)
        comp[key].backward(retain_graph=True)
        path_grads = _grads(model)
        assert path_grads, f"{key} path is dead (no gradients)"
        assert any(g.norm() > 0 for g in path_grads), f"{key} path is dead"


def test_phase_fix_matches_anchor_sign():
    # Oracle model returning the exact FEM anchor labels with every mode
    # globally flipped.  The batch is the anchor parameters themselves, so
    # both pinn_loss arms query identical inputs and the oracle answers
    # with the correct labels for each.  The flip must be absorbed by the
    # phase fix: mode ~ 0, while a missing phase fix would score
    # (phi - (-phi))^2 = 4 phi^2 = O(1) (modes are M-orthonormal).
    class FlippedOracle(torch.nn.Module):
        def forward(self, x):
            n = x.shape[0]
            return (_ANCHORS["u_red"][:n],
                    _ANCHORS["omega"][:n].log(),
                    -_ANCHORS["phi"][:n])

    oracle = FlippedOracle()
    total, comp = pinn_loss(oracle, _ANCHORS["params"], _RESIDUAL_FN,
                            _ANCHORS, weights=(1.0, 100.0, 10.0))
    assert comp["sup"].item() == 0.0                      # exact labels
    assert comp["mode"].item() < 1e-10                    # flip recovered
    assert comp["eq"].item() < 1e-10                      # FEM-consistent u
    assert comp["eig_reg"].item() < 1e-10                 # sign-invariant

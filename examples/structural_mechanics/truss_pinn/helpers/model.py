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

"""PINN surrogate for the prestressed Pratt truss: network head + composite loss.

Design decisions (locked at the plan review):

- The network is a plain physicsnemo ``FullyConnected`` (8 -> 47) in float32;
  the 47 outputs are sliced into u_hat (11 free DOFs), log omega_hat (3
  modes) and phi_hat (3 modes x 11 free DOFs).  omega_hat = exp(log_omega)
  is positive by construction, matching the strictly positive eigenvalues
  of (K + K_G) phi = omega^2 M phi inside the frozen load envelope.
- All loss math runs in float64 (the FEM discipline): network outputs are
  cast where they meet the f64 matrices; the cast keeps autograd intact.
- ``eig_reg`` uses the geometric stiffness from FEM statics at the batch
  load, DETACHED: the regularizer teaches the network the eigen relation
  against a correct tangent stiffness, so errors in u_hat cannot
  contaminate the modal loss through a wrong K_G (the two tasks would
  fight).  The anchor modal supervision provides the mode ordering; the
  regularizer provides physics smoothness across load space.
- The mode term phase-fixes each predicted mode to the anchor sign
  convention before the MSE: eigenvectors are sign-ambiguous, so a
  globally flipped prediction must not be penalized.  The fix multiplies
  phi_hat by sign(<phi_hat, phi_anchor>), falling back to the
  largest-entry sign product when the overlap is ~0; the +-1 scale is
  piecewise constant, so gradients flow straight through s * phi_hat.
- Weights are fixed (no adaptation): lambda_eq = 1, lambda_sup = 100,
  lambda_mode = 10, eig_reg weight 1.
"""

import torch
import torch.nn.functional as F

from physicsnemo.models.mlp import FullyConnected

from helpers import geometry
from helpers.fem import (
    assemble_K,
    assemble_KG,
    assemble_M,
    nodal_load,
    reduce_matrix,
    solve_static,
)

_DTYPE = torch.float64

_N_DOF = 2 * geometry.NODES_XY.shape[0]  # 14
_N_FREE = len(geometry.FREE_DOFS)  # 11
_N_MODES = 3
_FREE = torch.tensor(geometry.FREE_DOFS)
_LOADABLE = tuple(geometry.LOADABLE_NODES)

# node index -> slot in LOADABLE_NODES (for the one-hot feature).
_SLOT_LOOKUP = torch.full((geometry.NODES_XY.shape[0],), -1, dtype=torch.int64)
_SLOT_LOOKUP[torch.tensor(_LOADABLE)] = torch.arange(len(_LOADABLE))

# Below-overlap threshold for the phase-fix fallback (|<phi_hat, phi>|).
_PHASE_EPS = 1e-12

_K_RED = None
_M_RED = None


def _reduced_matrices():
    """Cached reduced stiffness and mass (11, 11) float64."""
    global _K_RED, _M_RED
    if _K_RED is None:
        _K_RED = reduce_matrix(assemble_K())
        _M_RED = reduce_matrix(assemble_M())
    return _K_RED, _M_RED


def encode_params(n_idx, theta, P):
    """Map load parameters to the network input (..., 8) float32.

    n_idx holds NODE indices; each is mapped to its slot in
    geometry.LOADABLE_NODES and one-hot encoded (5 dims), then
    (cos theta, sin theta, P / P_MAX) is appended.  cos/sin are evaluated
    in the incoming dtype (f64 labels) and cast to f32 afterwards.
    """
    slots = _SLOT_LOOKUP[n_idx.long()]
    one_hot = F.one_hot(slots, num_classes=len(_LOADABLE)).to(torch.float32)
    feats = torch.stack([theta.cos(), theta.sin(), P / geometry.P_MAX], dim=-1).to(
        torch.float32
    )
    return torch.cat([one_hot, feats], dim=-1)


class TrussPINN(torch.nn.Module):
    """Truss surrogate: load parameters -> (u_hat, log_omega_hat, phi_hat)."""

    def __init__(self):
        super().__init__()
        self.net = FullyConnected(
            in_features=8,
            out_features=_N_FREE + _N_MODES + _N_FREE * _N_MODES,  # 47
            layer_size=128,
            num_layers=4,
        )

    def forward(self, x):
        """x (..., 8) -> (u_hat (..., 11), log_omega_hat (..., 3),
        phi_hat (..., 11, 3)); phi_hat columns are the mode shapes."""
        out = self.net(x)
        u_hat = out[..., :_N_FREE]
        log_omega_hat = out[..., _N_FREE : _N_FREE + _N_MODES]
        phi_hat = out[..., _N_FREE + _N_MODES :].reshape(
            *out.shape[:-1], _N_FREE, _N_MODES
        )
        return u_hat, log_omega_hat, phi_hat


def _batch_f_red(n_idx, theta, P):
    """Reduced load vectors (N, 11) f64, built through fem.nodal_load."""
    return torch.stack(
        [
            nodal_load(int(n), float(t), float(p))[_FREE]
            for n, t, p in zip(n_idx.tolist(), theta.tolist(), P.tolist())
        ]
    )


def _mode_component(log_omega_hat, phi_hat, phi_anchor, omega_anchor):
    """Anchor modal supervision: log-omega MSE + phase-fixed shape MSE.

    log_omega_hat, phi_hat: network predictions for the anchor cases,
    phi_hat (N, 11, 3) with modes as columns; phi_anchor, omega_anchor:
    FEM labels.  Each predicted mode is first phase-fixed to the anchor
    sign convention: s = sign(<phi_hat, phi_anchor>) per (case, mode),
    falling back to the product of the largest-entry signs when the
    overlap is below _PHASE_EPS.  s is +-1 (piecewise constant), so the
    gradient flows through s * phi_hat untouched.
    """
    omega_term = (log_omega_hat.to(_DTYPE) - omega_anchor.log()).pow(2).mean()

    phi_hat64 = phi_hat.to(_DTYPE)
    overlap = (phi_hat64 * phi_anchor).sum(dim=-2)  # (N, 3)
    lead_hat = phi_hat64.abs().argmax(dim=-2)
    lead_anchor = phi_anchor.abs().argmax(dim=-2)
    fallback = (
        phi_hat64.gather(-2, lead_hat.unsqueeze(-2)).sign()
        * phi_anchor.gather(-2, lead_anchor.unsqueeze(-2)).sign()
    ).squeeze(-2)
    s = torch.where(overlap.abs() < _PHASE_EPS, fallback, overlap.sign())
    phi_fixed = s.unsqueeze(-2) * phi_hat64
    phi_term = (phi_fixed - phi_anchor).pow(2).mean()
    return omega_term + phi_term


def pinn_loss(model, batch, residual_fn, anchors, weights=(1.0, 100.0, 10.0)):
    """Composite PINN loss -> (total, components dict).

    Parameters
    ----------
    model : TrussPINN
    batch : dict
        Collocation draws from helpers.sampling.sample_batch with keys
        'n_idx', 'theta', 'P'.
    residual_fn : callable
        helpers.truss_pde.make_residual_fn() output; maps free-DOF
        displacements (..., 11) to the equilibrium residual K_red @ u.
    anchors : dict
        helpers.sampling.make_anchors() output ('params', 'u_red',
        'omega', 'phi').
    weights : tuple of float
        (w_eq, w_sup, w_mode); the eig_reg weight is fixed at 1.

    Returns
    -------
    (total, {'eq', 'sup', 'mode', 'eig_reg'})
        Each component is a differentiable scalar tensor (float64 math);
        total = w_eq*eq + w_sup*sup + w_mode*mode + eig_reg.
    """
    w_eq, w_sup, w_mode = weights
    K_red, M_red = _reduced_matrices()

    # --- batch (collocation) arm: equilibrium residual + eigen regularizer.
    x_b = encode_params(batch["n_idx"], batch["theta"], batch["P"])
    u_hat_b, log_omega_hat_b, phi_hat_b = model(x_b)
    f_red = _batch_f_red(batch["n_idx"], batch["theta"], batch["P"])
    residual = residual_fn(u_hat_b.to(_DTYPE))
    eq = (residual - f_red).pow(2).mean()

    # Tangent stiffness from FEM statics at the batch loads, FROZEN: no
    # gradient flows into K_G (detached by construction + no_grad).
    with torch.no_grad():
        u14 = torch.zeros(f_red.shape[0], _N_DOF, dtype=_DTYPE)
        u14[:, _FREE] = solve_static(K_red, f_red)
        K_t = K_red + reduce_matrix(assemble_KG(u14))  # (N, 11, 11)
    omega_hat = log_omega_hat_b.to(_DTYPE).exp()  # positive
    phi_hat_b64 = phi_hat_b.to(_DTYPE)  # raw signs
    lhs = torch.matmul(K_t, phi_hat_b64)
    rhs = omega_hat.pow(2).unsqueeze(-2) * torch.matmul(M_red, phi_hat_b64)
    eig_reg = (lhs - rhs).pow(2).mean()

    # --- anchor (supervision) arm: displacements + modal labels.
    params = anchors["params"]
    x_a = encode_params(params["n_idx"], params["theta"], params["P"])
    u_hat_a, log_omega_hat_a, phi_hat_a = model(x_a)
    sup = (u_hat_a.to(_DTYPE) - anchors["u_red"]).pow(2).mean()
    mode = _mode_component(log_omega_hat_a, phi_hat_a, anchors["phi"], anchors["omega"])

    total = w_eq * eq + w_sup * sup + w_mode * mode + eig_reg
    return total, {"eq": eq, "sup": sup, "mode": mode, "eig_reg": eig_reg}

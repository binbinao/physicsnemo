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

"""physicsnemo.sym symbolic equilibrium residual for the Pratt truss.

Total strain energy of the pin-jointed truss,

    U = sum_e (E*A / (2*L_e)) * (c_e*(u_jx - u_ix) + s_e*(u_jy - u_iy))**2,

is built symbolically over the 14 nodal displacement symbols u_0..u_13 and
differentiated with respect to each free DOF to give the equilibrium
equations dU/du_d = f_d.

The residual equals K_red @ u exactly (up to floating-point rounding),
because the support displacements u_0 = u_1 = u_7 = 0 are substituted into U
BEFORE differentiation: zero-substitution removes the reaction-force terms,
so each dU/du_d is exactly one row of the reduced stiffness matrix.
Keeping the support symbols free instead would leak reaction forces into
the residual.
"""

import sympy as sp
import torch

from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.utils.sympy.torch_printer import SympyToTorch

from helpers import geometry

_N_DOF = 2 * geometry.NODES_XY.shape[0]  # 14


class TrussEquilibrium(PDE):
    """Equilibrium equations dU/du_d for each free DOF of the truss.

    Attributes
    ----------
    dim : int
        Set to 1 for the PDE base class; unused since the residual is pure
        algebra (no spatial derivatives enter the energy).
    equations : dict[str, sp.Expr]
        ``{"equilibrium_<d>": dU/du_d}`` for each free DOF d, reaction-free
        (support symbols already substituted to zero).
    u_symbols : dict[int, sp.Symbol]
        Displacement symbols keyed by global DOF index.
    """

    def __init__(self):
        self.dim = 1

        L = geometry.element_lengths()  # (11,) float32
        c, s = geometry.element_directions()  # (11,) float32 each
        self.u_symbols = {d: sp.Symbol(f"u_{d}", real=True) for d in range(_N_DOF)}
        u = self.u_symbols

        energy = 0
        for e, (i, j) in enumerate(geometry.ELEMENTS.tolist()):
            elong = float(c[e]) * (u[2 * j] - u[2 * i]) + float(s[e]) * (
                u[2 * j + 1] - u[2 * i + 1]
            )
            energy += geometry.E * geometry.AREA / (2.0 * float(L[e])) * elong**2

        # Substitute the support DOFs to zero BEFORE differentiating: the
        # resulting equations are exactly the reduced stiffness rows.
        energy = energy.subs({u[d]: 0 for d in geometry.SUPPORT_DOFS})
        self.equations = {
            f"equilibrium_{d}": sp.diff(energy, u[d]) for d in geometry.FREE_DOFS
        }


def make_residual_fn():
    """Compile the symbolic equilibrium equations into a torch callable.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        ``residual(u_red)`` mapping a free-DOF displacement vector of any
        batch shape ``(..., 11)`` to the equilibrium residual
        ``K_red @ u_red`` broadcast over the batch.
    """
    pde = TrussEquilibrium()
    modules = {name: SympyToTorch(expr, name) for name, expr in pde.equations.items()}
    free_dofs = geometry.FREE_DOFS
    support_dofs = geometry.SUPPORT_DOFS

    def residual(u_red: torch.Tensor) -> torch.Tensor:
        if u_red.shape[-1] != len(free_dofs):
            raise ValueError(
                f"expected trailing dimension {len(free_dofs)} (free DOFs), "
                f"got shape {tuple(u_red.shape)}"
            )
        # Every equation reads only free symbols after the zero substitution,
        # but supply all 14 keys so each module finds its sorted key list.
        var = {f"u_{d}": u_red[..., k] for k, d in enumerate(free_dofs)}
        zero = torch.zeros_like(u_red[..., 0])
        for d in support_dofs:
            var[f"u_{d}"] = zero
        out = [modules[f"equilibrium_{d}"](var)[f"equilibrium_{d}"] for d in free_dofs]
        return torch.stack(out, dim=-1)

    return residual

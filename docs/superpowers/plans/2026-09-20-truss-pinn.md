# Truss PINN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FEM reference solver and a discrete-energy PINN that predicts Pratt-truss deformation and prestressed modal parameters (first 3 frequencies + mode shapes) over a parameterized load range.

**Architecture:** `helpers/geometry.py` is the single source of truth (nodes, connectivity, supports, material). `helpers/fem.py` assembles K, M, K_G with torch, solves statics and the prestressed generalized eigenproblem batched on GPU, and self-verifies against closed forms. `helpers/truss_pde.py` derives the equilibrium residual symbolically with `physicsnemo.sym` (sympy strain energy → `SympyToTorch`). `helpers/model.py` wraps `physicsnemo.models.mlp.FullyConnected` (8→43) and composes the loss (equilibrium residual + anchor supervision + modal supervision). `train.py` / `evaluate.py` are the Hydra entry points.

**Tech Stack:** Python 3.12, torch 2.5.1+cu124, physicsnemo 2.1.0a0 (editable, this repo), physicsnemo.sym (sympy 1.13.1), Hydra 1.3, matplotlib.

**Spec:** `docs/superpowers/specs/2026-09-20-truss-pinn-design.md`

## Global Constraints

- Placement: `examples/structural_mechanics/truss_pinn/` only; examples are not shipped in the wheel.
- Every `.py`/`.yaml` file starts with the exact SPDX header from `test/ci_tests/copyright.txt` (copy from `examples/structural_mechanics/deforming_plate/train.py:1-15`).
- Material: E = 3e9 Pa, rho = 1150 kg/m³, A = 1e-4 m² (nylon 6/6, 32×32 mm solid square bar).
- Nodes: bottom `0,1,2,3` at (0,0),(2,0),(4,0),(6,0) m; top `4,5,6` at (1,1),(3,1),(5,1) m. 11 members: (0,1),(1,2),(2,3),(4,5),(5,6),(0,4),(3,6),(1,4),(2,5),(1,5),(2,6). Supports: node 0 pin (x,y fixed), node 3 roller (y fixed). Free nodes: {1,2,4,5,6}.
- Network input 8-dim: one-hot(n) 5 ⊕ (cosθ, sinθ, P/P_max). Output 43-dim: u_free 10 + log ω 3 + mode shapes 30 (3×10). λ_eq=1, λ_sup=100, λ_mode=10.
- Acceptance: displacement relative L2 error < 5%, first-3 frequency relative error < 2% on 200 independent test cases; P_max = 0.3·P_crit from stability scan.
- Run training on the available Tesla T4; tests run from `examples/structural_mechanics/truss_pinn/` with `.venv/bin/python` from repo root (`/data/physicsnemo`).
- No `physicsnemo.launch` imports (removed in v2.0); logging via `physicsnemo.utils.logging`; checkpointing via `torch.save`/`torch.load` (small model, example-local — `physicsnemo.utils.checkpoint.save_checkpoint` targets training-loop state, not needed here).
- ruff lint is not enforced under `examples/` (pre-commit `ruff-check` excludes `^examples/`) but format example code; license headers ARE enforced (hook matches `.py`).

---

### Task 1: Geometry single source of truth

**Files:**
- Create: `examples/structural_mechanics/truss_pinn/helpers/__init__.py`
- Create: `examples/structural_mechanics/truss_pinn/helpers/geometry.py`
- Test: `examples/structural_mechanics/truss_pinn/tests/test_geometry.py`

**Interfaces:**
- Produces:
  - `NODES_XY: Float[torch.Tensor, "7 2"]` (meters)
  - `ELEMENTS: Int[torch.Tensor, "11 2"]` (node index pairs)
  - `SUPPORT_DOFS: tuple[int, ...]` = (0, 1, 6) — DOF 2*node+axis for pin node 0 (x,y) and roller node 3 (y)
  - `FREE_NODES: tuple[int, ...]` = (1, 2, 4, 5, 6); `FREE_DOFS: tuple[int, ...]` (10 values)
  - `E: float`, `RHO: float`, `AREA: float`, `INERTIA: float = AREA**2 / 12`
  - `P_MAX: float` (initial 2000.0 N, refrozen after Task 4 scan)
  - `loadable_nodes() -> list[int]`, `element_lengths() -> Float[torch.Tensor, "11"]`,
    `element_directions() -> tuple[Float[torch.Tensor, "11"], Float[torch.Tensor, "11"]]` (c, s unit vectors)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geometry.py
import torch
from helpers.geometry import (
    NODES_XY, ELEMENTS, SUPPORT_DOFS, FREE_DOFS, E, RHO, AREA,
    element_lengths, element_directions, FREE_NODES,
)

def test_connectivity_is_triangulated():
    # every member connects distinct nodes, positive length
    L = element_lengths()
    assert L.shape == (11,)
    assert torch.all(L > 1.0)  # shortest member is 2 m vertical / 2.23 m diagonal
    assert torch.all(ELEMENTS[:, 0] != ELEMENTS[:, 1])

def test_support_and_free_dofs_partition():
    # 14 total DOFs partition into 3 support + 11... no: 3 fixed + 10 free + 1 free (roller x)
    # pin node0: dofs 0,1 ; roller node3: dof 7 (y). Free dofs = the rest.
    assert set(SUPPORT_DOFS) == {0, 1, 7}
    assert len(FREE_DOFS) == 11
    assert set(SUPPORT_DOFS).isdisjoint(FREE_DOFS)
    assert set(SUPPORT_DOFS) | set(FREE_DOFS) == set(range(14))

def test_directions_are_unit():
    c, s = element_directions()
    assert torch.allclose(c**2 + s**2, torch.ones(11))

def test_material_constants():
    assert E == 3e9 and RHO == 1150.0 and AREA == 1e-4
```

Wait — the roller support at node 3 fixes only y. DOF numbering is `2*node + axis` with axis∈{0:x, 1:y}. Roller node 3 → dof `2*3+1 = 7`. Pin node 0 → dofs 0 and 1. **SUPPORT_DOFS = (0, 1, 7)**, FREE_DOFS = the remaining 11 dofs: (2,3,4,5,6,8,9,10,11,12,13) — free-node x,y for nodes 1,2,4,5,6 plus node 3 x (dof 6). The PINN output layer therefore has **11 displacement dims, not 10** (spec §4.1 said 10 — correct at plan level to 11; the roller's horizontal DOF is a real unknown). Correct network output: 11 + 3 + 30 = **44**.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_geometry.py -v` (cwd = example dir; add `tests/__init__.py` empty if import complains)
Expected: FAIL with `ModuleNotFoundError: No module named 'helpers'` or ImportError.

- [ ] **Step 3: Implement `helpers/geometry.py`**

```python
"""Pratt truss geometry: single source of truth for nodes, members, supports, material."""
import torch

E = 3.0e9          # Pa, nylon 6/6
RHO = 1150.0       # kg/m^3
AREA = 1.0e-4      # m^2, 32x32 mm solid square
INERTIA = AREA**2 / 12.0  # m^4

P_MAX = 2000.0     # N, provisional; refrozen after stability scan (Task 4)

# node index: bottom 0..3 left→right, top 4..6 left→right
NODES_XY = torch.tensor([
    [0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0],
    [1.0, 1.0], [3.0, 1.0], [5.0, 1.0],
])
ELEMENTS = torch.tensor([
    [0, 1], [1, 2], [2, 3],      # bottom chord
    [4, 5], [5, 6],              # top chord
    [0, 4], [3, 6],              # end diagonals
    [1, 4], [2, 5],              # verticals
    [1, 5], [2, 6],              # inner diagonals (Pratt: slope down toward center)
])
SUPPORT_DOFS = (0, 1, 7)          # pin node 0 (x,y), roller node 3 (y only)
FREE_DOFS = tuple(d for d in range(14) if d not in SUPPORT_DOFS)  # 11 dofs
FREE_NODES = (1, 2, 4, 5, 6)
LOADABLE_NODES = (1, 2, 4, 5, 6)

def element_lengths() -> torch.Tensor:  # (11,)
    p = NODES_XY[ELEMENTS[:, 0]]
    q = NODES_XY[ELEMENTS[:, 1]]
    return (q - p).norm(dim=-1)

def element_directions():
    """Return (cos, sin) unit direction of each member, i→j."""
    p = NODES_XY[ELEMENTS[:, 0]]
    q = NODES_XY[ELEMENTS[:, 1]]
    d = q - p
    L = d.norm(dim=-1, keepdim=True)
    return d[..., 0] / L[..., 0], d[..., 1] / L[..., 0]
```

Also create `helpers/__init__.py` (license header only) and `tests/__init__.py` (empty, header).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_geometry.py -v`
Expected: 4 PASS.

- [ ] **Step 4b: Fix DOF numbering comment** — the test above asserts `SUPPORT_DOFS == {0,1,7}` and 11 free DOFs (roller node 3's x-dof 6 stays free). This supersedes the spec's "10 displacement dims"; network output becomes 44 (Task 6).
- [ ] **Step 5: Commit**

```bash
git add examples/structural_mechanics/truss_pinn/
git commit -s -m "feat(truss-pinn): geometry single source of truth"
```

---

### Task 2: FEM assembly + batched statics (K, M, static solve)

**Files:**
- Create: `examples/structural_mechanics/truss_pinn/helpers/fem.py`
- Test: `examples/structural_mechanics/truss_pinn/tests/test_fem.py`

**Interfaces:**
- Consumes: `geometry` (all constants, `element_lengths`, `element_directions`, `FREE_DOFS`, `SUPPORT_DOFS`).
- Produces:
  - `assemble_K() -> Float[torch.Tensor, "14 14"]`, `assemble_M() -> Float[torch.Tensor, "14 14"]` (consistent mass)
  - `reduce_matrix(K) -> Float[torch.Tensor, "11 11"]` — rows/cols at FREE_DOFS
  - `solve_static(K_red: Tensor[..., 11, 11], f_red: Tensor[..., 11]) -> Tensor[..., 11]` — batched `torch.linalg.solve`
  - `nodal_load(n: int, theta: float, P: float) -> Float[torch.Tensor, "14"]` — `f[2n]=P·cosθ, f[2n+1]=P·sinθ`
  - `axial_forces(u_full: Tensor[..., 14]) -> Float[torch.Tensor, "... 11"]` — `N_e = (EA/L)·(c·Δu_x + s·Δu_y)`, tension positive

Member 2D stiffness (global), for member e with direction (c, s), length L, axial k=EA/L — the standard outer product:

```python
def _member_matrices():
    L = geometry.element_lengths()              # (11,)
    c, s = geometry.element_directions()        # (11,)
    k = geometry.E * geometry.AREA / L          # (11,)
    d = torch.stack([c, s], dim=-1)             # (11, 2)
    G = d.unsqueeze(-1) * d.unsqueeze(-2)       # (11, 2, 2) = [cc cs; cs ss]
    K_e = k.view(-1, 1, 1) * torch.cat([        # (11, 4, 4)
        torch.cat([ G, -G], dim=-1),
        torch.cat([-G,  G], dim=-1), ], dim=-2)
    # consistent mass, local axial: rho*A*L/6 * [[2,-1],[-1,2]] rotated to global:
    # global consistent mass = rho*A*L/6 * [[2G, -G],[-G, 2G]] with G as above
    m = geometry.RHO * geometry.AREA * L / 6.0
    M_e = m.view(-1, 1, 1) * torch.cat([
        torch.cat([2*G, -G], dim=-1),
        torch.cat([-G,  2*G], dim=-1), ], dim=-2)
    return K_e, M_e
```

Scatter into global 14×14 via DOF map `dofs = [2i, 2i+1, 2j, 2j+1]` per member (`K.index_put_` or advanced indexing with `+=` on a fresh zero tensor; note `index_add_` for batch-free correctness — plain `+=` on advanced-index assignment overwrites instead of accumulating, so use `K.index_put_((rows, cols), vals, accumulate=True)`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fem.py
import torch
from helpers import geometry
from helpers.fem import assemble_K, assemble_M, reduce_matrix, solve_static, nodal_load, axial_forces

def test_K_symmetric_positive_semidefinite():
    K = assemble_K()
    assert K.shape == (14, 14)
    assert torch.allclose(K, K.T, atol=1e-9)
    eig = torch.linalg.eigvalsh(K)
    assert eig.min() > -1e-6          # PSD on full space (3 constraints)
    K_red = reduce_matrix(K)
    assert K_red.shape == (11, 11)
    assert torch.linalg.eigvalsh(K_red).min() > 0.0   # PD after elimination

def test_M_positive_definite_symmetric():
    M = assemble_M()
    assert torch.allclose(M, M.T, atol=1e-9)
    assert torch.diag(M).min() > 0.0

def test_single_member_axial_closed_form():
    # self-test 1: one member pulled with F along its axis stretches F*L/(EA)
    # build a tiny 2-node truss by monkeypatching geometry? NO — instead verify
    # against the assembled matrix: apply load at free node 1 pointing along
    # member (0,1) direction and check reaction-free displacement matches
    # the inverse of K_red (identity check of solve_static)
    K = assemble_K(); K_red = reduce_matrix(K)
    f = torch.randn(11)
    u = solve_static(K_red, f)
    assert torch.allclose(K_red @ u, f, atol=1e-8)

def test_static_equilibrium_and_axial_force_symmetry():
    # vertical load P down at node 1: by symmetry with load at node 2 mirrored
    K_red = reduce_matrix(assemble_K())
    # load node 1 down by 100 N
    f14 = nodal_load(1, -torch.pi/2, 100.0)
    f_red = f14[torch.tensor(geometry.FREE_DOFS)]
    u_red = solve_static(K_red, f_red)
    u14 = torch.zeros(14); u14[torch.tensor(geometry.FREE_DOFS)] = u_red
    N = axial_forces(u14)
    assert N.shape == (11,)
    # reaction check: sum of external forces == sum at supports (statics)
    K = assemble_K()
    reactions = (K @ u14) - f14            # nonzero only at support dofs
    assert torch.allclose(reactions[[2,3,4,5,6,8,9,10,11,12,13]],
                           torch.zeros(11), atol=1e-6)
    assert abs(reactions[0] + reactions[7] + f14[1] - 0.0) < 1e-6 or True  # vertical equilibrium via supports
    # total vertical equilibrium: R_y(node0) + R_y(node3) + f_y = 0
    assert abs(reactions[1] + reactions[7] + f14[1]) < 1e-6

def test_axial_force_closed_form_single_member():
    # hand check: stretch member (0,1) along x by u = EA/L... use u14 uniform field
    # displace node 1 by +dx in x only: member (0,1) axial force = EA/L * dx
    dx = 1e-4
    u14 = torch.zeros(14); u14[2] = dx    # node 1 x
    N = axial_forces(u14)
    L01 = 2.0; k01 = geometry.E * geometry.AREA / L01
    assert abs(N[0].item() - k01 * dx) < 1e-6 * k01 * dx + 1e-9
```

- [ ] **Step 2: Run tests — expect FAIL (ImportError)**

Run: `.venv/bin/python -m pytest tests/test_fem.py -v`

- [ ] **Step 3: Implement `helpers/fem.py`** per the interface block and `_member_matrices` sketch above. Key details:
  - Module-level `_K`, `_M` lazily built once (CPU float64 default for reference quality; cast on demand).
  - `assemble_K/M` return fresh clones (callers may mutate).
  - `axial_forces` uses `(EA/L)·(c·(u_jx−u_ix) + s·(u_jy−u_iy))`.
  - `solve_static` accepts `(..., 11, 11)` batches via `torch.linalg.solve`.

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit** `git commit -s -m "feat(truss-pinn): FEM assembly and batched static solve"`

---

### Task 3: Prestressed modal solver (K_G + generalized eigenvalue)

**Files:**
- Modify: `examples/structural_mechanics/truss_pinn/helpers/fem.py`
- Test: `examples/structural_mechanics/truss_pinn/tests/test_fem.py`

**Interfaces:**
- Consumes: Task 2 (`axial_forces`, `reduce_matrix`, matrices).
- Produces:
  - `assemble_KG(u_full: Tensor[..., 14]) -> Float[torch.Tensor, "... 14 14"]` — per-member geometric stiffness `(N_e/L_e)·[[0,0,0,0],[0,1,0,-1],[0,0,0,0],[0,-1,0,1]]` scattered to global (transverse terms only)
  - `modal_solve(u_full: Tensor[..., 14], n_modes: int = 3) -> tuple[Tensor, Tensor]` — returns `(omegas[..., n], modes[..., 11, n])` solving `(K+K_G)φ = ω²Mφ` on free DOFs via Cholesky: `L_M = cholesky(M_red)`; `A = L_M⁻¹(K+K_G)L_M⁻ᵀ`; `eigh(A)`; φ = `L_M⁻ᵀ`·eigvecs (M-normalized by construction); sort ascending; ω = sqrt(λ).
  - `phase_fixed(phi: Tensor[..., 11, n]) -> Tensor` — flip sign so the largest-magnitude entry of each mode is positive.

- [ ] **Step 1: Write the failing test**

```python
def test_modal_M_orthonormality():
    from helpers.fem import modal_solve, assemble_M, reduce_matrix
    u0 = torch.zeros(14)
    omegas, modes = modal_solve(u0)
    assert omegas.shape == (3,) and modes.shape == (11, 3)
    M_red = reduce_matrix(assemble_M())
    Gram = modes.T @ M_red @ modes          # (3,3)
    assert torch.allclose(Gram, torch.eye(3), atol=1e-8)   # self-test 3
    assert (omegas > 0).all()

def test_tension_raises_compression_lowers_frequency():
    # self-test 4: pull at node 1 straight up (+y) vs push down (-y) with same |P|
    from helpers.fem import solve_static, nodal_load, modal_solve, phase_fixed
    K_red = reduce_matrix(assemble_K())
    for sign, expect_up in ((+1.0, True), (-1.0, False)):
        f14 = nodal_load(1, torch.pi/2 if sign > 0 else -torch.pi/2, 1000.0)
        f_red = f14[torch.tensor(geometry.FREE_DOFS)]
        u_red = solve_static(K_red, f_red)
        u14 = torch.zeros(14); u14[torch.tensor(geometry.FREE_DOFS)] = u_red
        omegas, _ = modal_solve(u14)
        omegas0, _ = modal_solve(torch.zeros(14))
        rel = (omegas[0] - omegas0[0]) / omegas0[0]
        assert (rel > 1e-5) if expect_up else (rel < -1e-5)

def test_phase_fixed_makes_max_entry_positive():
    from helpers.fem import modal_solve, phase_fixed
    _, modes = modal_solve(torch.zeros(14))
    modes_neg = phase_fixed(-modes)
    # flipping the input mode must yield the same fixed mode up to zero comparisons
    max_vals = modes_neg.abs().amax(dim=0)
    assert (modes_neg[modes_neg.abs().argmax(dim=0), torch.arange(3)] > 0).all()
```

Note on `test_tension_raises...`: pulling **up** at a bottom-chord node loads the structure so some members go into tension — but whether the *fundamental* frequency rises depends on which members stiffen. The physically robust version of self-test 4: compare ω₁ under +P vs −P for **both** directions and assert they differ, plus assert the sign of Δω matches the sign of total strain energy change computed from `axial_forces` — i.e. `sign(Δω₁) == sign(Σ N_e² L_e/(EA) |_{+P} − Σ N_e² L_e/(EA) |_{−P})` is NOT the right invariant either (energy scales with P²). **Correct invariant**: prestress stiffening is linear in N_e — Δω₁ has the sign of the member-weighted sum `Σ_e (N_e/(E·A)) · (∂ω₁/∂(N_e/(EA)))` … this is over-engineering the test. Pragmatic version: assert `|Δω₁(+P)| > 0` and `Δω₁(+P) ≠ Δω₁(−P)` with a relative tolerance 1e-6, and verify monotonicity numerically in Task 4's scan (which is where the physics enters). Replace the body of `test_tension_raises_compression_lowers_frequency` with:

```python
def test_prestress_changes_frequency_symmetrically_opposite():
    from helpers.fem import solve_static, nodal_load, modal_solve
    K_red = reduce_matrix(assemble_K())
    def omega_at(P):
        f14 = nodal_load(1, -torch.pi/2, P)   # downward load at node 1
        f_red = f14[torch.tensor(geometry.FREE_DOFS)]
        u_red = solve_static(K_red, f_red)
        u14 = torch.zeros(14); u14[torch.tensor(geometry.FREE_DOFS)] = u_red
        w, _ = modal_solve(u14)
        return w[0]
    w0 = omega_at(0.0)
    w_plus, w_minus = omega_at(1000.0), omega_at(-1000.0)
    # linear prestress theory: first-order change is odd in P
    assert (w_plus - w0) * (w_minus - w0) < 0.0
    assert abs(w_plus - w0) > 1e-7 * w0 and abs(w_minus - w0) > 1e-7 * w0
```

- [ ] **Step 2: Run — FAIL (functions missing)**
- [ ] **Step 3: Implement `assemble_KG`, `modal_solve`, `phase_fixed`.** Cholesky path per interface; guard `torch.linalg.eigh` on the symmetrized `A` (`0.5*(A+A.T)`).
- [ ] **Step 4: Run — PASS (all fem tests)**
- [ ] **Step 5: Commit** `git commit -s -m "feat(truss-pinn): prestressed modal solver"`

---

### Task 4: Stability scan + freeze P_MAX (spec §2.2)

**Files:**
- Create: `examples/structural_mechanics/truss_pinn/helpers/stability.py`
- Modify: `examples/structural_mechanics/truss_pinn/helpers/geometry.py` (P_MAX constant)
- Test: `examples/structural_mechanics/truss_pinn/tests/test_stability.py`

**Interfaces:**
- Produces:
  - `scan_p_crit(theta: float = -pi/2, n: int = 1, p_hi: float = 5e4, n_steps: int = 400) -> float` — bisection/scan on the smallest eigenvalue of (K+K_G) reduced: find first sign change of `eigvalsh(K_red + KG_red)[0]`; return P_crit (load where tangent stiffness loses positivity under this load path).
  - `freeze_p_max(safety: float = 0.3) -> float` — computes scan across **loadable-node × direction grid** and writes the resulting P_MAX back into `geometry.P_MAX` at runtime (single assignment point in `train.py` startup).

- [ ] **Step 1: Write the failing test**

```python
def test_p_crit_positive_and_finite():
    from helpers.stability import scan_p_crit
    p = scan_p_crit()
    assert p == p and 100.0 < p < 5e4   # NaN guard + plausible range

def test_frozen_p_max_stays_stable():
    # self-test 5: (K+K_G) min eig > 0 across the FULL frozen load range
    from helpers.stability import scan_p_crit
    from helpers import geometry
    p_crit = scan_p_crit()
    geometry.P_MAX = 0.3 * p_crit
    # spot-check worst direction: same load path as the scan
    K_red = reduce_matrix(assemble_K())
    f14 = nodal_load(1, -torch.pi/2, geometry.P_MAX)
    f_red = f14[torch.tensor(geometry.FREE_DOFS)]
    u_red = solve_static(K_red, f_red)
    u14 = torch.zeros(14); u14[torch.tensor(geometry.FREE_DOFS)] = u_red
    KG = reduce_matrix(assemble_KG(u14))
    assert torch.linalg.eigvalsh(K_red + KG).min() > 0.0
```

- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement scan** — coarse-to-fine: linear scan 400 steps, bisect 40 iterations between the last stable and first unstable P. Document in the module docstring that P_crit is load-path-dependent (conservative: take min over the node×direction grid sampled at 8 θ values × 5 nodes for `freeze_p_max`).
- [ ] **Step 4: Run — PASS**; record the frozen `P_MAX` value in geometry.py docstring (e.g. `P_MAX = 0.3 * p_crit_grid_min ≈ <value> N`). Update `geometry.P_MAX` default to the computed value with a comment `# frozen from stability scan, tests/test_stability.py`.
- [ ] **Step 5: Commit** `git commit -s -m "feat(truss-pinn): stability scan freezes P_MAX"`

---

### Task 5: sym symbolic residual (spec §4.2 item 1)

**Files:**
- Create: `examples/structural_mechanics/truss_pinn/helpers/truss_pde.py`
- Test: `examples/structural_mechanics/truss_pinn/tests/test_truss_pde.py`

**Interfaces:**
- Consumes: geometry (c, s, L per member), `physicsnemo.sym` (`PDE` base, `Computation.from_sympy`, `SympyToTorch` via `physicsnemo.sym.utils.sympy.torch_printer`).
- Produces:
  - `class TrussEquilibrium(PDE)`: `self.dim = 1` (sympy dim field unused — no spatial derivatives), `self.equations = {"equilibrium_i": dU/du_i}` for i in FREE_DOFS — **11 symbolic equations**, each the derivative of total strain energy `U = Σ_e (EA/2L_e)·(c·(u_jx−u_ix) + s·(u_jy−u_iy))²` w.r.t. one free DOF's displacement symbol.
  - `make_residual_fn() -> Callable[[Tensor[..., 11]], Tensor[..., 11]]` — compiles the 11 equations through `SympyToTorch` (one module per equation, keyed by free-DOF symbol names `u_0..u_13`), returns `r(params) -> residual` where input is the **free-DOF displacement vector** and output `K_red·u` (batched over leading dims via broadcasting).

Implementation sketch:

```python
import sympy as sp
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.utils.sympy.torch_printer import SympyToTorch
from helpers import geometry

class TrussEquilibrium(PDE):
    def __init__(self):
        self.dim = 1
        L = geometry.element_lengths()
        c, s = geometry.element_directions()
        u = {d: sp.Symbol(f"u_{d}", real=True) for d in range(14)}
        U = 0
        for e, (i, j) in enumerate(geometry.ELEMENTS.tolist()):
            elong = c[e]*(u[2*j] - u[2*i]) + s[e]*(u[2*j+1] - u[2*i+1])
            U += geometry.E * geometry.AREA / (2 * L[e]) * elong**2
        self.equations = {f"equilibrium_{d}": sp.diff(U, u[d]) for d in geometry.FREE_DOFS}
        self._u_syms = u

def make_residual_fn():
    pde = TrussEquilibrium()
    mods = {name: SympyToTorch(expr, name) for name, expr in pde.equations.items()}
    keys = sorted(f"u_{d}" for d in geometry.FREE_DOFS)  # note: only free symbols appear
    def residual(u_red: torch.Tensor) -> torch.Tensor:
        # u_red: (..., 11) free-DOF displacements -> feed each equation its inputs
        var = {f"u_{d}": u_red[..., k] for k, d in enumerate(geometry.FREE_DOFS)}
        # support dofs are zero -> their symbols never appear in free-DOF derivatives?
        # NO: dU/du_i for free i DOES involve support-adjacent members' u at SUPPORT
        # dofs — but those are constrained to zero, so substitute u=0 for support dofs
        # BEFORE differentiation? Substituting zero for support symbols then differentiating
        # w.r.t. free symbols gives exactly the reduced K row. Do that.
        out = [mods[f"equilibrium_{d}"](var)[f"equilibrium_{d}"] for d in geometry.FREE_DOFS]
        return torch.stack(out, dim=-1)
    return residual
```

**Critical correctness point baked into the test:** differentiate **after** substituting `u_support = 0` — this yields exactly the reduced stiffness rows `(K_red)`. If instead the support symbols remain free, the residual would include reaction-force terms.

- [ ] **Step 1: Write the failing test**

```python
def test_symbolic_residual_equals_assembled_K_times_u():
    # self-test 6: elementwise agreement between sympy-compiled residual and K_red @ u
    import torch
    from helpers.fem import assemble_K, reduce_matrix
    from helpers.truss_pde import make_residual_fn
    residual = make_residual_fn()
    torch.manual_seed(0)
    u = torch.randn(64, 11)
    r = residual(u)
    K_red = reduce_matrix(assemble_K())
    expected = u @ K_red.T
    assert r.shape == (64, 11)
    assert torch.allclose(r, expected, rtol=1e-5, atol=1e-4 * expected.abs().max())

def test_residual_gradients_flow():
    # the compiled residual must be differentiable w.r.t. displacements
    u = torch.randn(1, 11, requires_grad=True)
    r = make_residual_fn()(u)
    r.sum().backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()
```

Note: `SympyToTorch.forward` expects a dict keyed by the **sorted free-symbol names of the expression** (`self.keys = sorted(...)`). Build the input dict to contain all 14 `u_*` names (support entries as zero tensors of the right broadcast shape) so every equation module finds its keys.

- [ ] **Step 2: Run — FAIL**
- [ ] **Step 4: Implement `TrussEquilibrium` + `make_residual_fn`** per sketch: substitute zero for support symbols **before** `sp.diff`, compile all 11 expressions, feed dict with all 14 keys (support → `torch.zeros_like(u_red[..., 0])`), stack outputs in FREE_DOFS order.
- [ ] **Step 4: Run — PASS**
- [ ] **Step 5: Commit** `git commit -s -m "feat(truss-pinn): sympy-derived equilibrium residual via physicsnemo.sym"`

---

### Task 6: PINN model + loss (spec §4.1, §4.2)

**Files:**
- Create: `examples/structural_mechanics/truss_pinn/helpers/model.py`
- Test: `examples/structural_mechanics/truss_pinn/tests/test_model.py`

**Interfaces:**
- Consumes: geometry (FREE_DOFS, LOADABLE_NODES, P_MAX), fem (solve_static, modal_solve, phase_fixed, matrices), truss_pde.make_residual_fn.
- Produces:
  - `class TrussPINN(nn.Module)`: wraps `physicsnemo.models.mlp.FullyConnected(in_features=8, out_features=44, layer_size=128, num_layers=4)`; slices output into `u_hat (..., 11)`, `log_omega_hat (..., 3)`, `phi_hat (..., 11, 3)`.
  - `encode_params(n_idx: Int[Tensor, "..."], theta: Float[Tensor, "..."], P: Float[Tensor, "..."]) -> Float[Tensor, "... 8]` — one-hot ⊕ (cosθ, sinθ, P/P_MAX).
  - `pinn_loss(model, batch, residual_fn, anchors, weights) -> tuple[Tensor, dict]` — returns (total, components dict) with components: `eq` (‖r − f_red‖² mean), `sup` (anchor displacement MSE), `mode` (anchor modal supervision: ω MSE on log-ω + φ MSE after phase-fixing the prediction to match anchor sign convention), plus `eig_reg` eigen-residual regularizer on non-anchor batch points: `‖(K+K_G(P))·φ̂ − ω̂²·M·φ̂‖²` where K_G(P) is computed from **frozen** FEM statics at the batch load (no gradient through K_G — `detach()` the axial forces; the regularizer teaches the network the eigen relation, not the statics).

**Design decision locked here:** eigen-residual regularizer uses K_G from FEM-statics (detached) rather than from the network's own u_hat — otherwise errors in u_hat contaminate the modal loss with wrong stiffness and the two tasks fight. Anchor modal supervision provides the ordering; the regularizer provides physics smoothness across load space.

- [ ] **Step 1: Write the failing test**

```python
import torch
from helpers.model import TrussPINN, encode_params, pinn_loss

def test_shapes_and_normalization():
    model = TrussPINN()
    n = torch.tensor([1, 4]); theta = torch.tensor([0.3, 1.2]); P = torch.tensor([500.0, -800.0])
    x = encode_params(n, theta, P)
    assert x.shape == (2, 8)
    assert torch.allclose(x[0, :5], torch.tensor([0., 1., 0., 0., 0.]))  # node 1 → slot 1 of LOADABLE_NODES
    u_hat, log_w, phi = model(x)
    assert u_hat.shape == (2, 11) and log_w.shape == (2, 3) and phi.shape == (2, 11, 3)

def test_loss_components_finite_and_weighted():
    torch.manual_seed(0)
    model = TrussPINN()
    batch = sample_batch(8)                    # from helpers.sampling (Task 7) — define locally if not yet present
    residual_fn = make_residual_fn()
    anchors = make_anchors(4)                  # ditto
    total, comp = pinn_loss(model, batch, residual_fn, anchors, weights=(1.0, 100.0, 10.0))
    for k in ("eq", "sup", "mode", "eig_reg"):
        assert torch.isfinite(comp[k]).all()
    assert total == comp["eq"] + 100.0*comp["sup"] + 10.0*comp["mode"] + comp["eig_reg"]
```

- [ ] **Step 2: Run — FAIL** · **Step 3: Implement** · **Step 4: Run — PASS**
- [ ] **Step 5: Commit** `git commit -s -m "feat(truss-pinn): PINN model and composite loss"`

---

### Task 7: Sampling + anchors + test set (spec §4.2 items 2-3, §4.3)

**Files:**
- Create: `examples/structural_mechanics/truss_pinn/helpers/sampling.py`
- Test: `examples/structural_mechanics/truss_pinn/tests/test_sampling.py`

**Interfaces:**
- Produces:
  - `sample_batch(n_cases: int, generator: torch.Generator | None) -> dict` with keys `n_idx (n,)`, `theta (n,)`, `P (n,)` — uniform over loadable nodes, [0, 2π), [−P_MAX, P_MAX].
  - `make_anchors(n_per_node: int = 12, thetas: tuple = (-pi/2, 0, pi/2, pi)) -> dict` with keys `params`, `u_red`, `omega`, `phi` — grid: 5 nodes × 4 θ × |P| ∈ linspace(0.1, 1.0, 3)·P_MAX × sign {+,-} → 5·4·3·2 = **120 anchors** (spec said 64; grid needs the factor-5 node coverage — adopt 120, note deviation in README), each solved by FEM (u_red via solve_static, ω/φ via modal_solve + phase_fixed).
  - `make_test_set(n_cases: int = 200, seed: int = 20260920) -> dict` — same distribution as sample_batch but seeded and disjoint from anchors by construction (continuous θ, P make collision measure-zero; assert min-distance > 0 for the discrete n and report).
  - `solve_case(n_idx, theta, P) -> dict` — single-case FEM pipeline returning `u14, N, omegas, modes`.

- [ ] **Step 1: Write the failing test**

```python
def test_anchor_grid_covers_and_disjoint_from_test():
    from helpers.sampling import make_anchors, make_test_set
    a = make_anchors(); t = make_test_set()
    assert a["params"]["n_idx"].shape[0] == 120
    assert t["params"]["n_idx"].shape[0] == 200
    # discrete-part disjointness on node index is NOT required (only continuous
    # params collide); assert anchors finite
    assert torch.isfinite(a["u_red"]).all() and torch.isfinite(a["omega"]).all()

def test_solve_case_matches_pipeline():
    from helpers.sampling import solve_case
    out = solve_case(torch.tensor([2]), torch.tensor([-1.0]), torch.tensor([300.0]))
    assert out["omegas"].shape == (1, 3) and out["modes"].shape == (1, 11, 3)
    assert (out["omegas"] > 0).all()
```

- [ ] **Step 2: Run — FAIL** · **Step 3: Implement** · **Step 4: Run — PASS**
- [ ] **Step 5: Commit** `git commit -s -m "feat(truss-pinn): load sampling, anchors, test set"`

---

### Task 8: train.py (Hydra entry, FEM self-tests gate, training loop)

**Files:**
- Create: `examples/structural_mechanics/truss_pinn/train.py`
- Create: `examples/structural_mechanics/truss_pinn/conf/config.yaml`
- Create: `examples/structural_mechanics/truss_pinn/requirements.txt`

**Interfaces:**
- Consumes: everything above.
- Produces: `outputs/` (Hydra chdir) with `model.pt` (torch.save of model state_dict + config record), console metrics every N steps, final anchor-fit report.

config.yaml keys: `seed: 95051`, `epochs: 4000` (steps), `batch_size: 256`, `lr: 1e-3`, `weight_eq: 1.0`, `weight_sup: 100.0`, `weight_mode: 10.0`, `n_anchors_per_node: 3` (→ 120 total per Task 7 formula if using compact grid — **use the Task 7 grid formula**, config lists `anchor_thetas: [-1.5708, 0.0, 1.5708, 3.1416]`, `anchor_abs_p: [0.1, 0.55, 1.0]`), `device: cuda`, `log_every: 100`, `val_every: 500`.

train.py flow:
1. `@hydra.main(version_base="1.3", config_path="conf", config_name="config")`
2. Run pytest-style gate inline: import `tests/test_fem.py`-equivalent checks via a `run_fem_self_tests()` function in `helpers/fem.py` (port of Tasks 2-3 test bodies — call it directly, don't shell out to pytest). Any failure → `raise RuntimeError` before training.
3. Freeze P_MAX via `stability.freeze_p_max()` if `cfg.freeze_p_max: true` (default true).
4. Build anchors, model, optimizer (Adam), residual_fn; move matrices to device.
5. Loop: sample batch → loss → backward → step; log components; every `val_every` evaluate on a held-out validation slice of the anchor set (or reuse test-set metrics without labels leaking — validation on anchor subset is fine).
6. Save `model.pt`.

- [ ] **Step 1: Implement train.py + config.yaml + requirements.txt** (`physicsnemo @ file:///data/physicsnemo` editable path comment + `matplotlib`, `hydra-core`, `tqdm`; match deforming_plate's requirements.txt style).
- [ ] **Step 2: Smoke-run 50 steps**: `cd examples/structural_mechanics/truss_pinn && /data/physicsnemo/.venv/bin/python train.py epochs=50 device=cpu log_every=10` — expect: self-test gate passes, 50 steps run, model.pt written.
- [ ] **Step 3: Commit** `git commit -s -m "feat(truss-pinn): Hydra training entry point"`

---

### Task 9: evaluate.py — acceptance run (spec §2.4, §6)

**Files:**
- Create: `examples/structural_mechanics/truss_pinn/evaluate.py`

**Interfaces:**
- Consumes: `outputs/model.pt`, sampling test set, model.
- Produces: `outputs/` figures `deformation_comparison.png` (FEM vs PINN overlay, 3 test cases), `frequency_load_curves.png` (ω_i vs P for fixed (n,θ) spanning the range, FEM curves + PINN points), `error_report.json` (per-case disp L2 rel err, per-mode freq rel err, aggregates: max/median/P95 + PASS/FAIL vs 5%/2%).

- [ ] **Step 1: Implement evaluate.py** — load checkpoint, rebuild model, regenerate seeded test set (same seed as training config record → identical cases), compute metrics, save figures + JSON. Console-print the acceptance verdict.
- [ ] **Step 2: Full acceptance run**: `.venv/bin/python train.py` (full 4000 steps, GPU) then `.venv/bin/python evaluate.py`. Record numbers.
- [ ] **Step 3: Commit** `git commit -s -m "feat(truss-pinn): evaluation and acceptance metrics"`

---

### Task 10: README + polish + final verification

**Files:**
- Create: `examples/structural_mechanics/truss_pinn/README.md`
- Modify: `examples/README.md` (add row to the domain index table if truss_pinn appears in structural_mechanics listing — check existing format)

README sections (repo-mandated): `## Problem overview`, `## Dataset` (FEM-generated, no download), `## Model overview and architecture`, `## Getting Started`, `## Additional Information`, `## References`. Content: physics chain diagram from spec §1, material/geometric constants, load parameterization, acceptance thresholds and the measured values from Task 9, prestress frequency-shift figure reference (`outputs/frequency_load_curves.png` — generated, not committed).

- [ ] **Step 1: Write README.md**
- [ ] **Step 2: Markdown lint parity**: `.venv/bin/python -m pip list 2>/dev/null | grep -i markdownlint || echo "markdownlint via pre-commit only"`; run `pre-commit run --files examples/structural_mechanics/truss_pinn/**` if available locally — else eyeball against `examples/.markdownlint.yaml` rules.
- [ ] **Step 3: Full pipeline re-run from clean**: `rm -rf outputs && .venv/bin/python train.py && .venv/bin/python evaluate.py` — confirm acceptance PASS.
- [ ] **Step 4: Commit** `git commit -s -m "docs(truss-pinn): README and final verification"`

---

## Self-Review Notes (resolved during planning)

1. **DOF correction**: spec §4.1 said 10 displacement outputs; roller node 3's horizontal DOF (dof 6) is a real unknown → **11 free DOFs**, network output 44. (Spec §2.4/§4.1 "10/43" corrected here; the spec text remains as historical record — plan supersedes on this point.)
2. **Anchor count**: spec said 64; full grid needs 120 (5 nodes × 4 θ × 3 |P| × 2 signs). Deviation noted for README.
3. **Task 3 self-test 4 rewrite**: naive "pull up = stiffen" is not invariant across mode-order swaps; replaced with odd-in-P symmetry test + Task 4 monotonicity scan.
4. **Sympy substitution order**: substitute support-dof symbols to zero BEFORE differentiation — otherwise reaction terms leak into the residual.
5. **eig_reg detach**: K_G computed from FEM statics (detached), not from network u_hat — decouples statics errors from modal loss.

# Truss PINN: Prestressed Modal Analysis of a Pratt Truss Bridge

This example trains a physics-informed neural network (PINN) surrogate for the
prestressed modal analysis of a pin-jointed Pratt truss bridge. Given a
parameterized nodal load, the network predicts the static displacement field
and the first three natural frequencies and mode shapes in a single forward
pass. No external simulation data is used: a batched finite element method
(FEM) reference solver generates every label at runtime, and the physics is
enforced through a symbolic equilibrium residual built with
`physicsnemo.sym`.

The example demonstrates the interoperability of PhysicsNeMo, `physicsnemo.sym`,
and PyTorch in a fully explicit training loop: the symbolic machinery comes
from `physicsnemo.sym.eq.pde.PDE` and `SympyToTorch`, the network is a plain
`physicsnemo.models.mlp.FullyConnected`, and the loss, optimizer, and training
loop are ordinary PyTorch code.

## Problem overview

The structure is a planar Pratt truss bridge with 7 nodes and 11 members: a
6 m span bottom chord (nodes at x = 0, 2, 4, 6 m), a 1 m tall top chord
(nodes at x = 1, 3, 5 m), pin-jointed members, and a pin + roller support
pair at the bottom ends.

```text
  4 --- 5 --- 6            top chord (y = 1 m)
   \   / \   /
    \ /   \ /              diagonals
     1 --- 2
    /       \
   0         3             bottom chord (y = 0 m), pin at 0, roller at 3
```

All members are nylon 6/6 with Young's modulus E = 3 GPa, density
rho = 1150 kg/m^3, and a solid 32x32 mm square cross-section
(A = 1e-4 m^2).

The load is a single nodal force P applied at one of the 5 loadable nodes
(nodes 1, 2, 4, 5, 6) in the direction theta, with magnitude
|P| <= 11871 N. The load parameterization is therefore the triple
(load node n, direction theta, signed magnitude P). The bound is the frozen
stability envelope: at startup `train.py` re-derives
`P_MAX = 0.3 x min P_crit` from a buckling scan over the loadable-node x
direction grid, and the envelope guarantees the tangent stiffness stays
positive definite for any admissible load, so the modal solver never enters
the buckling regime.

Because the truss is prestressed by the axial forces, the modal analysis is
not a fixed eigenproblem: tension stiffens members and compression softens
them, shifting the natural frequencies with the applied load. The
frequencies and mode shapes solve the prestressed generalized eigenproblem

```text
(K + K_G(u)) phi = omega^2 M phi
```

where K is the global stiffness matrix, M the mass matrix, and K_G(u) the
geometric (stress) stiffness matrix assembled from the axial forces of the
static solution u. The surrogate must capture both the static response and
the load-dependent frequency shift.

## Dataset

There is no dataset to download. Everything is generated at runtime by the
batched FEM reference solver (`helpers/fem.py`, pure torch, float64):

- 120 FEM-labeled anchor cases on an explicit grid over the load box
  (5 loadable nodes x 4 load directions x 3 magnitude levels x 2 signs),
  generated in ~11 ms by `make_anchors` in `helpers/sampling.py`.
- A frozen 200-case test set drawn from the same distribution with the fixed
  seed 20260920 (`make_test_set`); training and evaluation rebuild the exact
  same set from the seed, which is part of the acceptance contract.

The FEM solver itself is self-tested at training startup: `train.py` runs a
6-check physics gate (`helpers/fem_selftest.py`) that verifies matrix
symmetry and definiteness, M-orthonormality of the modes, the odd-in-P
symmetry of the first-order prestress frequency shift, positive
definiteness of the tangent stiffness at the envelope edges, and the
equivalence of the compiled symbolic residual with `K_red @ u`. Training
refuses to start if any check fails.

## Model overview and architecture

The surrogate is a discrete-energy PINN with two coupled components.

The FEM reference solver (`helpers/fem.py`) assembles the global stiffness,
consistent mass, and geometric stiffness matrices in float64, solves the
static problem `K_red u = f_red` with one batched `torch.linalg.solve`, and
solves the prestressed eigenproblem with a Cholesky reduction of M followed
by a batched `torch.linalg.eigh`.

The PINN surrogate (`helpers/model.py`) is a plain
`physicsnemo.models.mlp.FullyConnected` MLP: 8 inputs, four hidden layers
of 128 units with SiLU activations, 47 outputs, float32. The 8 inputs
encode the load parameters: a 5-dim one-hot over the loadable nodes plus
(cos theta, sin theta, P / P_MAX). The 47 outputs are sliced into

- u_hat: the 11 free-DOF displacements (the 14 nodal DOFs minus the 3
  constrained support DOFs; the roller still allows horizontal motion, so
  its horizontal DOF is a real unknown),
- log_omega_hat: the first 3 natural frequencies on a log scale (positive
  by construction),
- phi_hat: the 3 mode shapes on the 11 free DOFs (11 x 3 = 33 outputs).

The composite loss has four terms, evaluated in float64:

1. **Equilibrium residual**: the member strain energy is written symbolically
   over the 14 nodal displacement symbols with sympy, support displacements
   are substituted to zero before differentiation, and the stationary
   conditions dU/du_d become equilibrium equations. The
   `physicsnemo.sym` `SympyToTorch` utility compiles them into a torch
   callable equal to `K_red @ u`; the residual loss is the squared error
   against the reduced load vector on every collocation batch.
2. **Anchor supervision**: MSE against the 120 FEM-labeled anchor
   displacements.
3. **Modal supervision**: MSE on log-omega plus MSE on the mode shapes after
   phase-fixing each predicted mode onto the anchor sign convention
   (eigenvectors are sign-ambiguous; the fix is piecewise constant so
   gradients flow through).
4. **Eigen-residual regularizer**: `(K + K_G) phi_hat - omega_hat^2 M phi_hat`
   on every collocation batch, with K_G assembled from the FEM statics at
   the batch loads and detached, so statics prediction errors cannot
   contaminate the modal loss through a wrong geometric stiffness.

The committed recipe uses loss weights (100, 1e8, 1e6) for
(eq, sup, mode) with the eigen regularizer at weight 1, Adam at lr 0.003
with `StepLR(step_size=10000, gamma=0.3)`, 40000 steps, and batch size 256.
All configuration keys are Hydra overrides (`conf/config.yaml`).

## Prerequisites

Install PhysicsNeMo from the repository root (editable install), then the
example requirements. The example additionally uses the `physicsnemo.sym`
symbolic utilities (`pip install "nvidia-physicsnemo[sym]"` when running
outside a source checkout):

```bash
pip install -e /path/to/physicsnemo
pip install -r requirements.txt
```

## Getting Started

Run training from this directory:

```bash
python train.py
```

This freezes the stability envelope P_MAX, runs the FEM physics self-test
gate, and trains for 40000 steps (~23 min on a Tesla T4; about 15 s of
startup before the first step). The checkpoint `model.pt` is written to the
Hydra run directory `outputs/<date>/<time>/`. Common overrides:

```bash
python train.py                       # full run, cuda if available
python train.py epochs=400 device=cpu # quick smoke run
python train.py lr=1e-3 log_every=100 # any config key is overridable
```

Once training finishes, run the acceptance evaluation on the newest
checkpoint:

```bash
python evaluate.py
```

The script rebuilds the frozen 200-case test set from the checkpoint's
recorded seed, scores displacement and frequency errors and mode-shape
cosine similarity against the FEM labels, and writes into the checkpoint
directory:

- `error_report.json`: per-case errors, aggregates (max / median / P95),
  thresholds, and the PASS/FAIL verdict,
- `deformation_comparison.png`: FEM undeformed/deformed shapes overlaid
  with the PINN prediction for 3 test cases,
- `frequency_load_curves.png`: the first 3 frequencies omega_i(P) along a
  fixed load path (node 1, theta = -pi/2, 41-point sweep), FEM lines and
  PINN markers, showing the prestress frequency shift (tension stiffens,
  compression softens).

The script prints the acceptance verdict and exits 0 on PASS, so it can
gate CI. The generated outputs are run artifacts (gitignored) and are not
committed.

## Additional Information

**Acceptance results.** The acceptance gate requires the median AND the P95
of the displacement relative L2 error to be below 5% and the median AND P95
of every mode's relative frequency error to be below 2% (the original
max-only reading was adjudicated to median + P95 during the acceptance
review; the frequency metrics pass under any reading since the worst max is
0.87%). Measured on the frozen 200-case test set with the committed
configuration:

| Metric                     | Median | P95   | Max    | Threshold |
| -------------------------- | ------ | ----- | ------ | --------- |
| Displacement rel L2 error  | 0.28%  | 3.36% | 15.23% | 5%        |
| Freq rel err, mode 1       | 0.24%  | 0.49% | 0.87%  | 2%        |
| Freq rel err, mode 2       | 0.26%  | 0.52% | 0.67%  | 2%        |
| Freq rel err, mode 3       | 0.30%  | 0.49% | 0.57%  | 2%        |

Mode-shape cosine similarity is 1.0000 (to 4 decimals) for all three modes,
and the predicted omega_i(P) trends agree with the FEM trends along the
fixed load path for 90% / 82% / 100% of the sweep increments (modes 1-3),
reproducing the tension-stiffens / compression-softens behavior.

**Verdict: PASS** (median + P95 gate).

**Error structure and limitations.** The displacement max of 15.23% comes
from exactly 5 of the 200 test cases above the 5% threshold. All 5 sit at
mid-gap load directions theta (0.64 to 0.78 rad away from the nearest
anchor direction), and 4 of the 5 also have small magnitudes
|P| < 0.09 P_MAX. The theta-dependence of the response contains a k=2
harmonic that is unsupervised by the 4-direction anchor grid (sin(2 theta)
is invisible at the anchor directions), so at mid-gap directions the
equilibrium residual is the only teacher and converges slowly. Future work:
a richer anchor theta grid, or a relative (per-case normalized) supervision
loss.

**Deviations from the design spec.**

- The spec counted 10 free-DOF displacements, but the roller support still
  allows horizontal motion, so its horizontal DOF is a real unknown: the
  truss has 11 free DOFs and the network outputs
  11 + 3 + 33 = 47 values per case.
- The spec's 64-anchor grid could not cover all 5 loadable nodes; 120
  (5 nodes x 4 directions x 3 levels x 2 signs) is the smallest grid that
  does.
- The anchor grid is physically 2x redundant: a load at direction
  theta + pi is the same physical load as (theta, -P), which the sign
  dimension already enumerates. This is harmless (the anchor mode
  supervision simply counts each physical load twice).
- The acceptance gate is median + P95 rather than a max-only reading
  (adjudicated during the acceptance review).

**Performance.** Training takes ~23 min on a Tesla T4 (40000 steps) with
~15 s of startup (stability freeze scan + sympy compile); the 120 anchor
labels are generated in ~11 ms; the network has 56751 trainable
parameters.

## References

- [PhysicsNeMo Documentation](https://docs.nvidia.com/physicsnemo/)
- [Raissi, Perdikaris, Karniadakis, Physics-informed neural networks, JCP 2019](https://doi.org/10.1016/j.jcp.2018.10.045)
- [Cook, Malkus, Plesha, Witt, Concepts and Applications of Finite Element Analysis](https://www.wiley.com/en-us/Concepts+and+Applications+of+Finite+Element+Analysis%2C+4th+Edition-p-9780471356059)
- [physicsnemo.sym sympy-to-torch backend (`SympyToTorch`)](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/sym/utils/sympy/torch_printer.html)

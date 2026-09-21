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

"""Acceptance evaluation for the prestressed Pratt truss PINN (spec 2.4, 6).

Pipeline:

1. Load a checkpoint (model.pt) written by train.py: state dict, config
   record, and the frozen P_MAX.  The config's ``test_seed`` rebuilds the
   frozen test set (helpers.sampling.make_test_set) -- training and
   evaluation use the same seed, so the case list is identical.
2. Predict every test case through the trained TrussPINN and score it
   against the FEM labels:
   - displacement relative L2 error per case,
   - relative frequency error per mode (3 modes),
   - mode-shape cosine similarity per mode after phase-fixing the
     prediction to the label sign convention (informative only).
3. Aggregate max / median / P95 per metric family and print the
   acceptance verdict (spec 2.4): PASS iff median AND P95 of the
   displacement relative L2 error are < 5% AND median AND P95 of every
   mode's relative frequency error are < 2%.
4. Save error_report.json (per-case errors + aggregates + verdict) and
   two figures next to the checkpoint:
   - deformation_comparison.png: FEM undeformed/deformed vs PINN
     deformed overlay for 3 test cases,
   - frequency_load_curves.png: omega_i vs P over a 41-point sweep at a
     fixed load path (node 1, theta = -pi/2), FEM lines + PINN markers.
5. Physics consistency check (spec 6 item 4): the predicted omega_1(P)
   trend must match the FEM trend along the sweep -- the sign of each
   successive increment d(omega)/dP must agree; the agreement fraction
   is printed and written to the report.

Run (from this directory)::

    python evaluate.py                          # newest outputs/**/model.pt
    python evaluate.py path/to/model.pt         # explicit checkpoint
"""

import glob
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 (backend set above)
import torch  # noqa: E402

# Make the helpers package importable regardless of the launch directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers import geometry  # noqa: E402
from helpers.fem import modal_solve, nodal_load, reduce_matrix  # noqa: E402
from helpers.fem import assemble_K, solve_static  # noqa: E402
from helpers.model import TrussPINN, encode_params  # noqa: E402
from helpers.sampling import make_test_set  # noqa: E402

_DTYPE = torch.float64
_N_DOF = 2 * geometry.NODES_XY.shape[0]  # 14
_N_MODES = 3
_FREE = torch.tensor(geometry.FREE_DOFS)
_PHASE_EPS = 1e-12

_DISP_THRESH = 0.05  # spec 2.4: displacement relative L2 error < 5%
_FREQ_THRESH = 0.02  # spec 2.4: relative frequency error < 2%

# Fixed load path for the frequency-load curves (spec 6 item 4).
_CURVE_NODE = 1
_CURVE_THETA = -0.5 * math.pi
_CURVE_POINTS = 41


def find_latest_checkpoint():
    """Newest outputs/**/model.pt under the current directory, or None."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = glob.glob(
        os.path.join(here, "outputs", "**", "model.pt"), recursive=True
    )
    return max(candidates, key=os.path.getmtime) if candidates else None


def load_model(ckpt_path):
    """Load checkpoint -> (model, cfg dict, device).

    torch.load(weights_only=True) keeps the untrusted-pickle door shut;
    the config record travels as plain python types.  The device comes
    from the config with a cpu fallback (evaluation must run anywhere).
    """
    ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    cfg = dict(ckpt["cfg"])
    geometry.P_MAX = float(ckpt["p_max"])
    requested = str(cfg.get("device", "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    device = torch.device(requested)
    model = TrussPINN()
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    return model, cfg, device


def phase_fix(phi_hat, phi_label):
    """Flip predicted modes onto the label sign convention.

    phi_hat, phi_label: (..., N_FREE, N_MODES).  Returns the phase-fixed
    prediction (same dtype as phi_hat).  Mirrors the loss-side phase fix
    (largest-entry sign fallback when the overlap is ~0).
    """
    overlap = (phi_hat * phi_label).sum(dim=-2)
    lead_hat = phi_hat.abs().argmax(dim=-2)
    lead_lab = phi_label.abs().argmax(dim=-2)
    fallback = (
        phi_hat.gather(-2, lead_hat.unsqueeze(-2)).sign()
        * phi_label.gather(-2, lead_lab.unsqueeze(-2)).sign()
    ).squeeze(-2)
    s = torch.where(overlap.abs() < _PHASE_EPS, fallback, overlap.sign())
    return s.unsqueeze(-2) * phi_hat


def evaluate_test_set(model, device, test_seed):
    """Frozen test set + per-case metrics -> (test dict, metrics dict).

    test_seed rebuilds the SAME frozen test set training recorded in the
    config (part of the contract): identical case list, identical FEM
    labels (up to LAPACK roundoff, see helpers.sampling docstring).
    """
    test = make_test_set(200, seed=int(test_seed))
    with torch.no_grad():
        x = encode_params(test["n_idx"], test["theta"], test["P"]).to(device)
        u_hat, log_omega_hat, phi_hat = model(x)
    u_hat = u_hat.to(_DTYPE).cpu()
    omega_hat = log_omega_hat.to(_DTYPE).exp().cpu()
    phi_hat = phi_hat.to(_DTYPE).cpu()

    u_ref = test["u_red"]
    disp_rel = (u_hat - u_ref).norm(dim=-1) / u_ref.norm(dim=-1)
    freq_rel = (omega_hat - test["omega"]).abs() / test["omega"]
    phi_fixed = phase_fix(phi_hat, test["phi"])
    cos = (phi_fixed * test["phi"]).sum(dim=-2) / (
        phi_fixed.norm(dim=-2) * test["phi"].norm(dim=-2)
    )
    metrics = {
        "disp_rel_l2": disp_rel,
        "freq_rel_err": freq_rel,
        "mode_cos_sim": cos,
    }
    return test, metrics


def _stats(t):
    """max / median / P95 of a 1-D tensor -> python floats."""
    t = t.detach().cpu().to(_DTYPE)
    return {
        "max": float(t.max()),
        "median": float(t.median()),
        "p95": float(torch.quantile(t, 0.95)),
    }


def frequency_load_curves(model, device):
    """omega_i(P) along the fixed load path: FEM and PINN -> tensors.

    Returns (Ps, omega_fem (41, 3), omega_pinn (41, 3)).
    """
    Ps = torch.linspace(-geometry.P_MAX, geometry.P_MAX, _CURVE_POINTS, dtype=_DTYPE)
    n_idx = torch.full((_CURVE_POINTS,), _CURVE_NODE, dtype=torch.int64)
    theta = torch.full((_CURVE_POINTS,), _CURVE_THETA, dtype=_DTYPE)

    K_red = reduce_matrix(assemble_K())
    om_fem = []
    for p in Ps.tolist():
        f = nodal_load(_CURVE_NODE, _CURVE_THETA, p)
        u_red = solve_static(K_red, f[_FREE])
        u14 = torch.zeros(_N_DOF, dtype=_DTYPE)
        u14[_FREE] = u_red
        omegas, _ = modal_solve(u14, n_modes=_N_MODES)
        om_fem.append(omegas)
    omega_fem = torch.stack(om_fem)  # (41, 3)

    with torch.no_grad():
        x = encode_params(n_idx, theta, Ps).to(device)
        _, log_omega_hat, _ = model(x)
    omega_pinn = log_omega_hat.to(_DTYPE).exp().cpu()
    return Ps, omega_fem, omega_pinn


def trend_agreement(omega_a, omega_b):
    """Monotonic-trend agreement of two omega(P) sweeps -> fraction.

    Spec 6 item 4: the PINN must reproduce the FEM d(omega)/dP trend
    (tension stiffens / compression softens along a fixed path).  Each
    successive increment contributes a vote: agreement requires matching
    sign of diff(omega) in both curves.
    """
    da = omega_a.diff().sign()
    db = omega_b.diff().sign()
    return float((da == db).float().mean())


def plot_deformation_comparison(fig_path, test, model, device):
    """FEM undeformed/deformed vs PINN deformed overlay, 3 test cases."""
    nodes = geometry.NODES_XY.to(_DTYPE)
    elems = geometry.ELEMENTS
    idx = torch.linspace(0, test["u_red"].shape[0] - 1, 3).long()

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    for ax, i in zip(axes, idx.tolist()):
        n = int(test["n_idx"][i])
        th = float(test["theta"][i])
        P = float(test["P"][i])
        u_red = test["u_red"][i]

        u14_fem = torch.zeros(_N_DOF, dtype=_DTYPE)
        u14_fem[_FREE] = u_red
        # scale for visibility: the deformation is small vs the span
        scale = 0.15 * 6.0 / max(u14_fem.abs().max().item(), 1e-12)

        with torch.no_grad():
            x = encode_params(
                test["n_idx"][i : i + 1],
                test["theta"][i : i + 1],
                test["P"][i : i + 1],
            ).to(device)
            u_hat, _, _ = model(x)
        u14_pinn = torch.zeros(_N_DOF, dtype=_DTYPE)
        u14_pinn[_FREE] = u_hat[0].to(_DTYPE).cpu()

        for geom, kwargs in (
            (nodes, dict(color="0.6", lw=2.0, label="undeformed")),
            (
                nodes + scale * u14_fem.view(-1, 2),
                dict(color="tab:blue", lw=2.0, label="FEM deformed"),
            ),
            (
                nodes + scale * u14_pinn.view(-1, 2),
                dict(color="tab:red", lw=1.4, ls="--", label="PINN deformed"),
            ),
        ):
            for e in elems.tolist():
                ax.plot(geom[e, 0], geom[e, 1], **kwargs)
            ax.plot([], [], **kwargs)  # ensure legend entries exist
        # mark the loaded node
        ax.plot(nodes[n, 0], nodes[n, 1], "k^", ms=8)
        disp_rel = ((u_hat[0].to(_DTYPE).cpu() - u_red).norm() / u_red.norm()).item()
        ax.set_title(
            f"case {i}: node {n}, P = {P:.0f} N\ndisp rel L2 err = {disp_rel:.2%}"
        )
        ax.set_aspect("equal")
        ax.axis("off")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Deformation: FEM vs PINN (displacements scaled for visibility)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def plot_frequency_load_curves(fig_path, Ps, omega_fem, omega_pinn):
    """omega_i vs P: FEM lines + PINN markers for the 3 modes."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i in range(_N_MODES):
        ax.plot(
            Ps.tolist(),
            omega_fem[:, i].tolist(),
            color=f"C{i}",
            lw=2.0,
            label=f"FEM mode {i + 1}",
        )
        ax.plot(
            Ps.tolist(),
            omega_pinn[:, i].tolist(),
            color=f"C{i}",
            ls="none",
            marker="o",
            ms=4,
            label=f"PINN mode {i + 1}",
        )
    ax.axvline(0.0, color="0.8", lw=0.8)
    ax.set_xlabel("P [N]  (negative = load reversed)")
    ax.set_ylabel(r"$\omega_i$ [rad/s]")
    ax.set_title(
        r"Frequency-load curves at node "
        rf"{_CURVE_NODE}, $\theta = -\pi/2$"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main():
    """Run the acceptance evaluation; print the verdict."""
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_checkpoint()
    if ckpt_path is None:
        raise SystemExit("no checkpoint found: pass a model.pt path")
    ckpt_path = os.path.abspath(ckpt_path)
    out_dir = os.path.dirname(ckpt_path)

    model, cfg, device = load_model(ckpt_path)
    print(f"checkpoint: {ckpt_path}")
    print(f"config: {cfg}")
    print(f"device: {device}")

    # --- frozen test set + per-case metrics
    test, res = evaluate_test_set(model, device, cfg["test_seed"])
    disp_stats = _stats(res["disp_rel_l2"])
    freq_stats = [_stats(res["freq_rel_err"][:, i]) for i in range(_N_MODES)]
    cos_stats = [_stats(res["mode_cos_sim"][:, i]) for i in range(_N_MODES)]
    # --- frequency-load curves + trend agreement (spec 6 item 4)
    Ps, omega_fem, omega_pinn = frequency_load_curves(model, device)
    trend = [
        trend_agreement(omega_fem[:, i], omega_pinn[:, i]) for i in range(_N_MODES)
    ]

    # --- acceptance verdict (spec 2.4)
    disp_pass = disp_stats["median"] < _DISP_THRESH and disp_stats["p95"] < _DISP_THRESH
    freq_pass = all(
        s["median"] < _FREQ_THRESH and s["p95"] < _FREQ_THRESH for s in freq_stats
    )
    verdict = "PASS" if (disp_pass and freq_pass) else "FAIL"

    # --- report JSON
    report = {
        "checkpoint": ckpt_path,
        "config": cfg,
        "n_test_cases": int(res["disp_rel_l2"].shape[0]),
        "thresholds": {
            "disp_rel_l2": _DISP_THRESH,
            "freq_rel_err": _FREQ_THRESH,
        },
        "disp_rel_l2": disp_stats,
        "freq_rel_err": {f"mode_{i + 1}": s for i, s in enumerate(freq_stats)},
        "mode_cos_sim": {f"mode_{i + 1}": s for i, s in enumerate(cos_stats)},
        "trend_agreement": {f"mode_{i + 1}": trend[i] for i in range(_N_MODES)},
        "disp_pass": bool(disp_pass),
        "freq_pass": bool(freq_pass),
        "verdict": verdict,
        "per_case": {
            "n_idx": test["n_idx"].tolist(),
            "theta": test["theta"].tolist(),
            "P": test["P"].tolist(),
            "disp_rel_l2": res["disp_rel_l2"].tolist(),
            "freq_rel_err": res["freq_rel_err"].tolist(),
            "mode_cos_sim": res["mode_cos_sim"].tolist(),
        },
    }
    report_path = os.path.join(out_dir, "error_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"error report: {report_path}")

    # --- figures
    plot_deformation_comparison(
        os.path.join(out_dir, "deformation_comparison.png"), test, model, device
    )
    plot_frequency_load_curves(
        os.path.join(out_dir, "frequency_load_curves.png"),
        Ps,
        omega_fem,
        omega_pinn,
    )
    print(
        "figures: "
        + os.path.join(out_dir, "deformation_comparison.png")
        + ", "
        + os.path.join(out_dir, "frequency_load_curves.png")
    )

    # --- console verdict
    print()
    print("=" * 72)
    print("ACCEPTANCE EVALUATION (spec 2.4)")
    print("=" * 72)
    print(
        f"displacement rel L2 err:  median {disp_stats['median']:.4%}  "
        f"P95 {disp_stats['p95']:.4%}  max {disp_stats['max']:.4%}  "
        f"(threshold 5%)"
    )
    for i, s in enumerate(freq_stats):
        print(
            f"freq rel err mode {i + 1}:    median {s['median']:.4%}  "
            f"P95 {s['p95']:.4%}  max {s['max']:.4%}  (threshold 2%)"
        )
    for i, s in enumerate(cos_stats):
        print(
            f"mode {i + 1} shape cos sim: median {s['median']:.4f}  "
            f"P95 {s['p95']:.4f}  (informative)"
        )
    for i, t in enumerate(trend):
        print(f"omega_{i + 1}(P) trend agreement with FEM: {t:.0%}")
    print("=" * 72)
    print(f"VERDICT: {verdict}")
    print("=" * 72)

    # exit code 0 iff PASS (scriptable acceptance gate)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

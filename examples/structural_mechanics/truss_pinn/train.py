# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

"""Hydra training entry point for the prestressed Pratt truss PINN.

Pipeline (spec section 2.4):

1. Resolve the training device from the config (cuda falls back to cpu
   with a warning when unavailable).  On cuda the torch default device is
   switched BEFORE the helpers are imported: helpers build their constants
   and FEM caches at import time (slot lookup, K/M/K_G), so the default
   device decides where the entire pipeline -- model, anchors, matrices,
   collocation batches -- lives.
2. Freeze the stability envelope P_MAX at runtime
   (stability.freeze_p_max re-derives the shipped geometry.P_MAX).
3. Gate on the FEM physics self-tests (helpers.fem_selftest); any failure
   raises RuntimeError before the first training step.
4. Train: a fresh collocation batch per step from a per-step seeded
   generator (reproducible run), Adam on the composite PINN loss,
   component logging every log_every steps and validation metrics on the
   last 20 anchors every val_every steps.
5. Save model.pt (state dict + config record + frozen P_MAX) into the
   Hydra run directory (outputs/<date>/<time>/ by default).

Run (from this directory)::

    python train.py                       # full run, cuda if available
    python train.py epochs=50 device=cpu  # quick smoke test

Untrained loss-component magnitudes (measured): eq ~ 4e7, sup ~ 1e-2,
mode ~ 3e1, eig_reg ~ 3e7.  Training drives eq and eig_reg down by orders
of magnitude while sup and mode fall toward zero.
"""
import logging
import os
import sys
import time

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from physicsnemo.utils.logging import PythonLogger

# Make the helpers package importable regardless of the launch directory
# (Hydra 1.3 with version_base="1.3" does not change the working
# directory; the run artifacts land in an absolute outputs/ path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(name)s - %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = PythonLogger("main")

_COMPONENTS = ("eq", "sup", "mode", "eig_reg")
_N_VAL_ANCHORS = 20  # held-out validation slice: the last 20 anchors


def _resolve_device(requested: str) -> torch.device:
    """Map the requested device string to an available torch device."""
    requested = str(requested)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("cuda requested but not available; falling back to cpu")
        return torch.device("cpu")
    return torch.device(requested)


def _validation_metrics(model, anchors: dict) -> dict:
    """Relative errors of the CURRENT model on the held-out anchor slice.

    Displacement: relative L2 error against the FEM labels.  Frequency:
    mean over the 3 modes of |omega_hat / omega - 1|.  Informative only
    -- validation never gates training.
    """
    from helpers.model import encode_params  # noqa: PLC0415 (device order)

    params = anchors["params"]
    n = anchors["u_red"].shape[0]
    sl = slice(n - _N_VAL_ANCHORS, n)
    with torch.no_grad():
        x = encode_params(
            params["n_idx"][sl], params["theta"][sl], params["P"][sl]
        )
        u_hat, log_omega_hat, _ = model(x)
        u_hat = u_hat.to(torch.float64)
        disp_rel = (
            (u_hat - anchors["u_red"][sl]).norm() / anchors["u_red"][sl].norm()
        ).item()
        omega_hat = log_omega_hat.to(torch.float64).exp()
        freq_rel = (
            (omega_hat - anchors["omega"][sl]).abs() / anchors["omega"][sl]
        ).mean().item()
    return {"disp_rel_l2": disp_rel, "freq_rel_err": freq_rel}


def _train(cfg: DictConfig, device: torch.device) -> tuple:
    """Run the training loop; return (model, final loss components)."""
    # Imported here (not at module top) so torch.set_default_device in
    # main() runs first and the helpers' import-time constants and FEM
    # caches are built on the training device.
    from helpers import geometry, stability
    from helpers.fem_selftest import run_fem_self_tests
    from helpers.model import TrussPINN, pinn_loss
    from helpers.sampling import make_anchors, sample_batch
    from helpers.truss_pde import make_residual_fn

    torch.manual_seed(int(cfg.seed))

    if cfg.freeze_p_max:
        p_max = stability.freeze_p_max()
        logger.info(f"froze stability envelope P_MAX = {p_max:.6f} N")
    else:
        p_max = float(geometry.P_MAX)
        logger.info(f"using shipped P_MAX = {p_max:.6f} N (no re-freeze)")

    logger.info("running FEM physics self-tests...")
    run_fem_self_tests()
    logger.info("FEM self-tests passed")

    anchors = make_anchors()
    logger.info(f"built {anchors['u_red'].shape[0]} FEM-labeled anchors")

    model = TrussPINN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"TrussPINN parameters: {n_params}")

    residual_fn = make_residual_fn()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
    weights = (float(cfg.weight_eq), float(cfg.weight_sup), float(cfg.weight_mode))
    logger.info(
        f"training {cfg.epochs} steps on {device}: batch {cfg.batch_size}, "
        f"lr {cfg.lr}, weights {weights}"
    )

    comp = {}
    start = time.time()
    for step in range(cfg.epochs):
        # Fresh per-step draws from a generator seeded by cfg.seed + step
        # (a cpu generator cannot serve draws under a cuda default device).
        gen = torch.Generator(device=torch.get_default_device())
        gen.manual_seed(int(cfg.seed) + step)
        batch = sample_batch(cfg.batch_size, generator=gen)
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        total, comp = pinn_loss(model, batch, residual_fn, anchors, weights)
        total.backward()
        optimizer.step()

        done = step + 1
        if done % cfg.log_every == 0 or done == cfg.epochs or done == 1:
            parts = " ".join(
                f"{name}={comp[name].item():.4e}" for name in _COMPONENTS
            )
            per_step = 1e3 * (time.time() - start) / done
            logger.info(f"step {done:6d}  {parts}  [{per_step:.1f} ms/step]")
        if cfg.val_every > 0 and (done % cfg.val_every == 0 or done == cfg.epochs):
            val = _validation_metrics(model, anchors)
            logger.info(
                f"validation @ {done:6d}  disp rel L2 err "
                f"{val['disp_rel_l2']:.4e}  freq rel err {val['freq_rel_err']:.4e}"
            )

    return model, comp


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train the truss PINN according to the Hydra config."""
    device = _resolve_device(cfg.device)
    if device.type == "cuda":
        torch.set_default_device(device)
    logger.info(f"device: {device}")

    model, comp = _train(cfg, device)

    logger.info(
        "final components: "
        + " ".join(f"{name}={comp[name].item():.6e}" for name in _COMPONENTS)
    )

    output_dir = HydraConfig.get().runtime.output_dir
    ckpt_path = os.path.join(output_dir, "model.pt")
    from helpers import geometry  # noqa: PLC0415 (device order)

    torch.save(
        {
            "state_dict": {
                k: v.detach().to("cpu") for k, v in model.state_dict().items()
            },
            "cfg": OmegaConf.to_container(cfg),
            "p_max": float(geometry.P_MAX),
        },
        ckpt_path,
    )
    logger.info(f"saved model checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()

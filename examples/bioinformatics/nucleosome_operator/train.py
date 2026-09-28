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

"""Hydra training entry point for the nucleosome-positioning operator.

Pipeline:

1. Freeze the sequence -> energy-landscape model (seeded table) and run the
   reference-solver self-tests; any failure raises before the first step.
2. Build the training and validation corpora and solve their exact
   occupancy labels with the transfer-matrix reference.
3. Fit the FNO operator on minibatches of (one-hot sequence, energy
   landscape, inverse temperature) -> occupancy, optimizing the per-sample
   squared relative L2 error -- the metric the acceptance gate reports.
4. Write ``model.pt`` (weights + full config record + input normalization
   constants + the energy-table parameters) into the Hydra run directory.

The checkpoint carries every constant needed to rebuild the *same* energy
model and the *same* test corpus, so ``evaluate.py`` never has to trust
anything but the checkpoint.

Run (from this directory)::

    python train.py                     # full run, cuda if available
    python train.py steps=200 device=cpu   # quick smoke test
"""

import json
import os
import sys
import time

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from physicsnemo.utils.logging import PythonLogger

# Make the helpers package importable regardless of the launch directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers import data as data_mod  # noqa: E402
from helpers import energy as energy_mod  # noqa: E402
from helpers import model as model_mod  # noqa: E402
from helpers import selftest  # noqa: E402

_DTYPE = torch.float64


def _resolve_device(requested: str) -> torch.device:
    """Map the requested device string to an available torch device."""
    requested = str(requested)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _config_record(cfg: DictConfig, norm: dict) -> dict:
    """Everything ``evaluate.py`` needs to rebuild the corpus and inputs."""
    record = OmegaConf.to_container(cfg, resolve=True)
    record["normalization"] = norm
    return record


def _val_metrics(
    model: torch.nn.Module, split: dict, norm: dict, device: torch.device
) -> dict:
    """Relative-L2 statistics on the validation corpus (in-sample monitor)."""
    from helpers.model import sample_batch

    model.eval()
    with torch.no_grad():
        errors = []
        for start in range(0, split["seq"].shape[0], 256):
            index = torch.arange(start, min(start + 256, split["seq"].shape[0]))
            x, y = sample_batch(split, index, norm, device)
            pred = model(x).to(_DTYPE).cpu()
            errors.append(data_mod.relative_l2(pred, y.to(_DTYPE).cpu()))
    model.train()
    err = torch.cat(errors)
    return {
        "rel_l2_median": float(err.median()),
        "rel_l2_p95": float(torch.quantile(err, 0.95)),
        "rel_l2_max": float(err.max()),
    }


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train the operator and write the checkpoint."""
    logger = PythonLogger("main")
    torch.manual_seed(int(cfg.seed))
    device = _resolve_device(cfg.device)
    logger.info(f"training on {device}")

    table = energy_mod.make_energy_table(
        seed=int(cfg.table_seed), width=int(cfg.width), scale=float(cfg.energy_scale)
    )
    logger.info(
        f"energy table: seed={int(cfg.table_seed)} width={int(cfg.width)} "
        f"scale={float(cfg.energy_scale)}"
    )

    logger.info("running reference-solver self-tests...")
    diagnostics = selftest.run_self_tests(seed=int(cfg.seed))
    logger.info("self-tests passed")

    t0 = time.time()
    train_split = data_mod.build_split(
        int(cfg.n_train),
        int(cfg.seq_length),
        int(cfg.seed),
        table,
        gc_low=float(cfg.gc_low),
        gc_high=float(cfg.gc_high),
        planted_fraction=float(cfg.planted_fraction),
        gamma_low=float(cfg.gamma_low),
        gamma_high=float(cfg.gamma_high),
    )
    val_split = data_mod.build_split(
        int(cfg.n_val),
        int(cfg.seq_length),
        int(cfg.seed) + int(cfg.val_seed_offset),
        table,
        gc_low=float(cfg.gc_low),
        gc_high=float(cfg.gc_high),
        planted_fraction=float(cfg.planted_fraction),
        gamma_low=float(cfg.gamma_low),
        gamma_high=float(cfg.gamma_high),
    )
    logger.info(
        f"corpus: {int(cfg.n_train)} train / {int(cfg.n_val)} val sequences of "
        f"{int(cfg.seq_length)} bp in {time.time() - t0:.1f}s; "
        f"mean occupancy {float(train_split['occupancy'].to(_DTYPE).mean()):.4f}"
    )

    norm = data_mod.normalization_stats(train_split)

    model = model_mod.NucleosomeOperator(
        latent_channels=int(cfg.latent_channels),
        num_fno_layers=int(cfg.num_fno_layers),
        num_fno_modes=int(cfg.num_fno_modes),
        padding=int(cfg.padding),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"operator: {n_params} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(cfg.lr_step_size), gamma=float(cfg.lr_gamma)
        )
        if int(cfg.lr_step_size) > 0
        else None
    )
    loss_fn = model_mod.make_loss(str(cfg.loss))

    # The whole corpus fits on the device: minibatches are gathered by index.
    order_gen = torch.Generator().manual_seed(int(cfg.seed) + 7)
    batch_size = int(cfg.batch_size)
    steps = int(cfg.steps)
    running: list[float] = []

    for step in range(1, steps + 1):
        index = torch.randint(0, int(cfg.n_train), (batch_size,), generator=order_gen)
        x, y = model_mod.sample_batch(train_split, index, norm, device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running.append(float(loss.detach()))
        if step % int(cfg.log_every) == 0 or step == 1:
            window = torch.tensor(running[-int(cfg.log_every) :]).median()
            logger.info(
                f"step {step:>6}/{steps}  loss {float(window):.3e}  "
                f"lr {optimizer.param_groups[0]['lr']:.2e}"
            )
        if step % int(cfg.val_every) == 0 or step == steps:
            metrics = _val_metrics(model, val_split, norm, device)
            logger.info(
                f"step {step:>6}/{steps}  val rel L2  median "
                f"{metrics['rel_l2_median']:.4%}  P95 {metrics['rel_l2_p95']:.4%}  "
                f"max {metrics['rel_l2_max']:.4%}"
            )

    final = _val_metrics(model, val_split, norm, device)
    logger.info(
        f"final val rel L2: median {final['rel_l2_median']:.4%}  "
        f"P95 {final['rel_l2_p95']:.4%}  max {final['rel_l2_max']:.4%}"
    )

    run_dir = HydraConfig.get().runtime.output_dir
    ckpt_path = os.path.join(run_dir, "model.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": _config_record(cfg, norm),
            "val_metrics": final,
            "self_tests": diagnostics,
            "params": n_params,
        },
        ckpt_path,
    )
    logger.info(f"checkpoint written to {ckpt_path}")
    with open(os.path.join(run_dir, "summary.json"), "w") as handle:
        json.dump(
            {
                "params": n_params,
                "val_metrics": final,
                "self_tests": diagnostics,
                "steps": steps,
                "checkpoint": ckpt_path,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()

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

"""Receptive-field ablation: why the surrogate needs a spectral operator.

A local convolutional surrogate sees only a few dozen base pairs around each
position, but the label depends on the 147 bp footprint window *and*, through
the partition function, on the whole locus.  This script trains convolutional
operators with deliberately restricted receptive fields under the same
corpus, loss, optimizer budget and normalization as the FNO, and reports the
error gap:

===============  ==========================================================
receptive field  construction
===============  ==========================================================
33 bp            four kernel-9 convolutions (dilation 1)
121 bp           four kernel-9 convolutions (dilation 1, 2, 4, 8)
full locus       the trained FNO checkpoint (spectral mixing)
===============  ==========================================================

Run (from this directory)::

    python ablate_receptive_field.py                       # newest checkpoint
    python ablate_receptive_field.py path/to/model.pt --steps 4000
"""

import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers import data as data_mod  # noqa: E402
from helpers import energy as energy_mod  # noqa: E402
from helpers import model as model_mod  # noqa: E402

_DTYPE = torch.float64


class LocalConvOperator(torch.nn.Module):
    """Convolutional surrogate with an explicit, bounded receptive field.

    Parameters
    ----------
    dilations : tuple[int, ...]
        Dilation per convolution layer; each layer has kernel size 9, so the
        receptive field is ``9 + 8 * sum(dilations)`` base pairs.
    channels : int, optional, default=64
        Hidden width.
    in_channels : int, optional, default=6
        Input channels (one-hot bases, landscape, inverse temperature).
    """

    def __init__(
        self,
        dilations: tuple[int, ...],
        channels: int = 64,
        in_channels: int = 6,
    ) -> None:
        super().__init__()
        self.dilations = tuple(dilations)
        layers: list[torch.nn.Module] = []
        width = in_channels
        for dilation in self.dilations:
            layers.append(
                torch.nn.Conv1d(
                    width,
                    channels,
                    kernel_size=9,
                    padding=4 * dilation,
                    dilation=dilation,
                )
            )
            layers.append(torch.nn.GELU())
            width = channels
        layers.append(torch.nn.Conv1d(channels, 1, kernel_size=1))
        self.net = torch.nn.Sequential(*layers)

    @property
    def receptive_field(self) -> int:
        """Receptive field in base pairs."""
        return 9 + 8 * sum(self.dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(n, 6, L)`` input -> ``(n, L)`` occupancy in ``[0, 1]``."""
        return torch.sigmoid(self.net(x)).squeeze(1)


def _resolve_device(requested: str) -> torch.device:
    """Map a device string to an available torch device."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _train(model, cfg: dict, table, device, steps: int) -> dict:
    """Train one surrogate on the checkpoint's corpus and report val errors."""
    train_split = data_mod.build_split(
        int(cfg["n_train"]),
        int(cfg["seq_length"]),
        int(cfg["seed"]),
        table,
        gc_low=float(cfg["gc_low"]),
        gc_high=float(cfg["gc_high"]),
        planted_fraction=float(cfg["planted_fraction"]),
        gamma_low=float(cfg["gamma_low"]),
        gamma_high=float(cfg["gamma_high"]),
    )
    val_split = data_mod.build_split(
        int(cfg["n_val"]),
        int(cfg["seq_length"]),
        int(cfg["seed"]) + int(cfg["val_seed_offset"]),
        table,
        gc_low=float(cfg["gc_low"]),
        gc_high=float(cfg["gc_high"]),
        planted_fraction=float(cfg["planted_fraction"]),
        gamma_low=float(cfg["gamma_low"]),
        gamma_high=float(cfg["gamma_high"]),
    )
    norm = cfg["normalization"]
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    loss_fn = model_mod.make_loss(str(cfg["loss"]))
    order_gen = torch.Generator().manual_seed(int(cfg["seed"]) + 7)
    batch_size = int(cfg["batch_size"])
    for _ in range(steps):
        index = torch.randint(
            0, int(cfg["n_train"]), (batch_size,), generator=order_gen
        )
        x, y = model_mod.sample_batch(train_split, index, norm, device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    return _metrics(model, val_split, norm, device)


def _metrics(model, split: dict, norm: dict, device) -> dict:
    """Relative-L2 statistics over a split."""
    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, split["seq"].shape[0], 256):
            stop = min(start + 256, split["seq"].shape[0])
            x, y = model_mod.sample_batch(
                split, torch.arange(start, stop), norm, device
            )
            errors.append(
                data_mod.relative_l2(model(x).to(_DTYPE).cpu(), y.to(_DTYPE).cpu())
            )
    model.train()
    err = torch.cat(errors)
    return {
        "median": float(err.median()),
        "p95": float(torch.quantile(err, 0.95)),
        "max": float(err.max()),
        "mean": float(err.mean()),
    }


def main() -> int:
    """Train the restricted surrogates and print the ablation table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", default=None, help="FNO checkpoint")
    parser.add_argument("--steps", type=int, default=None, help="override step budget")
    parser.add_argument("--device", default=None, help="override device")
    parser.add_argument(
        "--out", default=None, help="JSON report path (default outputs/...)"
    )
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if args.checkpoint is None:
        candidates = glob.glob(
            os.path.join(here, "outputs", "**", "model.pt"), recursive=True
        )
        if not candidates:
            print("no checkpoint found under outputs/**/model.pt")
            return 2
        args.checkpoint = max(candidates, key=os.path.getmtime)
    ckpt = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    cfg = dict(ckpt["cfg"])
    if args.device:
        cfg["device"] = args.device
    steps = int(args.steps if args.steps is not None else cfg["steps"])
    device = _resolve_device(cfg["device"])
    table = energy_mod.make_energy_table(
        seed=int(cfg["table_seed"]),
        width=int(cfg["width"]),
        scale=float(cfg["energy_scale"]),
    )
    norm = cfg["normalization"]
    test_split = data_mod.build_split(
        int(cfg["n_test"]),
        int(cfg["seq_length"]),
        int(cfg["test_seed"]),
        table,
        gc_low=float(cfg["gc_low"]),
        gc_high=float(cfg["gc_high"]),
        planted_fraction=float(cfg["planted_fraction"]),
        gamma_low=float(cfg["gamma_low"]),
        gamma_high=float(cfg["gamma_high"]),
    )

    fno = model_mod.NucleosomeOperator(
        latent_channels=int(cfg["latent_channels"]),
        num_fno_layers=int(cfg["num_fno_layers"]),
        num_fno_modes=int(cfg["num_fno_modes"]),
        padding=int(cfg["padding"]),
    )
    fno.load_state_dict(ckpt["state_dict"])
    fno = fno.to(device)
    results = {
        "fno (spectral, full locus)": {
            "params": sum(p.numel() for p in fno.parameters()),
            "receptive_field": int(cfg["seq_length"]),
            **_metrics(fno, test_split, norm, device),
        }
    }
    print(
        f"FNO baseline: median {results['fno (spectral, full locus)']['median']:.4%}  "
        f"P95 {results['fno (spectral, full locus)']['p95']:.4%}  "
        f"({results['fno (spectral, full locus)']['params']} params)"
    )

    for dilations in ((1, 1, 1, 1), (1, 2, 4, 8)):
        conv = LocalConvOperator(dilations)
        rf = conv.receptive_field
        label = f"conv RF {rf} bp"
        results[label] = {"params": sum(p.numel() for p in conv.parameters())}
        results[label]["receptive_field"] = rf
        results[label].update(_train(conv, cfg, table, device, steps))
        print(
            f"{label}: median {results[label]['median']:.4%}  "
            f"P95 {results[label]['p95']:.4%}  ({results[label]['params']} params)"
        )

    print("=" * 72)
    print(f"Receptive-field ablation at equal budget ({steps} steps)")
    print("=" * 72)
    print(f"{'surrogate':<28}{'field (bp)':>11}{'params':>10}{'median':>10}{'P95':>10}")
    for label, entry in results.items():
        print(
            f"{label:<28}{entry['receptive_field']:>11}{entry['params']:>10}"
            f"{entry['median']:>9.2%}{entry['p95']:>10.2%}"
        )
    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "steps": steps,
        "results": results,
    }
    out_path = args.out or os.path.join(
        here, "outputs", "receptive_field_ablation.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

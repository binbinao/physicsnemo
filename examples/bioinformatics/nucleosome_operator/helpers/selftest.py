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

"""Startup gate: reference-solver and corpus self-tests.

``train.py`` runs these before the first optimizer step and aborts on any
failure, so a silently wrong label generator can never be trained against:

1. **Brute force equivalence** -- the transfer-matrix occupancy must match
   exhaustive enumeration over every admissible footprint configuration on a
   tiny locus, for several inverse temperatures.
2. **Exact combinatorics at gamma = 0** -- with all Boltzmann factors equal,
   the partition function must equal the exact big-integer tiling count and
   the occupancy the exact coverage-pair count.
3. **Energy landscape** -- the sliding window energies must match a naive
   independent window sum, and the table columns must be centred over the 16
   dinucleotide classes.
4. **Limits** -- a strongly favoured landscape drives occupancy to 1, a
   strongly penalised one to 0, and occupancy never leaves ``[0, 1]``.
5. **Monotone sequence dependence** -- raising gamma on a fixed corpus must
   increase the profile contrast (the physics the trend metric checks).
6. **Corpus contract** -- the same seed reproduces an identical split, both
   strata are present, and the reference solver is deterministic.
"""

import math

import torch

from helpers import data as data_mod
from helpers import energy as energy_mod
from helpers import genome, nucleosome

_DTYPE = torch.float64
_TOL_BRUTE_FORCE = 1e-10
_TOL_EXACT = 1e-9


class SelfTestFailure(RuntimeError):
    """Raised when a reference-solver or corpus self-test fails."""


def run_self_tests(seed: int = 95051, verbose: bool = True) -> dict:
    """Run every self-test; raise :class:`SelfTestFailure` on the first failure.

    Parameters
    ----------
    seed : int
        Seed for the random landscapes used by the checks.
    verbose : bool
        Print one line per check.

    Returns
    -------
    dict
        Measured diagnostics (max deviations, limit occupancies, contrasts).
    """

    def report(name: str, detail: str) -> None:
        if verbose:
            print(f"[selftest] {name}: {detail}")

    out: dict = {}

    # 1. Brute force equivalence on a tiny locus.
    length, width = 22, 5
    gen = torch.Generator().manual_seed(seed)
    gammas = torch.tensor([0.7, 1.5, 2.5], dtype=_DTYPE)
    energies = torch.randn(3, length - width + 1, generator=gen, dtype=_DTYPE)
    occ, log_z = nucleosome.reference_profiles(gammas, energies, length, width)
    worst = 0.0
    for b in range(3):
        z_bf, occ_bf = nucleosome.brute_force_profiles(
            float(gammas[b]), energies[b], length, width
        )
        worst = max(worst, float((occ[b] - occ_bf).abs().max()))
        if abs(float(log_z[b]) - math.log(z_bf)) > _TOL_BRUTE_FORCE:
            raise SelfTestFailure(f"log Z mismatch against brute force (case {b})")
    if worst > _TOL_BRUTE_FORCE:
        raise SelfTestFailure(f"brute-force occupancy deviation {worst:.3e}")
    out["brute_force_max_dev"] = worst
    report(
        "brute force", f"max occupancy deviation {worst:.2e} (L={length}, W={width})"
    )

    # 2. Exact combinatorics at gamma = 0.
    length, width = 60, 7
    zeros = torch.zeros(1, length - width + 1, dtype=_DTYPE)
    occ0, logz0 = nucleosome.reference_profiles(
        torch.zeros(1, dtype=_DTYPE), zeros, length, width
    )
    tilings = nucleosome.exact_tiling_counts(length, width)
    counts = nucleosome.exact_coverage_counts(length, width)
    exact_occ = torch.tensor(counts, dtype=_DTYPE) / float(tilings[length])
    z_rel = abs(float(torch.exp(logz0[0])) - tilings[length]) / tilings[length]
    occ_dev = float((occ0[0] - exact_occ).abs().max())
    if z_rel > _TOL_EXACT or occ_dev > _TOL_EXACT:
        raise SelfTestFailure(
            f"gamma=0 exact check failed (Z rel err {z_rel:.3e}, occ {occ_dev:.3e})"
        )
    out["gamma0_z_rel_err"] = z_rel
    out["gamma0_occ_max_dev"] = occ_dev
    report(
        "gamma=0 exact",
        f"Z={tilings[length]} rel err {z_rel:.2e}, occupancy dev {occ_dev:.2e}",
    )

    # 3. Energy landscape: table centring + naive window sum.
    table = energy_mod.make_energy_table()
    col_mean = float(table.mean(dim=0).abs().max())
    if col_mean > 1e-12:
        raise SelfTestFailure(f"energy table columns not centred ({col_mean:.3e})")
    seq = genome.sample_sequences(4, 400, torch.Generator().manual_seed(seed))
    fast = energy_mod.sliding_energies(seq, table)
    slow = _naive_sliding_energies(seq, table)
    dev = float((fast - slow).abs().max())
    if dev > 1e-10:
        raise SelfTestFailure(f"sliding energy deviation {dev:.3e}")
    out["energy_max_dev"] = dev
    out["table_col_mean"] = col_mean
    report("energy landscape", f"naive window-sum deviation {dev:.2e}")

    # 4. Limits.
    length, width = 256, 8
    gamma = torch.tensor([3.0], dtype=_DTYPE)
    hot, _ = nucleosome.reference_profiles(
        gamma, torch.full((1, length - width + 1), -5.0, dtype=_DTYPE), length, width
    )
    cold, _ = nucleosome.reference_profiles(
        gamma, torch.full((1, length - width + 1), 5.0, dtype=_DTYPE), length, width
    )
    if float(hot.mean()) < 0.9 or float(cold.mean()) > 0.05:
        raise SelfTestFailure(
            f"limit check failed (favoured {float(hot.mean()):.4f}, "
            f"penalised {float(cold.mean()):.4f})"
        )
    for name, prof in (("favoured", hot), ("penalised", cold)):
        if float(prof.min()) < -1e-12 or float(prof.max()) > 1 + 1e-12:
            raise SelfTestFailure(f"{name} occupancy outside [0, 1]")
    out["favoured_mean_occ"] = float(hot.mean())
    out["penalised_mean_occ"] = float(cold.mean())
    report(
        "limits",
        f"favoured mean occupancy {float(hot.mean()):.4f}, "
        f"penalised {float(cold.mean()):.2e}",
    )

    # 5. Monotone sequence dependence (contrast grows with gamma).
    corpus = data_mod.build_split(64, 1024, seed + 1, table)
    contrasts = []
    for gamma_value in (0.5, 1.0, 2.0):
        g = torch.full((64,), gamma_value, dtype=_DTYPE)
        occ, _ = nucleosome.reference_profiles(
            g, energy_mod.sliding_energies(corpus["seq"], table), length=1024
        )
        contrasts.append(float(occ.std(dim=1).mean()))
    if not (contrasts[0] < contrasts[1] < contrasts[2]):
        raise SelfTestFailure(
            f"profile contrast not increasing with gamma: {contrasts}"
        )
    out["contrast_by_gamma"] = contrasts
    report(
        "gamma contrast",
        " -> ".join(f"{c:.4f}" for c in contrasts) + " (gamma 0.5/1/2)",
    )

    # 6. Corpus contract: reproducibility, strata, label determinism.
    a = data_mod.build_split(64, 512, seed + 2, table)
    b = data_mod.build_split(64, 512, seed + 2, table)
    if not torch.equal(a["seq"], b["seq"]) or not torch.equal(
        a["occupancy"], b["occupancy"]
    ):
        raise SelfTestFailure("same-seed splits are not identical")
    if int(a["stratum"].sum()) == 0 or int(a["stratum"].sum()) == a["stratum"].numel():
        raise SelfTestFailure("corpus is missing a stratum")
    occ_min = float(a["occupancy"].min())
    occ_max = float(a["occupancy"].max())
    if occ_min < 0.0 or occ_max > 1.0:
        raise SelfTestFailure(
            f"corpus occupancy outside [0, 1]: [{occ_min}, {occ_max}]"
        )
    out["corpus_mean_occupancy"] = float(a["occupancy"].to(_DTYPE).mean())
    out["planted_rows"] = int(a["stratum"].sum())
    report(
        "corpus",
        f"mean occupancy {out['corpus_mean_occupancy']:.4f}, "
        f"{out['planted_rows']}/{a['stratum'].numel()} planted rows",
    )
    return out


def _naive_sliding_energies(seq: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Independent O(n * starts * width) reference for the landscape sum."""
    width = table.shape[1] + 1
    codes = genome.dinucleotide_codes(seq)
    n, n_dinuc = codes.shape
    n_starts = n_dinuc - table.shape[1] + 1
    out = torch.zeros(n, n_starts, dtype=table.dtype)
    for row in range(n):
        for i in range(n_starts):
            total = 0.0
            for k in range(width - 1):
                total += float(table[int(codes[row, i + k]), k])
            out[row, i] = total
    return out

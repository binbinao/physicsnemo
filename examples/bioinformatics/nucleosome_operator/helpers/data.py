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

"""Corpus construction: DNA sequences, landscapes and exact occupancy labels.

A *split* is a plain dict of aligned tensors, all ``n`` sequences long:

===================  ==========================  ==============================
key                  shape / dtype               meaning
===================  ==========================  ==============================
``seq``              ``(n, L)`` int8             base codes (A=0, C=1, G=2, T=3)
``gamma``            ``(n,)`` float64            inverse-temperature scale
``energy``           ``(n, L)`` float32          footprint-start energy landscape,
                                                 zero padded to full length
``occupancy``        ``(n, L)`` float32          exact reference occupancy in [0, 1]
``stratum``          ``(n,)`` int8               0 = random, 1 = planted motif
===================  ==========================  ==============================

Labels come from :func:`helpers.nucleosome.reference_profiles`, the exact
transfer-matrix solver; the inverse temperature is drawn log-uniformly so a
single trained operator covers the weak- and strong-sequence-dependence
regimes.  The reference pass is chunked: the log recursions are sequential in
the locus coordinate, so chunking bounds peak memory without changing the
result (each sequence is solved independently).
"""

import math

import torch

from helpers import energy as energy_mod
from helpers import genome, nucleosome

_DTYPE = torch.float64
_CHUNK = 512

# Inverse-temperature (sequence-dependence) range; log-uniform draws.
GAMMA_LOW = 0.5
GAMMA_HIGH = 2.0


def build_split(
    n: int,
    length: int,
    seed: int,
    table: torch.Tensor,
    gc_low: float = 0.30,
    gc_high: float = 0.70,
    planted_fraction: float = 0.25,
    gamma_low: float = GAMMA_LOW,
    gamma_high: float = GAMMA_HIGH,
) -> dict:
    """Sample sequences and solve the exact reference occupancy for each.

    Parameters
    ----------
    n : int
        Number of sequences in the split.
    length : int
        Locus length ``L`` in bp.
    seed : int
        Owns the whole sampling stream: same seed -> identical split.
    table : torch.Tensor
        float64 ``(16, width-1)`` dinucleotide energy table.
    gc_low, gc_high : float
        Per-sequence GC content range.
    planted_fraction : float
        Fraction of sequences carrying a planted regulatory-landscape motif.
    gamma_low, gamma_high : float
        Log-uniform range of the inverse temperature.

    Returns
    -------
    dict
        Split tensors as described in the module docstring.
    """
    gen = torch.Generator().manual_seed(int(seed))
    seq = genome.sample_sequences(
        n,
        length,
        gen,
        gc_low=gc_low,
        gc_high=gc_high,
        planted_fraction=planted_fraction,
    )
    gamma = torch.exp(
        torch.empty(n, dtype=_DTYPE).uniform_(
            math.log(gamma_low), math.log(gamma_high), generator=gen
        )
    )
    energies = energy_mod.sliding_energies(seq, table)
    occupancy = _reference_occupancy(gamma, energies, length)
    n_planted = int(round(n * planted_fraction))
    stratum = torch.zeros(n, dtype=torch.int8)
    if n_planted > 0:
        stride = n / n_planted
        stratum[torch.tensor([int(i * stride) for i in range(n_planted)])] = 1
    return {
        "seq": seq,
        "gamma": gamma,
        "energy": energy_mod.energy_channel(energies, length).to(torch.float32),
        "occupancy": occupancy.to(torch.float32),
        "stratum": stratum,
    }


def _reference_occupancy(
    gamma: torch.Tensor, energies: torch.Tensor, length: int
) -> torch.Tensor:
    """Exact occupancy for a batch, chunked over sequences."""
    out = torch.empty(gamma.shape[0], length, dtype=_DTYPE)
    for start in range(0, gamma.shape[0], _CHUNK):
        stop = min(start + _CHUNK, gamma.shape[0])
        occ, _ = nucleosome.reference_profiles(
            gamma[start:stop], energies[start:stop], length=length
        )
        out[start:stop] = occ
    return out


def normalization_stats(split: dict) -> dict:
    """Per-channel input standardization constants, measured on one split.

    The landscape channel carries the **Boltzmann exponent** ``-gamma * E``
    rather than the bare energy: the occupancy depends on the landscape only
    through the footprint weights ``exp(-gamma E(i))``, so ``-gamma E`` is a
    sufficient statistic for the whole map and the network does not have to
    learn the product ``gamma * E`` itself.  ``gamma`` is carried as its own
    channel too, so the operator still sees the inverse temperature that
    parameterizes the sweep used by the physics-consistency check.

    Returns
    -------
    dict
        ``exponent_mean``, ``exponent_std``, ``gamma_mean``, ``gamma_std`` as
        python floats; stored in the checkpoint so evaluation reproduces the
        exact input scaling training used.
    """
    exponent = -(split["gamma"].to(_DTYPE)[:, None] * split["energy"].to(_DTYPE))
    gamma = split["gamma"].to(_DTYPE)
    return {
        "exponent_mean": float(exponent.mean()),
        "exponent_std": float(exponent.std()),
        "gamma_mean": float(gamma.mean()),
        "gamma_std": float(gamma.std()),
    }


def encode_input(
    seq: torch.Tensor,
    gamma: torch.Tensor,
    energy: torch.Tensor,
    norm: dict,
) -> torch.Tensor:
    """Operator input tensor ``(n, 6, L)`` float32.

    Channels are ``[one-hot A/C/G/T (4), standardized Boltzmann exponent (1),
    standardized inverse temperature (1)]``.  The raw sequence is carried
    alongside the exponent because the windowed sum that produced ``E`` is not
    invertible position by position, so the network gets both the sequence and
    its sequence-derived landscape.
    """
    length = seq.shape[1]
    bases = genome.one_hot(seq)  # (n, 4, L)
    exponent = -gamma.to(_DTYPE)[:, None] * energy.to(_DTYPE)
    e = (exponent.to(torch.float32) - float(norm["exponent_mean"])) / float(
        norm["exponent_std"]
    )
    g = (gamma.to(torch.float32) - float(norm["gamma_mean"])) / float(norm["gamma_std"])
    return torch.cat(
        [bases, e.unsqueeze(1), g.reshape(-1, 1, 1).expand(-1, 1, length)], dim=1
    )


def subset(split: dict, index: torch.Tensor) -> dict:
    """Rows of a split, for stratified evaluation."""
    return {key: value[index] for key, value in split.items()}


def relative_l2(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-sequence relative L2 error ``||pred - ref|| / ||ref||``.

    Parameters
    ----------
    pred, ref : torch.Tensor
        ``(n, L)`` predicted and reference profiles.

    Returns
    -------
    torch.Tensor
        ``(n,)`` float64 relative errors.
    """
    pred = pred.to(_DTYPE)
    ref = ref.to(_DTYPE)
    return (pred - ref).norm(dim=-1) / ref.norm(dim=-1)

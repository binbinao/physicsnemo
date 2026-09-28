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

"""Synthetic DNA sequence sampling and encoding.

Bases are coded ``A=0, C=1, G=2, T=3`` (int8).  A corpus is sampled in two
strata so the operator sees both unpatterned and patterned DNA:

- ``random``: i.i.d. bases with a per-sequence GC content drawn uniformly
  from ``[gc_low, gc_high]`` (a coarse model of isochore GC variation).
- ``planted``: random background with one of two regulatory-landscape
  motifs written in, both of which drive nucleosome organisation in vivo:

  - an **A/T-tract array**: short poly(A)/poly(T) tracts (5-12 bp) spaced
    with a ~10.5 bp period, the in-phase signal that favours nucleosome
    formation and fixes rotational phasing (as in 5S rDNA / MMTV LTR);
  - a **GC-rich island**: a 40-220 bp block of elevated GC content, the
    signature of CpG islands, which are nucleosome-depleted in vivo.

Sequence-level determinism is exact: every sampler consumes a caller-owned
``torch.Generator``, so the same seed reproduces the same corpus on any
device.
"""

import torch

N_BASES = 4
BASES = ("A", "C", "G", "T")

# Base codes (int8).
A, C, G, T = 0, 1, 2, 3

_DTYPE = torch.float64


def sample_sequences(
    n: int,
    length: int,
    generator: torch.Generator,
    gc_low: float = 0.30,
    gc_high: float = 0.70,
    planted_fraction: float = 0.25,
) -> torch.Tensor:
    """Sample a corpus of DNA sequences -> int8 ``(n, length)`` base codes.

    Parameters
    ----------
    n : int
        Number of sequences.
    length : int
        Sequence length in base pairs.
    generator : torch.Generator
        Owns the sampling stream (determinism contract).
    gc_low, gc_high : float
        Range of the per-sequence GC content.
    planted_fraction : float
        Fraction of sequences that additionally carry a planted motif.

    Returns
    -------
    torch.Tensor
        ``(n, length)`` int8 base codes.
    """
    gc = torch.rand(n, 1, generator=generator, dtype=_DTYPE)
    gc = gc * (gc_high - gc_low) + gc_low
    u = torch.rand(n, length, generator=generator, dtype=_DTYPE)

    # Cumulative base probabilities: P(A)=P(T)=(1-gc)/2, P(C)=P(G)=gc/2.
    p_at = (1.0 - gc) / 2.0
    b1 = p_at
    b2 = p_at + gc / 2.0
    b3 = p_at + gc
    seq = (u >= b1).to(torch.int8) + (u >= b2).to(torch.int8) + (u >= b3).to(torch.int8)

    n_planted = int(round(n * planted_fraction))
    if n_planted > 0:
        # Plant on a strided subset so every minibatch sees planted cases.
        stride = n / n_planted
        idx = torch.tensor(
            [int(i * stride) for i in range(n_planted)], dtype=torch.long
        )
        seq = _plant_motifs(seq, idx, length, generator)
    return seq


def _plant_motifs(
    seq: torch.Tensor,
    idx: torch.Tensor,
    length: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Write one A/T-tract array or GC-rich island into each indexed row."""
    for row in idx.tolist():
        if bool(torch.rand(1, generator=generator) < 0.5):
            _plant_tract_array(seq, row, length, generator)
        else:
            _plant_gc_island(seq, row, length, generator)
    return seq


def _plant_tract_array(
    seq: torch.Tensor, row: int, length: int, generator: torch.Generator
) -> None:
    """A/T tracts with a ~10.5 bp period over a 60-200 bp window."""
    span = 60 + int(torch.randint(0, 141, (1,), generator=generator))
    span = min(span, length)
    start = int(torch.randint(0, length - span + 1, (1,), generator=generator))
    tract = 5 + int(torch.randint(0, 8, (1,), generator=generator))  # 5..12
    period = 10.5
    pos = float(start)
    while pos + tract < start + span:
        i = int(round(pos))
        base = A if bool(torch.rand(1, generator=generator) < 0.5) else T
        seq[row, i : i + tract] = base
        pos += period
    # GC-biased filler between the tracts (the tract array sits in a GC-rich
    # context in the sequences that position nucleosomes in vivo).
    region = seq[row, start : start + span]
    filler = torch.randint(0, 2, region.shape, generator=generator).to(torch.int8) + C
    is_tract = (region == A) | (region == T)
    region[~is_tract] = filler[~is_tract]


def _plant_gc_island(
    seq: torch.Tensor, row: int, length: int, generator: torch.Generator
) -> None:
    """GC-rich 40-220 bp block (CpG-island-like, nucleosome-depleted)."""
    span = 40 + int(torch.randint(0, 181, (1,), generator=generator))
    span = min(span, length)
    start = int(torch.randint(0, length - span + 1, (1,), generator=generator))
    u = torch.rand(span, generator=generator, dtype=_DTYPE)
    gc = 0.75 + 0.15 * torch.rand(1, generator=generator, dtype=_DTYPE).item()
    block = torch.where(u < gc / 2, C, torch.where(u < gc, G, T))
    seq[row, start : start + span] = block.to(torch.int8)


def dinucleotide_codes(seq: torch.Tensor) -> torch.Tensor:
    """Adjacent-base codes -> int64 ``(n, length - 1)`` in ``[0, 16)``.

    Code ``4 * b_i + b_{i+1}`` orders the 16 dinucleotides as
    ``AA, AC, AG, AT, CA, ...``.
    """
    left = seq[:, :-1].to(torch.int64)
    right = seq[:, 1:].to(torch.int64)
    return 4 * left + right


def one_hot(seq: torch.Tensor) -> torch.Tensor:
    """int8 base codes -> float32 ``(n, 4, length)`` one-hot encoding."""
    return (
        torch.nn.functional.one_hot(seq.to(torch.int64), N_BASES)
        .permute(0, 2, 1)
        .to(torch.float32)
    )

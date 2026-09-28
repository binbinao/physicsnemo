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

"""Position-specific dinucleotide energy model (the sequence -> landscape map).

Following the thermodynamic description of sequence-encoded nucleosome
organisation (Kaplan et al., Nature 2009), a nucleosome whose footprint starts
at position ``i`` is assigned the additive energy

    E(i) = sum_{k=0}^{W-2} h[k, d(i + k)],

where ``d(j)`` is the dinucleotide class at ``j`` (16 classes, see
``genome.dinucleotide_codes``) and ``h`` is a position-specific
``(16, W-1)`` energy table: position ``k`` inside the footprint is one
helical turn away from position ``k + 10``, so a periodic table lets the model
express the rotational phasing that real nucleosomes have with respect to the
DNA helix.  The shipped table is a frozen, seeded random draw, not a fit to
experimental data; the example learns the *operator* from this landscape to
occupancy, so the table only has to be a plausible, non-degenerate linear
functional of the sequence.

Two construction details matter for the corpus statistics:

- **Per-position class mean removal.**  Each column of ``h`` is centred over
  the 16 dinucleotide classes, so ``E`` has zero mean under i.i.d. uniform
  DNA: ``gamma`` then tunes the *sequence dependence* of occupancy rather
  than a global packing bias.
- **Energy scale.**  Entries are drawn with standard deviation ``scale``;
  a footprint of ``W-1`` terms then has energy standard deviation
  ``scale * sqrt(W-1)``.  ``scale`` is the one model constant chosen for the
  corpus (see ``README.md``): it fixes how strongly sequence can move
  occupancy, and is calibrated once so the corpus spans a responsive
  occupancy range instead of saturating at 0/1.
"""

import torch

WIDTH = 147  # nucleosome footprint in bp
ENERGY_SCALE = 0.22  # per-entry std of the dinucleotide table
TABLE_SEED = 20260920

_DTYPE = torch.float64


def make_energy_table(
    seed: int = TABLE_SEED,
    width: int = WIDTH,
    scale: float = ENERGY_SCALE,
) -> torch.Tensor:
    """Frozen position-specific dinucleotide energy table ``(16, width-1)``.

    Parameters
    ----------
    seed : int
        Table draw seed (part of the checkpoint contract).
    width : int
        Footprint length in bp; the table spans the ``width - 1``
        dinucleotides of a footprint.
    scale : float
        Standard deviation of the raw entries before centring.

    Returns
    -------
    torch.Tensor
        float64 ``(16, width-1)`` table, each column centred over the 16
        dinucleotide classes.
    """
    gen = torch.Generator().manual_seed(seed)
    table = torch.randn(16, width - 1, generator=gen, dtype=_DTYPE) * scale
    return table - table.mean(dim=0, keepdim=True)


def sliding_energies(seq: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Energy ``E(i)`` of a footprint starting at every admissible ``i``.

    Parameters
    ----------
    seq : torch.Tensor
        int8 ``(n, length)`` base codes.
    table : torch.Tensor
        float64 ``(16, width-1)`` table from :func:`make_energy_table`.

    Returns
    -------
    torch.Tensor
        float64 ``(n, length - width + 1)`` energies: entry ``i`` is the
        energy of the footprint covering ``[i, i + width)``.

    Notes
    -----
    The sum is accumulated term by term (one gather per offset ``k``)
    rather than through ``conv1d``: a ``(16, width-1)`` kernel expands the
    input ``~2300x`` in the conv im2col buffer, which exceeds host memory
    for a full corpus, while the offset loop touches ``O(n * length)``
    memory in total.
    """
    n_terms = table.shape[1]
    codes = _dinucleotide_codes(seq)
    n, n_dinuc = codes.shape
    n_starts = n_dinuc - n_terms + 1
    if n_starts < 1:
        raise ValueError(
            f"sequence length {seq.shape[1]} is shorter than the footprint "
            f"width {n_terms + 1}"
        )
    out = torch.zeros(n, n_starts, dtype=table.dtype, device=seq.device)
    for k in range(n_terms):
        # table[:, k][codes[:, i + k]] for every footprint start i
        out += table[:, k][codes[:, k : k + n_starts]]
    return out


def _dinucleotide_codes(seq: torch.Tensor) -> torch.Tensor:
    """int8 ``(n, length)`` bases -> int64 ``(n, length - 1)`` codes."""
    from helpers.genome import dinucleotide_codes

    return dinucleotide_codes(seq)


def energy_channel(energies: torch.Tensor, length: int) -> torch.Tensor:
    """Pad the start-indexed energy landscape to a full-length channel.

    ``E`` is only defined for the ``length - width + 1`` footprint start
    positions; the trailing ``width - 1`` positions cannot start a footprint
    and are padded with zero (which is the corpus mean, since the table
    columns are centred).

    Parameters
    ----------
    energies : torch.Tensor
        ``(n, length - width + 1)`` energies.
    length : int
        Target sequence length.

    Returns
    -------
    torch.Tensor
        ``(n, length)`` float64 landscape, zero padded on the right.
    """
    n, m = energies.shape
    if m > length:
        raise ValueError(f"energy profile ({m}) longer than the sequence ({length})")
    out = torch.zeros(n, length, dtype=energies.dtype, device=energies.device)
    out[:, :m] = energies
    return out

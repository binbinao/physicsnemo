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

r"""Exact thermodynamic reference solver: nucleosome occupancy from sequence.

Model
-----
A locus of ``L`` bp is partitioned into linkers (1 bp) and nucleosomes of
fixed footprint ``W`` bp.  A *configuration* is a set of non-overlapping
footprint start positions ``S subseteq {0, ..., L - W}``; its statistical
weight at inverse temperature ``gamma`` is

    w(S) = prod_{i in S} exp(-gamma * E(i)),

the empty configuration (bare DNA) having weight 1.  The grand partition
function and the per-position occupancy are then

    Z = sum_S w(S),
    P(j covered) = (1 / Z) sum_{S : j covered by S} w(S).

Both are computed exactly by the transfer-matrix (dynamic program)
recursions

    f[j]   = f[j-1] + f[j-W] * exp(-gamma E(j-W))        (prefix [0, j))
    b[j]   = b[j+1] + exp(-gamma E(j)) * b[j+W]          (suffix [j, L))

and the inside-outside identity

    P(j covered) = (1 / Z) sum_{i = j-W+1}^{j} f[i] exp(-gamma E(i)) b[i+W],

where the sum runs over the footprint starts whose span contains ``j``
(clipped to ``[0, L-W]``).  ``f[L] = b[0] = Z``.

Numerics
--------
``f`` and ``b`` are products of up to ``L / W`` Boltzmann factors and
overflow float64 for long loci, so the whole recursion runs in log space
with ``logaddexp``; the occupancy ratio is formed after subtracting
``log Z``.  The result is exact up to float64 round-off, which the
self-tests pin against brute-force enumeration (``brute_force_profiles``)
and against exact big-integer combinatorics (``exact_tiling_counts``).

This module is the *label generator*: it is what a Ribo-seq / MNase-seq /
ATAC-seq profile would measure if the frozen energy table were the true
sequence code, and it is deliberately free of fitted parameters.
"""

import torch

from helpers.energy import WIDTH

_DTYPE = torch.float64
_NEG_INF = float("-inf")


def forward_log_partition(logw: torch.Tensor, length: int, width: int) -> torch.Tensor:
    """log ``f[j]`` = log partition function of the prefix ``[0, j)``.

    Parameters
    ----------
    logw : torch.Tensor
        float64 ``(n, length - width + 1)`` log Boltzmann factors of a
        footprint at each start position.
    length : int
        Locus length ``L``.
    width : int
        Footprint length ``W``.

    Returns
    -------
    torch.Tensor
        float64 ``(n, length + 1)`` log prefix partition functions;
        entry ``length`` is ``log Z``.
    """
    n = logw.shape[0]
    logf = torch.full((n, length + 1), _NEG_INF, dtype=logw.dtype, device=logw.device)
    logf[:, 0] = 0.0
    for j in range(1, length + 1):
        # Position j-1 is bare linker...
        acc = logf[:, j - 1]
        if j >= width:
            # ...or the last footprint of the prefix starts at j-W.
            acc = torch.logaddexp(acc, logf[:, j - width] + logw[:, j - width])
        logf[:, j] = acc
    return logf


def backward_log_partition(logw: torch.Tensor, length: int, width: int) -> torch.Tensor:
    """log ``b[j]`` = log partition function of the suffix ``[j, L)``.

    Parameters
    ----------
    logw : torch.Tensor
        float64 ``(n, length - width + 1)`` log Boltzmann factors.
    length : int
        Locus length ``L``.
    width : int
        Footprint length ``W``.

    Returns
    -------
    torch.Tensor
        float64 ``(n, length + 1)`` log suffix partition functions;
        entry ``0`` is ``log Z``.
    """
    n = logw.shape[0]
    logb = torch.full((n, length + 1), _NEG_INF, dtype=logw.dtype, device=logw.device)
    logb[:, length] = 0.0
    for j in range(length - 1, -1, -1):
        acc = logb[:, j + 1]  # position j is beyond the last start: linker
        if j <= length - width:
            acc = torch.logaddexp(acc, logw[:, j] + logb[:, j + width])
        logb[:, j] = acc
    return logb


def coverage_profile(
    logf: torch.Tensor,
    logb: torch.Tensor,
    logw: torch.Tensor,
    length: int,
    width: int,
) -> torch.Tensor:
    """Occupancy ``P(j covered)`` for every position, from the log recursions.

    Sums (in log space) the outside-weighted Boltzmann factor of every
    start whose footprint contains ``j``; for each offset ``k = j - i`` the
    valid positions form a contiguous slice, so the window sum costs
    ``O(W * L)`` with no ``(n, L, W)`` intermediate.

    Parameters
    ----------
    logf, logb : torch.Tensor
        float64 ``(n, length + 1)`` prefix / suffix log partition functions.
    logw : torch.Tensor
        float64 ``(n, length - width + 1)`` log Boltzmann factors.
    length : int
        Locus length ``L``.
    width : int
        Footprint length ``W``.

    Returns
    -------
    torch.Tensor
        float64 ``(n, length)`` occupancy profiles in ``[0, 1]``.
    """
    n_starts = logw.shape[1]
    log_u = logf[:, :n_starts] + logw + logb[:, width:]  # log f[i] w_i b[i+W]
    acc = torch.full(
        (logf.shape[0], length), _NEG_INF, dtype=logf.dtype, device=logf.device
    )
    for k in range(width):
        # j = i + k; positions of the window that lie at offset k
        take = min(length - k, n_starts)
        if take <= 0:
            break
        acc[:, k : k + take] = torch.logaddexp(acc[:, k : k + take], log_u[:, :take])
    return (acc - logf[:, length:]).exp()


def reference_profiles(
    gamma: torch.Tensor,
    energies: torch.Tensor,
    length: int | None = None,
    width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact occupancy profiles and log partition functions for a batch.

    Parameters
    ----------
    gamma : torch.Tensor
        ``(n,)`` inverse temperature scaling the energy landscape.
    energies : torch.Tensor
        ``(n, length - width + 1)`` footprint-start energies.
    length : int, optional
        Locus length; inferred from ``energies`` when omitted (using
        ``width``, or the default footprint when that is omitted too).
    width : int, optional
        Footprint length ``W``; inferred from ``length - energies.shape[1]
        + 1`` when omitted, so a table of any width is handled consistently.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(occupancy (n, length), log_z (n,))``.
    """
    energies = energies.to(_DTYPE)
    gamma = gamma.to(_DTYPE)
    n_starts = energies.shape[1]
    if width is None:
        width = (length - n_starts + 1) if length is not None else WIDTH
    if length is None:
        length = n_starts + width - 1
    elif length != n_starts + width - 1:
        raise ValueError(
            f"inconsistent geometry: length {length}, {n_starts} footprint "
            f"starts and width {width} imply length {n_starts + width - 1}"
        )
    logw = -gamma[:, None] * energies
    logf = forward_log_partition(logw, length, width)
    logb = backward_log_partition(logw, length, width)
    return coverage_profile(logf, logb, logw, length, width), logf[:, length]


def brute_force_profiles(
    gamma: float, energies: torch.Tensor, length: int, width: int
) -> tuple[float, torch.Tensor]:
    """Exhaustive enumeration reference (tiny loci only).

    Walks the locus left to right, branching on "linker" and "footprint
    starts here", and accumulates the weight of every admissible
    configuration together with the coverage indicator per position.  The
    cost is the number of configurations, i.e. exponential in
    ``length / width``: this is a test oracle for :func:`reference_profiles`,
    not a production path.

    Parameters
    ----------
    gamma : float
        Inverse temperature.
    energies : torch.Tensor
        ``(length - width + 1,)`` footprint-start energies.
    length : int
        Locus length ``L``.
    width : int
        Footprint length ``W``.

    Returns
    -------
    tuple[float, torch.Tensor]
        ``(Z, occupancy (length,))`` from explicit enumeration.
    """
    e = [float(x) for x in energies.reshape(-1)]
    configs: list[tuple[float, tuple[int, ...]]] = []

    def walk(pos: int, weight: float, starts: tuple[int, ...]) -> None:
        if pos >= length:
            configs.append((weight, starts))
            return
        walk(pos + 1, weight, starts)  # bare linker at ``pos``
        if pos + width <= length:
            walk(
                pos + width,
                weight * pow(2.718281828459045, -gamma * e[pos]),
                starts + (pos,),
            )

    walk(0, 1.0, ())
    total = sum(w for w, _ in configs)
    covered = torch.zeros(length, dtype=_DTYPE)
    for weight, starts in configs:
        for start in starts:
            covered[start : start + width] += weight
    return total, covered / total


def exact_tiling_counts(length: int, width: int) -> list[int]:
    """Exact number of admissible tilings, big-integer arithmetic.

    ``T[j] = T[j-1] + T[j-W]`` counts linker/footprint tilings of a locus of
    ``j`` positions; at ``gamma = 0`` every configuration has weight 1, so
    ``Z`` must equal ``T[length]`` exactly -- an independent check of the
    transfer-matrix structure with no floating point involved.

    Parameters
    ----------
    length : int
        Locus length ``L``.
    width : int
        Footprint length ``W``.

    Returns
    -------
    list[int]
        ``T[0 .. length]``.
    """
    t = [0] * (length + 1)
    t[0] = 1
    for j in range(1, length + 1):
        t[j] = t[j - 1] + (t[j - width] if j >= width else 0)
    return t


def exact_coverage_counts(length: int, width: int) -> list[int]:
    """Exact ``(tiling, coverage)`` pair counts per position, big integers.

    ``C[j] = sum_i T[i] * T[L - i - W]`` over footprint starts ``i`` whose
    span contains ``j``; ``C[j] / T[L]`` is the occupancy at ``gamma = 0``,
    computed without any floating-point DP.
    """
    t = exact_tiling_counts(length, width)
    n_starts = length - width + 1
    counts = []
    for j in range(length):
        total = 0
        for i in range(max(0, j - width + 1), min(j, n_starts - 1) + 1):
            total += t[i] * t[length - i - width]
        counts.append(total)
    return counts

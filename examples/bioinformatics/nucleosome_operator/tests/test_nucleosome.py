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

"""Reference-solver tests: transfer-matrix DP against exact oracles.

Every test here pins the label generator against something that does not
share its code path: exhaustive enumeration, exact big-integer
combinatorics, or an analytically known limit.
"""

import math

import torch
from helpers import nucleosome

_DTYPE = torch.float64


def test_matches_brute_force_enumeration():
    """DP occupancy and Z equal exhaustive enumeration on a tiny locus."""
    length, width = 20, 5
    gen = torch.Generator().manual_seed(11)
    energies = torch.randn(4, length - width + 1, generator=gen, dtype=_DTYPE)
    gammas = torch.tensor([0.3, 1.0, 2.2, 4.0], dtype=_DTYPE)
    occ, log_z = nucleosome.reference_profiles(gammas, energies, length, width)
    for b in range(4):
        z_bf, occ_bf = nucleosome.brute_force_profiles(
            float(gammas[b]), energies[b], length, width
        )
        assert math.isclose(float(log_z[b]), math.log(z_bf), rel_tol=0, abs_tol=1e-12)
        assert torch.allclose(occ[b], occ_bf, atol=1e-12)


def test_gamma_zero_matches_exact_tiling_counts():
    """At gamma = 0 Z equals the big-integer tiling count exactly."""
    length, width = 80, 6
    zeros = torch.zeros(1, length - width + 1, dtype=_DTYPE)
    occ, log_z = nucleosome.reference_profiles(
        torch.zeros(1, dtype=_DTYPE), zeros, length, width
    )
    tilings = nucleosome.exact_tiling_counts(length, width)
    assert int(round(float(torch.exp(log_z[0])))) == tilings[length]
    counts = nucleosome.exact_coverage_counts(length, width)
    exact = torch.tensor(counts, dtype=_DTYPE) / float(tilings[length])
    assert torch.allclose(occ[0], exact, atol=1e-12)


def test_gamma_zero_coverage_fraction_matches_geometry():
    """Mean occupancy at gamma = 0 sits in the admissible geometric band."""
    length, width = 120, 7
    zeros = torch.zeros(1, length - width + 1, dtype=_DTYPE)
    occ, _ = nucleosome.reference_profiles(
        torch.zeros(1, dtype=_DTYPE), zeros, length, width
    )
    assert 0.0 < float(occ.mean()) < 1.0
    assert float(occ.min()) >= 0.0 and float(occ.max()) <= 1.0


def test_strong_attraction_fills_and_repulsion_empties():
    """Favourable landscapes drive occupancy to 1, unfavourable to 0."""
    length, width = 300, 10
    gamma = torch.tensor([4.0], dtype=_DTYPE)
    favoured, _ = nucleosome.reference_profiles(
        gamma, torch.full((1, length - width + 1), -4.0, dtype=_DTYPE), length, width
    )
    penalised, _ = nucleosome.reference_profiles(
        gamma, torch.full((1, length - width + 1), 4.0, dtype=_DTYPE), length, width
    )
    assert float(favoured.mean()) > 0.9
    assert float(penalised.mean()) < 0.01


def test_occupancy_is_monotone_in_energy_shift():
    """Shifting every landscape energy down cannot lower occupancy."""
    length, width = 200, 8
    gen = torch.Generator().manual_seed(3)
    base = torch.randn(1, length - width + 1, generator=gen, dtype=_DTYPE)
    gamma = torch.tensor([1.5], dtype=_DTYPE)
    means = []
    for shift in (1.0, 0.0, -1.0):
        occ, _ = nucleosome.reference_profiles(gamma, base + shift, length, width)
        means.append(float(occ.mean()))
    assert means[0] < means[1] < means[2]


def test_single_footprint_matches_the_analytic_two_state_model():
    """With one admissible start the locus is a two-state system.

    Either the single footprint is placed or the locus stays bare, so
    ``Z = 1 + exp(-gamma E)`` and the occupancy is the same constant at every
    position -- an analytically closed case the DP must reproduce.
    """
    length, width = 12, 12
    gamma, energy = 1.0, -2.0
    occ, log_z = nucleosome.reference_profiles(
        torch.tensor([gamma], dtype=_DTYPE),
        torch.full((1, 1), energy, dtype=_DTYPE),
        length,
        width,
    )
    expected = math.exp(-gamma * energy) / (1.0 + math.exp(-gamma * energy))
    assert math.isclose(
        float(log_z[0]), math.log(1.0 + math.exp(-gamma * energy)), abs_tol=1e-12
    )
    assert torch.allclose(
        occ[0], torch.full((length,), expected, dtype=_DTYPE), atol=1e-12
    )


def test_partition_recursions_agree_at_the_boundary():
    """Forward and backward log recursions both terminate at log Z."""
    length, width = 150, 9
    gen = torch.Generator().manual_seed(5)
    energies = torch.randn(3, length - width + 1, generator=gen, dtype=_DTYPE)
    logw = -1.3 * energies
    logf = nucleosome.forward_log_partition(logw, length, width)
    logb = nucleosome.backward_log_partition(logw, length, width)
    assert torch.allclose(logf[:, length], logb[:, 0], atol=1e-12)
    occ, log_z = nucleosome.reference_profiles(
        torch.full((3,), 1.3, dtype=_DTYPE), energies, length, width
    )
    assert torch.allclose(log_z, logf[:, length], atol=1e-12)
    assert bool((occ >= 0).all() and (occ <= 1).all())


def test_long_locus_does_not_overflow():
    """A 4096 bp locus stays finite in log space (float64 would overflow)."""
    length, width = 4096, 32
    gen = torch.Generator().manual_seed(9)
    energies = torch.randn(2, length - width + 1, generator=gen, dtype=_DTYPE) * 5.0
    occ, log_z = nucleosome.reference_profiles(
        torch.full((2,), 2.0, dtype=_DTYPE), energies, length, width
    )
    assert bool(torch.isfinite(occ).all())
    assert bool(torch.isfinite(log_z).all())
    # Mathematically in [0, 1]; float64 round-off may exceed the bound by ulps.
    assert float(occ.max()) <= 1.0 + 1e-9
    assert float(occ.min()) >= -1e-9

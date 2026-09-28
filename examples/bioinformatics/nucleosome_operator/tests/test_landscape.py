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

"""Energy-landscape and sequence-sampler tests."""

import torch
from helpers import energy, genome

_DTYPE = torch.float64


def _naive_energies(seq: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Straightforward nested-loop window sum, independent of the fast path."""
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


def test_energy_table_columns_are_centred():
    """Every table column is centred over the 16 dinucleotide classes."""
    table = energy.make_energy_table()
    assert table.shape == (16, energy.WIDTH - 1)
    assert float(table.mean(dim=0).abs().max()) < 1e-12


def test_energy_table_is_seeded_and_scaled():
    """The table is a deterministic function of (seed, width, scale)."""
    a = energy.make_energy_table(seed=1, width=32, scale=0.5)
    b = energy.make_energy_table(seed=1, width=32, scale=0.5)
    c = energy.make_energy_table(seed=2, width=32, scale=0.5)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert 0.3 < float(a.std()) < 0.7


def test_sliding_energies_match_naive_window_sum():
    """The vectorized landscape equals an independent nested-loop sum."""
    table = energy.make_energy_table(width=24)
    seq = genome.sample_sequences(3, 120, torch.Generator().manual_seed(4))
    fast = energy.sliding_energies(seq, table)
    assert torch.allclose(fast, _naive_energies(seq, table), atol=1e-10)
    assert fast.shape == (3, 120 - 24 + 1)


def test_sliding_energies_are_zero_mean_over_uniform_dna():
    """Under uniform i.i.d. DNA the landscape has no global packing bias.

    The table columns are centred over the 16 dinucleotide classes, so the
    expectation of ``E`` vanishes exactly when the classes are equiprobable.
    That holds for uniform i.i.d. DNA (GC fixed at 0.5) with no planted
    motifs; a GC-varying or motif-bearing corpus carries a composition-driven
    mean by construction, which is why the corpus mean is not asserted.
    """
    table = energy.make_energy_table(width=32)
    seq = genome.sample_sequences(
        600,
        256,
        torch.Generator().manual_seed(8),
        gc_low=0.5,
        gc_high=0.5,
        planted_fraction=0.0,
    )
    energies = energy.sliding_energies(seq, table)
    assert abs(float(energies.mean())) < 0.08


def test_energy_channel_pads_the_tail_with_zero():
    """The padded landscape keeps the start-indexed values and zero tail."""
    energies = torch.arange(6, dtype=_DTYPE).reshape(1, 6)
    channel = energy.energy_channel(energies, 10)
    assert channel.shape == (1, 10)
    assert torch.equal(channel[0, :6], energies[0])
    assert float(channel[0, 6:].abs().max()) == 0.0


def test_sequence_sampler_is_seeded_and_in_range():
    """Base codes stay in [0, 4) and the same seed reproduces the corpus."""
    a = genome.sample_sequences(64, 256, torch.Generator().manual_seed(7))
    b = genome.sample_sequences(64, 256, torch.Generator().manual_seed(7))
    c = genome.sample_sequences(64, 256, torch.Generator().manual_seed(70))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.dtype == torch.int8
    assert int(a.min()) >= 0 and int(a.max()) < genome.N_BASES


def test_planted_motifs_change_the_corpus():
    """Planted rows differ from the same draw without planting."""
    gen_plain = torch.Generator().manual_seed(21)
    plain = genome.sample_sequences(40, 512, gen_plain, planted_fraction=0.0)
    gen_planted = torch.Generator().manual_seed(21)
    planted = genome.sample_sequences(40, 512, gen_planted, planted_fraction=0.5)
    changed = (plain != planted).any(dim=1)
    assert int(changed.sum()) >= 15


def test_dinucleotide_codes_and_one_hot_are_consistent():
    """Codes index the 16 dinucleotides and one-hot matches the base codes."""
    seq = genome.sample_sequences(2, 32, torch.Generator().manual_seed(13))
    codes = genome.dinucleotide_codes(seq)
    assert codes.shape == (2, 31)
    assert int(codes.min()) >= 0 and int(codes.max()) < 16
    expected = 4 * seq[:, :-1].to(torch.int64) + seq[:, 1:].to(torch.int64)
    assert torch.equal(codes, expected)
    hot = genome.one_hot(seq)
    assert hot.shape == (2, 4, 32)
    assert torch.allclose(hot.sum(dim=1), torch.ones(2, 32))

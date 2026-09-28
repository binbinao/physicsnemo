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

"""Corpus-construction and metric tests."""

import torch
from helpers import data as data_mod
from helpers import energy

_DTYPE = torch.float64


def _small_split(n: int = 12, length: int = 256, seed: int = 5) -> dict:
    """Build a small split with a compact footprint for fast tests."""
    table = energy.make_energy_table(width=24)
    return data_mod.build_split(n, length, seed, table)


def test_split_contract():
    """Split shapes, dtypes and value ranges match the documented contract."""
    split = _small_split()
    assert split["seq"].shape == (12, 256)
    assert split["seq"].dtype == torch.int8
    assert split["gamma"].shape == (12,)
    assert split["gamma"].dtype == _DTYPE
    assert split["energy"].shape == (12, 256)
    assert split["occupancy"].shape == (12, 256)
    assert float(split["occupancy"].min()) >= 0.0
    assert float(split["occupancy"].max()) <= 1.0
    assert bool(torch.isfinite(split["occupancy"]).all())


def test_split_is_reproducible_and_seed_dependent():
    """The same seed rebuilds the split exactly; another seed does not."""
    a = _small_split(seed=3)
    b = _small_split(seed=3)
    c = _small_split(seed=4)
    assert torch.equal(a["seq"], b["seq"])
    assert torch.equal(a["occupancy"], b["occupancy"])
    assert not torch.equal(a["seq"], c["seq"])


def test_split_contains_both_strata():
    """The planted fraction is present and bounded by the corpus size."""
    split = _small_split(n=40)
    planted = int(split["stratum"].sum())
    assert 0 < planted < 40


def test_gamma_range_is_respected():
    """Inverse temperatures lie inside the configured log-uniform range."""
    split = _small_split(n=64)
    assert float(split["gamma"].min()) >= data_mod.GAMMA_LOW
    assert float(split["gamma"].max()) <= data_mod.GAMMA_HIGH


def test_occupancy_tracks_the_landscape():
    """Sequence-dependent landscapes produce sequence-dependent occupancy."""
    split = _small_split(n=24)
    spread = split["occupancy"].to(_DTYPE).std(dim=1)
    assert float(spread.mean()) > 0.02
    assert float(split["energy"].to(_DTYPE).std()) > 0.5


def test_normalization_stats_are_usable():
    """Stats are finite with a positive exponent scale."""
    split = _small_split(n=32)
    norm = data_mod.normalization_stats(split)
    assert set(norm) == {
        "exponent_mean",
        "exponent_std",
        "gamma_mean",
        "gamma_std",
    }
    assert norm["exponent_std"] > 0.0
    assert norm["gamma_std"] > 0.0
    assert abs(norm["exponent_mean"]) < 20.0


def test_encode_input_channels():
    """Channel 4 is the standardized exponent, channel 5 the standardized gamma."""
    split = _small_split(n=4)
    norm = {
        "exponent_mean": 1.5,
        "exponent_std": 3.0,
        "gamma_mean": 1.0,
        "gamma_std": 0.5,
    }
    x = data_mod.encode_input(split["seq"], split["gamma"], split["energy"], norm)
    assert x.shape == (4, 6, 256)
    assert x.dtype == torch.float32
    exponent = -split["gamma"].to(_DTYPE)[:, None] * split["energy"].to(_DTYPE)
    expected_e = (exponent.to(torch.float32) - 1.5) / 3.0
    assert torch.allclose(x[:, 4], expected_e, atol=1e-6)
    expected_g = (split["gamma"].to(torch.float32) - 1.0) / 0.5
    assert torch.allclose(x[:, 5], expected_g.reshape(-1, 1).expand(-1, 256), atol=1e-6)
    assert torch.allclose(x[:, :4].sum(dim=1), torch.ones(4, 256))


def test_relative_l2_metric():
    """Relative L2 is zero for identical profiles and scales with the error."""
    ref = torch.rand(3, 64, dtype=_DTYPE) + 0.5
    assert torch.allclose(
        data_mod.relative_l2(ref, ref), torch.zeros(3, dtype=_DTYPE), atol=1e-12
    )
    err = data_mod.relative_l2(ref * 1.1, ref)
    assert bool((err > 0).all())
    assert float(err.max()) < 0.2


def test_subset_selects_rows():
    """Subsetting a split keeps the rows aligned across keys."""
    split = _small_split(n=10)
    index = torch.tensor([0, 3, 7])
    sub = data_mod.subset(split, index)
    assert torch.equal(sub["seq"], split["seq"][index])
    assert torch.equal(sub["occupancy"], split["occupancy"][index])

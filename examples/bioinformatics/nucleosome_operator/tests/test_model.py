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

"""Operator-model and loss tests."""

import pytest
import torch
from helpers import data as data_mod
from helpers import energy
from helpers import model as model_mod

_DTYPE = torch.float64


def _tiny_operator() -> model_mod.NucleosomeOperator:
    """Operator with a small spectral budget, for fast structural tests."""
    return model_mod.NucleosomeOperator(
        latent_channels=8, num_fno_layers=1, num_fno_modes=8, padding=4
    )


def test_operator_output_shape_and_range():
    """The operator maps (n, 6, L) to a valid occupancy profile (n, L)."""
    model = _tiny_operator()
    x = torch.randn(4, 6, 256)
    y = model(x)
    assert y.shape == (4, 256)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
    assert bool(torch.isfinite(y).all())


def test_operator_evaluates_at_other_lengths():
    """The spectral backbone accepts loci longer than the training length."""
    model = _tiny_operator()
    for length in (128, 512, 1024):
        y = model(torch.randn(2, 6, length))
        assert y.shape == (2, length)


def test_relative_l2_loss_properties():
    """The loss is zero at equality and invariant to a global output scale."""
    ref = torch.rand(5, 64, dtype=torch.float32) + 0.5
    assert float(model_mod.relative_l2_loss(ref, ref)) == pytest.approx(0.0, abs=1e-12)
    scaled = model_mod.relative_l2_loss(ref * 2.0, ref)
    assert float(scaled) == pytest.approx(1.0, abs=1e-3)


def test_mse_loss_properties():
    """MSE is zero at equality and quadratic in the perturbation."""
    ref = torch.ones(2, 8)
    assert float(model_mod.mse_loss(ref, ref)) == pytest.approx(0.0, abs=1e-12)
    assert float(model_mod.mse_loss(ref + 0.1, ref)) == pytest.approx(0.01, abs=1e-6)


def test_make_loss_rejects_unknown_name():
    """An unknown loss name is a configuration error, not a silent default."""
    assert model_mod.make_loss("rel_l2") is model_mod.relative_l2_loss
    assert model_mod.make_loss("mse") is model_mod.mse_loss
    with pytest.raises(ValueError):
        model_mod.make_loss("l1")


def test_sample_batch_shapes():
    """Batches carry the encoded input and the occupancy target."""
    table = energy.make_energy_table(width=24)
    split = data_mod.build_split(6, 256, 17, table)
    norm = data_mod.normalization_stats(split)
    x, y = model_mod.sample_batch(
        split, torch.tensor([1, 3]), norm, torch.device("cpu")
    )
    assert x.shape == (2, 6, 256)
    assert y.shape == (2, 256)
    exponent = -split["gamma"][[1, 3]].to(torch.float64)[:, None] * split["energy"][
        [1, 3]
    ].to(torch.float64)
    expected = (exponent.to(torch.float32) - norm["exponent_mean"]) / norm[
        "exponent_std"
    ]
    assert torch.allclose(x[:, 4], expected, atol=1e-6)


def test_operator_trains_on_a_tiny_problem():
    """A few optimizer steps reduce the loss (the training path is wired)."""
    table = energy.make_energy_table(width=24)
    split = data_mod.build_split(24, 256, 19, table)
    norm = data_mod.normalization_stats(split)
    model = _tiny_operator()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    index = torch.arange(24)
    x, y = model_mod.sample_batch(split, index, norm, torch.device("cpu"))
    first = float(model_mod.relative_l2_loss(model(x), y).detach())
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = model_mod.relative_l2_loss(model(x), y)
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < first


def test_operator_output_is_differentiable():
    """Gradients reach every parameter through the logistic head."""
    model = _tiny_operator()
    x = torch.randn(2, 6, 128)
    loss = model(x).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)

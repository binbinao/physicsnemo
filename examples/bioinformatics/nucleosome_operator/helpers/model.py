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

"""Nucleosome-positioning operator: FNO backbone + occupancy head + loss.

The operator is a 1D Fourier neural operator over the locus coordinate,

    (one-hot sequence, energy landscape, inverse temperature) -> occupancy(x),

with a logistic output head so the prediction is a valid probability in
``[0, 1]`` by construction (the reference occupancy is a coverage
probability, never outside the interval).

Why a spectral operator: the label at position ``x`` depends on every
footprint start in ``[x - W + 1, x]`` *and*, through the partition function,
on the whole locus -- the coupling length is the 147 bp footprint, and the
label is a global functional of the landscape.  A local convolution with a
small receptive field cannot represent that; ``README.md`` reports the
receptive-field ablation that measures the gap.

Length generality: the spectral conv operates on a fixed number of Fourier
modes, so the same trained operator evaluates at loci longer than the
training length (``evaluate.py`` reports the held-out length extrapolation).
"""

import torch

from physicsnemo.models.fno import FNO

_N_CHANNELS = 6  # one-hot bases (4) + energy landscape (1) + gamma (1)


class NucleosomeOperator(torch.nn.Module):
    """FNO-1D operator mapping a locus landscape to an occupancy profile.

    Parameters
    ----------
    latent_channels : int, optional, default=64
        Width of the spectral layers.
    num_fno_layers : int, optional, default=4
        Number of spectral convolution blocks.
    num_fno_modes : int, optional, default=64
        Number of Fourier modes retained (the operator's spectral budget).
    padding : int, optional, default=16
        Domain padding in the spectral convolution, with constant padding to
        keep the locus ends non-periodic.

    Notes
    -----
    A factored head (a position-wise numerator logit minus a globally pooled
    normalization logit, mirroring ``P(x) = N(x) / Z``) was measured against
    this single-field head at equal budget and came out neutral -- see the
    architecture ablations in ``README.md``.
    """

    def __init__(
        self,
        latent_channels: int = 64,
        num_fno_layers: int = 4,
        num_fno_modes: int = 64,
        padding: int = 16,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.num_fno_layers = num_fno_layers
        self.num_fno_modes = num_fno_modes
        self.padding = padding
        self.fno = FNO(
            in_channels=_N_CHANNELS,
            out_channels=1,
            dimension=1,
            latent_channels=latent_channels,
            num_fno_layers=num_fno_layers,
            num_fno_modes=num_fno_modes,
            padding=padding,
            padding_type="constant",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(n, 6, L)`` input tensor -> ``(n, L)`` occupancy in ``[0, 1]``."""
        return torch.sigmoid(self.fno(x)).squeeze(1)


def relative_l2_loss(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Mean squared relative L2 error over a batch of profiles.

    ``pred`` and ``ref`` are ``(n, L)``; each sample contributes
    ``||pred - ref||^2 / ||ref||^2``, which is the square of the metric the
    acceptance gate reports, so training optimizes exactly what is measured.

    Parameters
    ----------
    pred, ref : torch.Tensor
        Predicted and reference occupancy profiles.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    num = (pred - ref).pow(2).sum(dim=-1)
    den = ref.pow(2).sum(dim=-1).clamp_min(1e-12)
    return (num / den).mean()


def mse_loss(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Plain mean-squared error over all profile entries."""
    return (pred - ref).pow(2).mean()


def make_loss(name: str):
    """Resolve a loss name from the config to its callable."""
    if name == "rel_l2":
        return relative_l2_loss
    if name == "mse":
        return mse_loss
    raise ValueError(f"unknown loss {name!r} (expected 'rel_l2' or 'mse')")


def sample_batch(
    split: dict,
    index: torch.Tensor,
    norm: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch of split rows into (input, target) tensors on device."""
    from helpers.data import encode_input

    x = encode_input(
        split["seq"][index], split["gamma"][index], split["energy"][index], norm
    ).to(device)
    y = split["occupancy"][index].to(device).to(torch.float32)
    return x, y

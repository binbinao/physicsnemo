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

"""Startup-gate tests: the self-test suite must pass and report diagnostics."""

import torch
from helpers import selftest


def test_self_tests_pass_and_report():
    """The startup gate passes and returns the measured diagnostics."""
    out = selftest.run_self_tests(seed=4, verbose=False)
    assert out["brute_force_max_dev"] < 1e-10
    assert out["gamma0_occ_max_dev"] < 1e-9
    assert out["gamma0_z_rel_err"] < 1e-9
    assert out["energy_max_dev"] < 1e-10
    assert out["favoured_mean_occ"] > 0.9
    assert out["penalised_mean_occ"] < 0.05
    contrasts = out["contrast_by_gamma"]
    assert contrasts[0] < contrasts[1] < contrasts[2]
    assert 0.0 < out["corpus_mean_occupancy"] < 1.0


def test_self_test_failure_is_a_runtime_error():
    """The gate raises a RuntimeError subclass so train.py can abort."""
    assert issubclass(selftest.SelfTestFailure, RuntimeError)


def test_naive_reference_matches_fast_path():
    """The gate's independent window-sum reference agrees with the fast path."""
    from helpers import energy, genome

    table = energy.make_energy_table(width=16)
    seq = genome.sample_sequences(2, 64, torch.Generator().manual_seed(6))
    fast = energy.sliding_energies(seq, table)
    slow = selftest._naive_sliding_energies(seq, table)
    assert torch.allclose(fast, slow, atol=1e-10)


def test_self_tests_are_deterministic():
    """Two runs with the same seed report identical diagnostics."""
    a = selftest.run_self_tests(seed=8, verbose=False)
    b = selftest.run_self_tests(seed=8, verbose=False)
    assert a == b


def test_brute_force_oracle_rejects_a_wrong_model():
    """The oracle is sensitive: a perturbed recursion cannot pass it."""
    from helpers import nucleosome

    length, width = 16, 4
    energies = torch.zeros(1, length - width + 1, dtype=torch.float64)
    energies[0, 2] = -3.0
    z_ref, occ_ref = nucleosome.brute_force_profiles(1.0, energies[0], length, width)
    occ_dp, log_z = nucleosome.reference_profiles(
        torch.tensor([1.0], dtype=torch.float64), energies, length, width
    )
    assert abs(float(torch.exp(log_z[0])) - z_ref) < 1e-10
    # A model that ignores the favourable site must miss the oracle noticeably.
    wrong = occ_ref.clone()
    wrong[2 : 2 + width] -= 0.05
    assert float((wrong - occ_dp[0]).abs().max()) > 1e-2

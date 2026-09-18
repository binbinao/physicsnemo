# Drop-Test Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `examples/structural_mechanics/drop_test_benchmark/` — a reproducible accuracy + speedup benchmark over the existing drop_test GeoTransolver one-shot pipeline, emitting JSON + Markdown reports.

**Architecture:** A new self-contained example directory that imports `drop_test`'s `datapipe.py`, `rollout.py`, `vtu_reader.py`, and `utils.py` via `sys.path` injection (examples tree has no package structure). Benchmark owns only: metrics computation, timing loop, baseline loading, report rendering. The model forward is `model(sample=SimSample, data_stats=dict) -> [N, T, Fo]` tensor — identical to `inference.py`'s call. Never modify anything under `drop_test/`.

**Tech Stack:** Python 3.11, PyTorch, Hydra/OmegaConf, pyvista (synthetic VTU in tests), pytest.

**Spec:** `docs/superpowers/specs/2026-09-18-drop-test-benchmark-design.md`

## Global Constraints

- Repo is PhysicsNeMo at `/data/physicsnemo`; every new `.py`/`.yaml` file starts with the exact SPDX header from `test/ci_tests/copyright.txt` (`# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.` / `# SPDX-FileCopyrightText: All rights reserved.` / `# SPDX-License-Identifier: Apache-2.0`).
- README.md must have headings `## Getting Started`, `## Additional Information`, `## References` (per `examples/.markdownlint.yaml` MD043); lines ≤ 88 chars (MD013); README images would need `/docs/img/` paths — this example uses no images.
- Reuse via `sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "drop_test"))` — never copy datapipe/rollout/reader code, never edit `drop_test/`.
- Timing defaults: `warmup_iterations: 3`, `timed_iterations: 10` (spec-mandated); timing region = device transfer + forward, excluding file I/O.
- Tests must run CPU-only, no real data, no real checkpoint (synthetic micro-VTU + monkeypatched model).
- `requirements.txt` mirrors drop_test's: `pyvista>=0.43.0`, `tabulate>=0.9.0`, `tensorboard>=2.20.0`, `torchinfo>=1.8`.
- Commit style: conventional-ish one-liners with DCO `-s` sign-off.
- Tests run from repo root: `pytest examples/structural_mechanics/drop_test_benchmark/tests/ -v` (bare root pytest ignores examples in CI, but direct path invocation works).

---

### Task 1: Metrics module — accuracy + timing math

**Files:**
- Create: `examples/structural_mechanics/drop_test_benchmark/metrics.py`
- Test: `examples/structural_mechanics/drop_test_benchmark/tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing (pure functions on torch tensors).
- Produces (used by Task 3's `benchmark.py`):
  - `accuracy_metrics(pred: torch.Tensor, truth: torch.Tensor) -> dict` — inputs `[N, T, C]` float tensors (any channel count); returns `{"mae": float, "rel_l2": float}` where `rel_l2 = ||pred - truth||_2 / ||truth||_2` (if `||truth|| == 0`, `rel_l2 = 0.0` when pred is also zero else `float("inf")`).
  - `timing_stats(seconds: list[float]) -> dict` — returns `{"mean": float, "std": float, "cv": float}` where `cv = std/mean` (0.0 if mean == 0).
  - `speedup(baseline_seconds: float | None, mean_inference_seconds: float) -> float | None` — `None` if baseline is `None` or `mean_inference_seconds <= 0`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch

from metrics import accuracy_metrics, speedup, timing_stats


def test_accuracy_identical_zero_error():
    t = torch.randn(4, 5, 3)
    m = accuracy_metrics(t, t.clone())
    assert m["mae"] == 0.0
    assert m["rel_l2"] == 0.0


def test_accuracy_known_relative_l2():
    # pred = 2 * truth -> ||pred-truth|| / ||truth|| = 1.0
    t = torch.randn(6, 2, 3)
    m = accuracy_metrics(t * 2, t)
    assert math.isclose(m["rel_l2"], 1.0, rel_tol=1e-6)


def test_accuracy_known_mae():
    t = torch.zeros(3, 3, 1)
    p = torch.ones(3, 3, 1) * 0.5
    m = accuracy_metrics(p, t)
    assert math.isclose(m["mae"], 0.5, rel_tol=1e-9)


def test_accuracy_zero_truth_handles_gracefully():
    z = torch.zeros(2, 2, 3)
    m = accuracy_metrics(torch.zeros(2, 2, 3), z)
    assert m["rel_l2"] == 0.0
    m2 = accuracy_metrics(torch.ones(2, 2, 3), z)
    assert math.isinf(m2["rel_l2"])


def test_timing_stats_basic():
    s = timing_stats([1.0, 1.0, 1.0, 1.0])
    assert s["mean"] == 1.0
    assert s["std"] == 0.0
    assert s["cv"] == 0.0


def test_timing_stats_cv():
    s = timing_stats([1.0, 1.2])
    assert math.isclose(s["mean"], 1.1, rel_tol=1e-9)
    assert s["cv"] > 0.0


def test_speedup_math_and_null():
    assert speedup(100.0, 1.0) == 100.0
    assert speedup(None, 1.0) is None
    assert speedup(100.0, 0.0) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/physicsnemo && python -m pytest examples/structural_mechanics/drop_test_benchmark/tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# metrics.py
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy and timing metrics for the drop-test benchmark.

All functions are pure (no device moves, no I/O) so tests run CPU-only.
"""
import statistics

import torch


def accuracy_metrics(pred: torch.Tensor, truth: torch.Tensor) -> dict:
    """Compute MAE and relative L2 between two [N, T, C] field tensors."""
    diff = pred - truth
    mae = diff.abs().mean().item()
    truth_norm = truth.norm().item()
    if truth_norm == 0.0:
        rel_l2 = 0.0 if diff.norm().item() == 0.0 else float("inf")
    else:
        rel_l2 = (diff.norm() / truth.norm()).item()
    return {"mae": mae, "rel_l2": rel_l2}


def timing_stats(seconds: list[float]) -> dict:
    """Mean, std, and coefficient of variation of timed iterations."""
    mean = statistics.fmean(seconds)
    std = statistics.stdev(seconds) if len(seconds) > 1 else 0.0
    cv = std / mean if mean > 0 else 0.0
    return {"mean": mean, "std": std, "cv": cv}


def speedup(baseline_seconds: float | None, mean_inference_seconds: float) -> float | None:
    """Baseline wall-time over inference time; None when incomputable."""
    if baseline_seconds is None or mean_inference_seconds <= 0:
        return None
    return baseline_seconds / mean_inference_seconds
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/physicsnemo && python -m pytest examples/structural_mechanics/drop_test_benchmark/tests/test_metrics.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
cd /data/physicsnemo
git add examples/structural_mechanics/drop_test_benchmark/metrics.py \
        examples/structural_mechanics/drop_test_benchmark/tests/test_metrics.py
git commit -s -m "benchmark: add accuracy and timing metrics module"
```

---

### Task 2: Report renderer — JSON + Markdown

**Files:**
- Create: `examples/structural_mechanics/drop_test_benchmark/report.py`
- Test: `examples/structural_mechanics/drop_test_benchmark/tests/test_report.py`

**Interfaces:**
- Consumes: nothing (pure rendering of a results dict).
- Produces (used by Task 3):
  - `render_json(results: dict) -> str` — `json.dumps(results, indent=2, sort_keys=True)`.
  - `render_markdown(results: dict) -> str` — human report; must contain: a per-run table (columns: run, latency mean ± std s, speedup, disp MAE, disp rel-L2, stress MAE, stress rel-L2), an aggregate summary (mean latency, throughput runs/s, aggregate speedup, aggregate errors), `device` line, and warnings (`unstable_timing`, `cpu_device`, `skipped_runs` non-empty, missing-baseline notes).
  - `write_reports(results: dict, out_dir: str) -> list[str]` — writes `benchmark_report.json` + `benchmark_report.md` into `out_dir` (created if missing), returns the two paths.
- **Results dict schema** (produced by Task 3, consumed here — the contract):
  ```python
  {
    "device": "cuda:0" | "cpu",
    "model": "<checkpoint filename>",
    "num_runs": int,
    "runs": [
      {
        "run": "run0001",
        "latency": {"mean": float, "std": float, "cv": float},
        "warmup_iterations": int, "timed_iterations": int,
        "speedup": float | None,
        "baseline_seconds": float | None,   # user_recorded
        "disp": {"mae": float, "rel_l2": float},
        "stress": {"mae": float, "rel_l2": float},
        "warnings": [str, ...],
      }, ...
    ],
    "aggregate": {
      "mean_latency_s": float,
      "throughput_runs_per_s": float,
      "speedup_mean": float | None,     # mean over runs with speedup != None
      "speedup_runs_counted": int,
      "disp": {"mae": float, "rel_l2": float},
      "stress": {"mae": float, "rel_l2": float},
    },
    "skipped_runs": [str, ...],
    "warnings": [str, ...],
  }
  ```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_report.py
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

from report import render_json, render_markdown, write_reports


def _sample_results():
    return {
        "device": "cuda:0",
        "model": "model.mdlus",
        "num_runs": 2,
        "runs": [
            {
                "run": "run0001",
                "latency": {"mean": 0.5, "std": 0.01, "cv": 0.02},
                "warmup_iterations": 3, "timed_iterations": 10,
                "speedup": 200.0, "baseline_seconds": 100.0,
                "disp": {"mae": 0.001, "rel_l2": 0.01},
                "stress": {"mae": 2.0, "rel_l2": 0.05},
                "warnings": [],
            },
            {
                "run": "run0002",
                "latency": {"mean": 0.6, "std": 0.3, "cv": 0.5},
                "warmup_iterations": 3, "timed_iterations": 10,
                "speedup": None, "baseline_seconds": None,
                "disp": {"mae": 0.002, "rel_l2": 0.02},
                "stress": {"mae": 3.0, "rel_l2": 0.06},
                "warnings": ["unstable_timing"],
            },
        ],
        "aggregate": {
            "mean_latency_s": 0.55,
            "throughput_runs_per_s": 1.818,
            "speedup_mean": 200.0,
            "speedup_runs_counted": 1,
            "disp": {"mae": 0.0015, "rel_l2": 0.015},
            "stress": {"mae": 2.5, "rel_l2": 0.055},
        },
        "skipped_runs": ["run0003"],
        "warnings": ["missing_baseline: run0002", "skipped: run0003"],
    }


def test_render_json_roundtrip():
    r = _sample_results()
    assert json.loads(render_json(r))["num_runs"] == 2


def test_render_markdown_contains_core_sections():
    md = render_markdown(_sample_results())
    assert "# Drop-Test Benchmark Report" in md
    assert "cuda:0" in md
    assert "run0001" in md and "run0002" in md
    assert "200.00" in md            # speedup cell
    assert "unstable_timing" in md   # run-level warning surfaced
    assert "missing_baseline" in md
    assert "run0003" in md           # skipped run listed
    assert "1.82" in md              # throughput


def test_render_markdown_null_speedup_renders_dash():
    md = render_markdown(_sample_results())
    assert "—" in md  # null speedup rendered as em dash


def test_write_reports_creates_files(tmp_path):
    paths = write_reports(_sample_results(), str(tmp_path / "out"))
    assert len(paths) == 2
    for p in paths:
        assert open(p).read()
    names = {p.split("/")[-1] for p in paths}
    assert names == {"benchmark_report.json", "benchmark_report.md"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/physicsnemo && python -m pytest examples/structural_mechanics/drop_test_benchmark/tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'report'`

- [ ] **Step 3: Write minimal implementation**

```python
# report.py
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render benchmark results to JSON and Markdown reports."""
import json
import os


def render_json(results: dict) -> str:
    return json.dumps(results, indent=2, sort_keys=True)


def _fmt(x, spec=".4g"):
    return "—" if x is None else format(x, spec)


def render_markdown(results: dict) -> str:
    lines = ["# Drop-Test Benchmark Report", ""]
    lines.append(f"- **Device:** `{results['device']}`")
    lines.append(f"- **Model:** `{results['model']}`")
    lines.append(f"- **Runs:** {results['num_runs']} "
                 f"(skipped: {len(results['skipped_runs'])})")
    if results["device"] == "cpu":
        lines.append("- **Note:** ran on CPU — speedup vs Radioss is not comparable.")
    lines += ["", "| run | latency (s) | speedup | disp MAE | disp rel-L2 "
              "| stress MAE | stress rel-L2 |", "|---|---|---|---|---|---|---|"]
    for r in results["runs"]:
        lat = f"{r['latency']['mean']:.4f} ± {r['latency']['std']:.4f}"
        lines.append(
            f"| {r['run']} | {lat} | {_fmt(r['speedup'], '.2f')} "
            f"| {r['disp']['mae']:.4g} | {r['disp']['rel_l2']:.4g} "
            f"| {r['stress']['mae']:.4g} | {r['stress']['rel_l2']:.4g} |"
        )
    agg = results["aggregate"]
    lines += ["", "## Aggregate", ""]
    lines.append(f"- Mean latency: {agg['mean_latency_s']:.4f} s")
    lines.append(f"- Throughput: {agg['throughput_runs_per_s']:.2f} runs/s")
    lines.append(f"- Speedup (mean over {agg['speedup_runs_counted']} runs, "
                 f"baseline `user_recorded`): {_fmt(agg['speedup_mean'], '.2f')}×")
    lines.append(f"- Displacement: MAE {agg['disp']['mae']:.4g}, "
                 f"rel-L2 {agg['disp']['rel_l2']:.4g}")
    lines.append(f"- Stress: MAE {agg['stress']['mae']:.4g}, "
                 f"rel-L2 {agg['stress']['rel_l2']:.4g}")
    if results["skipped_runs"]:
        lines += ["", "## Skipped runs", ""] + [f"- `{r}`" for r in results["skipped_runs"]]
    if results["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in results["warnings"]]
    for r in results["runs"]:
        for w in r["warnings"]:
            lines.append(f"- `{r['run']}`: {w}")
    return "\n".join(lines) + "\n"


def write_reports(results: dict, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    j = os.path.join(out_dir, "benchmark_report.json")
    m = os.path.join(out_dir, "benchmark_report.md")
    with open(j, "w") as f:
        f.write(render_json(results))
    with open(m, "w") as f:
        f.write(render_markdown(results))
    return [j, m]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/physicsnemo && python -m pytest examples/structural_mechanics/drop_test_benchmark/tests/test_report.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
cd /data/physicsnemo
git add examples/structural_mechanics/drop_test_benchmark/report.py \
        examples/structural_mechanics/drop_test_benchmark/tests/test_report.py
git commit -s -m "benchmark: add JSON/Markdown report renderer"
```

---

### Task 3: Benchmark engine — Hydra entry, per-run loop, results assembly

**Files:**
- Create: `examples/structural_mechanics/drop_test_benchmark/benchmark.py`
- Create: `examples/structural_mechanics/drop_test_benchmark/conf/config.yaml`
- Create: `examples/structural_mechanics/drop_test_benchmark/conf/baseline.yaml`
- Create: `examples/structural_mechanics/drop_test_benchmark/conf/reader/vtu.yaml` (copy content of `drop_test/conf/reader/vtu.yaml`, plus SPDX header)
- Create: `examples/structural_mechanics/drop_test_benchmark/conf/datapipe/point_cloud.yaml` (copy content of `drop_test/conf/datapipe/point_cloud.yaml`, plus SPDX header)
- Create: `examples/structural_mechanics/drop_test_benchmark/conf/model/geotransolver_one_shot.yaml` (copy content of `drop_test/conf/model/geotransolver_one_shot.yaml`, plus SPDX header)
- Test: `examples/structural_mechanics/drop_test_benchmark/tests/test_benchmark.py`

**Interfaces:**
- Consumes:
  - Task 1: `accuracy_metrics(pred, truth) -> dict`, `timing_stats(seconds) -> dict`, `speedup(baseline, mean) -> float | None` (import from `metrics`).
  - Task 2: `write_reports(results, out_dir) -> list[str]` (import from `report`).
  - drop_test via sys.path (verified signatures):
    - `datapipe.DropTestPointCloudDataset` ctor kwargs follow `inference.py:220-230`: `name, reader, split, num_steps, num_samples=1, logger, data_dir, sample_type="all_time_steps"` plus its own config keys (`static_features, dynamic_features, dynamic_targets, global_features, global_features_filepath, log_transform_targets, stats_dir`).
    - **CRITICAL (datapipe.py:340-352):** `split="test"` raises `FileNotFoundError` unless `<stats_dir>/node_stats.json` + `feature_stats.json` exist. `split="train"` computes and saves them. Therefore the benchmark builds each run's dataset as `split="train"` **into a per-run temp dir** (`stats_dir=<tmp>/stats`) so stats are computed locally without needing the training stats directory. This mirrors what the data actually is (one run = one sample), keeps the benchmark self-contained, and avoids the reader's write side effects on cwd (see next bullet).
    - **CRITICAL (vtu_reader.py:292,159-196):** `Reader.__call__` with `split` not in `("train","validation")` sets `write_vtu=True`, which writes `frame_*.vtu` into `./output_<run>/` under the **current working directory**. Using `split="train"` keeps `write_vtu=False` — no side effects.
    - `datapipe.simsample_collate`, `rollout.GeoTransolverOneShot` (Hydra `_target_: rollout.GeoTransolverOneShot`, instantiate AFTER sys.path injection).
    - Normalization round-trip (both in `drop_test/inference.py`, importable): `denormalize_positions(y, pos_mean, pos_std)`, `_extract_extra_fields(data, target_series, T)`, `_denormalize_extra_fields(extra, dyn_stats, log_transform)`. Predictions and targets are in **normalized space**; accuracy MUST be computed on denormalized physical values.
    - `physicsnemo.utils.load_checkpoint(path, models=[model], device=device)`.
  - Model forward contract (`rollout.py:97`): `model(sample=SimSample, data_stats=dict) -> Tensor[N, T, Fo]` (normalized space); channels `[:, :, :3]` = positions.
  - Dataset stats dicts: `dataset.node_stats` (`pos_mean`, `pos_std`), `dataset.feature_stats`, `dataset.dynamic_target_stats` (`Von_Mises_mean`, `Von_Mises_std` — may be `{}` when stats absent).
- Produces: `run_benchmark(cfg: DictConfig) -> dict` — the results dict (schema in Task 2); CLI via `@hydra.main(config_path="conf", config_name="config", version_base="1.3")`.

**Config files.** `conf/config.yaml` (SPDX header elided here — add it):

```yaml
defaults:
  - reader: vtu
  - datapipe: point_cloud
  - model: geotransolver_one_shot
  - _self_

training:
  num_time_steps: 100
  ckpt_path: ???            # trained drop_test checkpoint (.mdlus)
  num_dataloader_workers: 0

inference:
  raw_data_dir_test: ???    # directory of .vtu test runs

benchmark:
  warmup_iterations: 3
  timed_iterations: 10
  output_dir: "./benchmark_output"
  baseline_file: ???        # per-run Radioss wall-times YAML (see conf/baseline.yaml)
```

`conf/baseline.yaml` (template; user edits with their measured Radioss wall-times):

```yaml
# Per-run OpenRadioss solver wall-times in seconds (user_recorded).
# Keys are VTU basenames without extension, e.g.:
# run0001: 3600.0
# run0002: 3600.0
```

- [ ] **Step 1: Write the failing test** (synthetic micro-VTU + monkeypatched model; CPU-only, no real checkpoint)

```python
# tests/test_benchmark.py
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

BENCH_DIR = Path(__file__).resolve().parents[1]
DROP_TEST_DIR = BENCH_DIR.parent / "drop_test"
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(DROP_TEST_DIR))

from benchmark import run_benchmark  # noqa: E402

pytest.importorskip("pyvista")


def _make_micro_vtu(path, n_nodes=6, t_steps=3):
    """Synthetic drop-test VTU matching the reader contract exactly.

    - UnstructuredGrid with >=1 tetra cell (vtu_reader.py:71 rejects PolyData)
    - displacement_tXXX per timestep (regex displacement_t0.\\d{3,})
    - Von_Mises_tXXX scalar per timestep (dynamic target series)
    """
    import pyvista as pv

    rng = np.random.default_rng(0)
    pts = rng.random((n_nodes, 3))
    # one tetra over the first 4 nodes; reader keeps all nodes
    cells = [4, 0, 1, 2, 3]
    grid = pv.UnstructuredGrid(
        cells, np.array([pv.CellType.TETRA]), pts
    )
    for t in range(t_steps):
        disp = rng.random((n_nodes, 3)) * 0.01
        if t == 0:
            disp = np.zeros((n_nodes, 3))  # curator convention: t0 disp = 0
        grid[f"displacement_t{t:0.3f}"] = disp
        grid[f"Von_Mises_t{t:0.3f}"] = rng.random((n_nodes, 1)) * 10.0
    grid.save(str(path))


def _cfg(tmp_path):
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    _make_micro_vtu(data_dir / "run0001.vtu")
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text("run0001: 100.0\n")
    return OmegaConf.create({
        "training": {"num_time_steps": 3, "ckpt_path": "fake.mdlus",
                     "num_dataloader_workers": 0},
        "inference": {"raw_data_dir_test": str(data_dir)},
        "benchmark": {"warmup_iterations": 1, "timed_iterations": 2,
                      "output_dir": str(tmp_path / "out"),
                      "baseline_file": str(baseline)},
        "reader": {"_target_": "vtu_reader.Reader", "_convert_": "all"},
        "datapipe": {"_target_": "datapipe.DropTestPointCloudDataset",
                     "static_features": [], "dynamic_features": [],
                     "dynamic_targets": ["Von_Mises"],
                     "global_features": None,
                     "global_features_filepath": None,
                     "log_transform_targets": False},
        "model": {"_target_": "rollout.GeoTransolverOneShot", "_convert_": "all"},
    })


class _FakeModel(torch.nn.Module):
    """Identity surrogate: predicts the normalized target verbatim."""

    def forward(self, sample, data_stats):
        return sample.node_target.clone()


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(
        "benchmark._build_model",
        lambda cfg, device: _FakeModel().to(device).eval(),
    )


def test_run_benchmark_end_to_end(tmp_path, monkeypatch, fake_model):
    # keep reader's (hypothetical) cwd writes inside tmp
    monkeypatch.chdir(tmp_path)
    results = run_benchmark(_cfg(tmp_path))
    assert results["num_runs"] == 1
    r = results["runs"][0]
    assert r["run"] == "run0001"
    assert r["latency"]["mean"] > 0
    assert set(r["latency"]) == {"mean", "std", "cv"}
    # identity model -> near-zero physical-space errors
    assert r["disp"]["mae"] < 1e-5
    assert r["stress"]["mae"] < 1e-5
    assert r["speedup"] == pytest.approx(
        100.0 / r["latency"]["mean"], rel=0.5
    )
    assert results["aggregate"]["speedup_mean"] is not None
    assert results["device"] == "cpu"
    assert "cpu_device" in results["warnings"]


def test_run_benchmark_missing_ckpt_fails_fast(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(tmp_path)  # fake_model fixture NOT applied here
    cfg.training.ckpt_path = str(tmp_path / "nonexistent.mdlus")
    with pytest.raises(FileNotFoundError, match="drop_test example"):
        run_benchmark(cfg)


def test_run_benchmark_empty_dir_fails_fast(tmp_path, monkeypatch, fake_model):
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    cfg.inference.raw_data_dir_test = str(empty)
    with pytest.raises(FileNotFoundError, match="PhysicsNeMo-Curator"):
        run_benchmark(cfg)


def test_run_benchmark_missing_baseline_yields_null_speedup(
    tmp_path, monkeypatch, fake_model
):
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(tmp_path)
    cfg.benchmark.baseline_file = str(tmp_path / "nope.yaml")
    results = run_benchmark(cfg)
    assert results["runs"][0]["speedup"] is None
    assert results["aggregate"]["speedup_mean"] is None
    assert any("missing_baseline" in w for w in results["warnings"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/physicsnemo && python -m pytest examples/structural_mechanics/drop_test_benchmark/tests/test_benchmark.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark'`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmark.py
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drop-test benchmark: accuracy + speedup vs an OpenRadioss baseline."""
import os
import sys
import tempfile
import time
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_DROP_TEST = _HERE.parent / "drop_test"
for _p in (str(_HERE), str(_DROP_TEST)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datapipe import simsample_collate  # noqa: E402  (drop_test)
from inference import (  # noqa: E402  (drop_test)
    _denormalize_extra_fields,
    _extract_extra_fields,
    denormalize_positions,
)
from metrics import accuracy_metrics, speedup, timing_stats  # noqa: E402


def _build_model(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    """Instantiate the model and load the trained checkpoint.

    Raises FileNotFoundError with actionable guidance when the checkpoint is
    missing — the benchmark cannot run without a trained drop_test model.
    """
    model = instantiate(cfg.model)
    ckpt = str(cfg.training.ckpt_path)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Train the drop_test example first "
            "(examples/structural_mechanics/drop_test) or point "
            "training.ckpt_path at a .mdlus checkpoint."
        )
    from physicsnemo.utils import load_checkpoint

    load_checkpoint(ckpt, models=[model], device=device)
    return model.to(device).eval()


def _load_baseline(cfg: DictConfig) -> dict:
    bf = cfg.benchmark.get("baseline_file", None)
    if not bf:
        return {}
    bf = str(bf)
    if not os.path.isfile(bf):
        return {}
    return dict(OmegaConf.load(bf))


class _RunBundle:
    """One run's dataset + precomputed device-resident stats."""

    def __init__(self, cfg: DictConfig, run_path: str, device):
        reader = instantiate(cfg.reader)
        # split="train" => reader write_vtu=False (no cwd side effects) and
        # the dataset computes stats locally instead of requiring a stats dir.
        with tempfile.TemporaryDirectory(prefix="drop_test_bench_") as tmp:
            os.symlink(run_path, os.path.join(tmp, os.path.basename(run_path)))
            self.dataset = instantiate(
                cfg.datapipe,
                name="drop_test_benchmark",
                reader=reader,
                split="train",
                num_steps=int(cfg.training.num_time_steps),
                num_samples=1,
                data_dir=tmp,
                logger=None,
                sample_type="all_time_steps",
                stats_dir=os.path.join(tmp, "stats"),
            )
        self.loader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=simsample_collate,
        )
        sample = next(iter(self.loader))
        if isinstance(sample, list):
            sample = sample[0]
        self.sample = sample.to(device)
        self.data_stats = {
            "node": {k: v.to(device) for k, v in self.dataset.node_stats.items()},
            "edge": {k: v.to(device) for k, v in getattr(self.dataset, "edge_stats", {}).items()},
            "feature": {k: v.to(device) for k, v in getattr(self.dataset, "feature_stats", {}).items()},
            "dynamic_target": {
                k: v.to(device)
                for k, v in getattr(self.dataset, "dynamic_target_stats", {}).items()
            },
        }


def _timed_forward(model, bundle, device, warmup, timed):
    """Warmup then time `timed` iterations of device-transfer + forward."""
    with torch.no_grad():
        for _ in range(warmup):
            model(sample=bundle.sample, data_stats=bundle.data_stats)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = []
        for _ in range(timed):
            t0 = time.perf_counter()
            model(sample=bundle.sample, data_stats=bundle.data_stats)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds.append(time.perf_counter() - t0)
    return timing_stats(seconds)


def _physical_accuracy(bundle, pred, T, log_transform):
    """Denormalize pred + target, then compute displacement / stress errors.

    Model outputs and dataset targets are normalized; physical-space errors
    require the round-trip through the dataset's own stats.
    """
    ds = bundle.data_stats
    pred_pos = denormalize_positions(
        pred[:, :, :3].transpose(0, 1), ds["node"]["pos_mean"], ds["node"]["pos_std"]
    )
    exact_pos = denormalize_positions(
        bundle.sample.node_target[:, :, :3].transpose(0, 1),
        ds["node"]["pos_mean"],
        ds["node"]["pos_std"],
    )
    disp = accuracy_metrics(pred_pos, exact_pos)

    target_series = getattr(bundle.sample, "target_series", None) or {}
    stress = {"mae": float("nan"), "rel_l2": float("nan")}
    vm = "Von_Mises"
    extra_p = _denormalize_extra_fields(
        _extract_extra_fields(pred, target_series, T),
        ds["dynamic_target"],
        log_transform,
    )
    extra_e = _denormalize_extra_fields(
        _extract_extra_fields(bundle.sample.node_target, target_series, T),
        ds["dynamic_target"],
        log_transform,
    )
    if vm in extra_p and vm in extra_e:
        p = torch.cat(extra_p[vm], dim=0)
        e = torch.cat(extra_e[vm], dim=0)
        stress = accuracy_metrics(p, e)
    return {"disp": disp, "stress": stress}


def _nanmean(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")


def run_benchmark(cfg: DictConfig) -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parent = str(cfg.inference.raw_data_dir_test)
    if not os.path.isdir(parent):
        raise FileNotFoundError(f"Test directory not found: {parent}")
    run_paths = sorted(
        f.path
        for f in os.scandir(parent)
        if f.is_file() and f.name.lower().endswith(".vtu")
    )
    if not run_paths:
        raise FileNotFoundError(
            f"No .vtu files under {parent}. Generate them with the "
            "PhysicsNeMo-Curator drop_test ETL recipe."
        )
    model = _build_model(cfg, device)
    baseline = _load_baseline(cfg)
    T = int(cfg.training.num_time_steps) - 1
    warmup = int(cfg.benchmark.warmup_iterations)
    timed = int(cfg.benchmark.timed_iterations)
    log_transform = bool(
        OmegaConf.select(cfg, "datapipe.log_transform_targets", default=False)
    )

    runs, skipped = [], []
    for rp in run_paths:
        name = Path(rp).stem
        try:
            bundle = _RunBundle(cfg, rp, device)
            latency = _timed_forward(model, bundle, device, warmup, timed)
            with torch.no_grad():
                pred = model(sample=bundle.sample, data_stats=bundle.data_stats)
            acc = _physical_accuracy(bundle, pred, T, log_transform)
        except torch.cuda.OutOfMemoryError:
            skipped.append(name)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        warnings = ["unstable_timing"] if latency["cv"] > 0.10 else []
        base = baseline.get(name)
        runs.append({
            "run": name,
            "latency": latency,
            "warmup_iterations": warmup,
            "timed_iterations": timed,
            "speedup": speedup(base, latency["mean"]),
            "baseline_seconds": base,
            "disp": acc["disp"],
            "stress": acc["stress"],
            "warnings": warnings,
        })

    if not runs:
        raise RuntimeError("All runs failed (OOM) — nothing to benchmark.")

    mean_lat = sum(r["latency"]["mean"] for r in runs) / len(runs)
    sps = [r["speedup"] for r in runs if r["speedup"] is not None]
    warnings = [f"missing_baseline: {r['run']}" for r in runs if r["speedup"] is None]
    warnings += [f"skipped: {s}" for s in skipped]
    if device.type == "cpu":
        warnings.append("cpu_device")
    return {
        "device": str(device),
        "model": os.path.basename(str(cfg.training.ckpt_path)),
        "num_runs": len(runs),
        "runs": runs,
        "aggregate": {
            "mean_latency_s": mean_lat,
            "throughput_runs_per_s": 1.0 / mean_lat if mean_lat > 0 else 0.0,
            "speedup_mean": (sum(sps) / len(sps)) if sps else None,
            "speedup_runs_counted": len(sps),
            "disp": {
                k: sum(r["disp"][k] for r in runs) / len(runs)
                for k in ("mae", "rel_l2")
            },
            "stress": {
                k: _nanmean([r["stress"][k] for r in runs])
                for k in ("mae", "rel_l2")
            },
        },
        "skipped_runs": skipped,
        "warnings": warnings,
    }


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    from report import write_reports

    results = run_benchmark(cfg)
    paths = write_reports(results, str(cfg.benchmark.output_dir))
    print(f"Benchmark reports written: {paths}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/physicsnemo && python -m pytest examples/structural_mechanics/drop_test_benchmark/tests/ -v`
Expected: all PASS (7 metrics + 4 report + 4 benchmark = 15)

- [ ] **Step 5: Commit**

```bash
cd /data/physicsnemo
git add examples/structural_mechanics/drop_test_benchmark/
git commit -s -m "benchmark: add Hydra benchmark engine with per-run timing loop"
```

---

### Task 4: README, requirements, license headers audit, markdown lint

**Files:**
- Create: `examples/structural_mechanics/drop_test_benchmark/README.md`
- Create: `examples/structural_mechanics/drop_test_benchmark/requirements.txt`
- Modify (if missing headers): all files from Tasks 1–3

**Interfaces:**
- Consumes: finished benchmark engine; repo conventions from `AGENTS.md`.
- Produces: complete example directory ready for review; updates `examples/README.md` index table with one row for `structural_mechanics/drop_test_benchmark`.

- [ ] **Step 1: Write README.md** (SPDX header not required for .md but follow drop_test README style)

Structure — `# Drop-Test Benchmark` intro (2 paragraphs: purpose = accuracy + speedup vs OpenRadioss for trained drop_test checkpoints), then:

```markdown
## Getting Started

### Prerequisites
- Trained drop_test checkpoint (`.mdlus`) — see the
  [drop_test example](../drop_test/README.md) for training.
- Test VTU data generated with
  [PhysicsNeMo-Curator](https://github.com/NVIDIA/physicsnemo-curator/tree/main/examples/structural_mechanics/drop_test).
- `pip install -r requirements.txt`

### Record Radioss baseline times
Edit `conf/baseline.yaml` with per-run OpenRadioss solver wall-times
(seconds), keyed by VTU basename without extension.

### Run the benchmark
```bash
python benchmark.py \
    training.ckpt_path=/path/to/model.mdlus \
    inference.raw_data_dir_test=/path/to/test_vtus \
    benchmark.output_dir=./benchmark_output
```

Reports: `benchmark_report.json` (machine-readable) and
`benchmark_report.md` (tables + warnings). Timing region = device transfer +
forward pass; defaults warmup=3, timed=10 — override via
`benchmark.warmup_iterations` / `benchmark.timed_iterations`.

### Run the tests
```bash
cd /data/physicsnemo
python -m pytest examples/structural_mechanics/drop_test_benchmark/tests/ -v
```

## Additional Information
- Metrics: per-run and aggregate MAE + relative L2 for displacement and
  Von_Mises stress; latency mean ± std; speedup = user_recorded Radioss
  wall-time / mean inference latency.
- CPU fallback: reports are tagged `cpu_device` and speedup is labeled
  not comparable.
- OOM runs are skipped and listed under "Skipped runs".

## References
- [drop_test example](../drop_test/README.md) — training and data prep
- [OpenRadioss](https://www.openradioss.org/)
- [PhysicsNeMo-Curator drop_test ETL](https://github.com/NVIDIA/physicsnemo-curator/tree/main/examples/structural_mechanics/drop_test)
```

- [ ] **Step 2: Write requirements.txt** (match drop_test's, minus tensorboard if unused — keep identical for simplicity):

```text
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
pyvista>=0.43.0
tabulate>=0.9.0
tensorboard>=2.20.0
torchinfo>=1.8
```

- [ ] **Step 3: Audit license headers**

Run: `cd /data/physicsnemo && python test/ci_tests/header_check.py examples/structural_mechanics/drop_test_benchmark/ -r`
Expected: no failures (fix any file missing the block; `.yaml` files need it, hook misses `.yml` — use `.yaml` extensions everywhere).

- [ ] **Step 4: Lint**

Run: `cd /data/physicsnemo && pre-commit run --files examples/structural_mechanics/drop_test_benchmark/` (markdownlint on README; ruff-format may reformat the Python — accept its changes).

- [ ] **Step 5: Update examples index**

Add row to the structural-mechanics section of `examples/README.md`:
`| [Drop-Test Benchmark](structural_mechanics/drop_test_benchmark/README.md) | Accuracy + speedup benchmark for trained drop-test surrogate models vs OpenRadioss | Structural Mechanics |`

- [ ] **Step 6: Commit**

```bash
cd /data/physicsnemo
git add examples/structural_mechanics/drop_test_benchmark/README.md \
        examples/structural_mechanics/drop_test_benchmark/requirements.txt \
        examples/README.md
git commit -s -m "benchmark: add README, requirements, and examples index entry"
```

---

### Task 5: Full verification pass

**Files:**
- No new files. Verification-only task.

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: verified acceptance criteria (spec §Acceptance criteria 1–4).

- [ ] **Step 1: Run the full test suite for the example**

Run: `cd /data/physicsnemo && python -m pytest examples/structural_mechanics/drop_test_benchmark/tests/ -v`
Expected: 15 PASS.

- [ ] **Step 2: Smoke-run the CLI help + error paths (CPU, no data)**

Run:
```bash
cd /data/physicsnemo/examples/structural_mechanics/drop_test_benchmark
python benchmark.py training.ckpt_path=/nonexistent.mdlus \
    inference.raw_data_dir_test=/nonexistent 2>&1 | tail -5
```
Expected: `FileNotFoundError` mentioning the test directory with Curator guidance (fail-fast criterion 4).

- [ ] **Step 3: Verify pre-commit on the whole example**

Run: `cd /data/physicsnemo && pre-commit run --files examples/structural_mechanics/drop_test_benchmark/`
Expected: all hooks pass.

- [ ] **Step 4: Final commit if verification produced fixes**

```bash
cd /data/physicsnemo
git add -A examples/structural_mechanics/drop_test_benchmark/
git commit -s -m "benchmark: verification fixes" || echo "nothing to commit"
```

Note for the implementer: acceptance criteria 1–3 (real checkpoint + real VTU end-to-end with meaningful numbers) require trained-model artifacts that don't exist on this machine — the synthetic-data tests verify the same code paths (load → forward → metrics → timing → report → CLI error handling). README documents the real-data commands. Report this limitation honestly when done.

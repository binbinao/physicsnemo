# Drop-Test Benchmark Case Design

- **Date:** 2026-09-18
- **Status:** Approved (design sections confirmed by user)
- **Scope:** New example `examples/structural_mechanics/drop_test_benchmark/`

## Purpose

Build a reproducible benchmark case for the structural-mechanics drop-test
scenario: given a trained GeoTransolver checkpoint and test VTU data, measure
**inference accuracy** (vs. OpenRadioss ground truth) and **inference speed**
(including speedup vs. the Radioss baseline), and emit both machine-readable
and human-readable reports.

This is a **benchmark case**, not a new model or dataset pipeline. It reuses
the existing `examples/structural_mechanics/drop_test/` training/inference
machinery end-to-end and adds only the measurement and reporting layer.

## Requirements (from brainstorming)

| Requirement | Decision |
|---|---|
| Purpose | Benchmark test (accuracy + speedup) |
| Baseline | OpenRadioss explicit solver wall-time |
| Scenario | drop_test (consumer-electronics drop onto rigid wall) |
| Metrics | Accuracy (displacement/stress error) + speedup vs Radioss |
| Scale | Small validation scale, single Tesla T4 (16 GB) |
| Success | Complete reproducible example directory, committable to repo |

## Deliverables

```
examples/structural_mechanics/drop_test_benchmark/
├── README.md                 # Getting Started / Additional Information / References
├── requirements.txt
├── benchmark.py              # Entry point: accuracy + timing + report
├── report.py                 # Report rendering (JSON + Markdown)
├── conf/
│   ├── config.yaml           # Benchmark config (inherits drop_test model config)
│   └── baseline.yaml         # Radioss baseline wall-times per run
└── tests/test_benchmark.py   # Smoke tests on synthetic micro-data
```

### Acceptance criteria

1. `python benchmark.py` runs end-to-end: load data → inference → accuracy
   metrics → timing → dual report output.
2. Accuracy report contains per-run and aggregate displacement and stress
   errors (MAE, relative L2).
3. Performance report contains per-run inference latency (warmup + N timed
   iterations, mean ± std), throughput, and speedup =
   Radioss baseline time / mean inference time.
4. Missing data/checkpoint produce clear, actionable error messages.

## Architecture & Data Flow

### Reuse strategy (zero code duplication, no changes to drop_test)

`benchmark.py` injects `../drop_test/` into `sys.path` and reuses two pieces
as-is:

- `datapipe.py` — VTU loading, mesh/graph construction (imported directly).
- The **model-loading path** — `physicsnemo.utils.load_checkpoint` with the
  same checkpoint-key resolution `InferenceWorkerSingleVtu.__init__` uses.

The **forward-call construction is benchmark-owned** (~30 lines): the worker's
`run_on_single_run` couples forward inference with VTU comparison-file
writing, and the timing scope requires writing to be excluded. Calling the
worker directly would make the mandated timing scope impossible, so
`benchmark.py` builds the forward inputs from `datapipe.py` outputs itself.
`drop_test/inference.py` is read as a reference, never modified.

The examples tree has no package structure (no `__init__.py`), so path
injection is the established pattern for cross-directory reuse.

### Data flow

```text
checkpoint (.mdlus) ──┐
test VTU directory ───┤→ benchmark.py
Radioss baseline (conf/baseline.yaml, per-run solver wall-time) ─┘
    │
    ├─ 1. Load model (physicsnemo.utils.load_checkpoint, same path as inference.py)
    ├─ 2. Per-run inference (one-shot GeoTransolver forward, all time steps at once)
    ├─ 3. Accuracy: predicted vs. VTU ground truth → MAE / relative L2
    │      (displacement, Von_Mises stress)
    ├─ 4. Timing: warmup W iterations → N timed iterations
    │      (torch.cuda.synchronize + time.perf_counter) → mean ± std
    ├─ 5. Speedup: baseline_radioss_time / mean_inference_time
    └─ 6. report.py → benchmark_report.json + benchmark_report.md
```

### Key decisions

- **Timing scope:** end-to-end single-run inference — data transfer to GPU +
  forward pass — excluding VTU file writing. This is the only scope
  comparable to Radioss solve time. Curator ETL preprocessing is excluded
  (one-time offline cost). Concretely, the timed region is:
  `graph tensors → device` through `model(...) → predictions on device`.
- **Timing defaults:** `warmup_iterations: W = 3`, `timed_iterations: N = 10`
  (Hydra-overridable). Per run: mean ± std over N. Aggregate latency is the
  mean over runs.
- **Throughput definition:** runs/second = 1 / aggregate mean latency
  (single-GPU, batch size 1 — the example trains and infers per-run).
- **One-shot model:** the primary drop_test config is GeoTransolver one-shot
  (single forward yields all time steps), so timing is a single forward per
  run; no autoregressive per-step timing complexity.
- **Baseline source:** per-run Radioss wall-times are recorded by the user in
  `conf/baseline.yaml` (from their solver runs); the benchmark divides them
  into measured inference times. No solver execution happens in this case.
  Baseline times are informational inputs, not measured here — the report
  labels them `user_recorded`.

## Error Handling & Edge Cases

| Situation | Behavior |
|---|---|
| Checkpoint path missing / not `.mdlus` | Fail fast; instruct to train drop_test first or pass `--ckpt` |
| Test directory has no `.vtu` | Fail fast; point to Curator ETL docs |
| `baseline.yaml` missing a run's Radioss time | Speedup is `null` for that run; aggregate uses mean over available runs, noted in report |
| GPU unavailable | Fall back to CPU; report tagged `device: cpu` with a note that speedup is not comparable |
| Single-run inference OOM | Skip run, record in `skipped`, continue |
| Timing instability (std/mean > 10%) | Warning flag in report |

## Testing Strategy

Following repo conventions (`examples/**/test_*.py` collected by pytest,
`S101` allowed there):

- **Synthetic smoke tests** (`tests/test_benchmark.py`): build a micro
  tetrahedral VTU with pyvista + a fake-checkpoint scenario; monkeypatch the
  model forward to identity; verify the core paths: metric correctness on
  known answers, report generation (JSON schema fields present), speedup
  arithmetic. No real data or GPU required.
- **Metric unit validation:** accuracy metrics asserted against analytically
  known values (e.g., prediction = truth × 2 → known relative L2), guarding
  against formula errors.
- **Real-data end-to-end:** not CI-run (needs data + GPU); full acceptance
  commands recorded in README.

## Non-Goals

- No new model architectures or training changes (drop_test owns those).
- No dataset generation (PhysicsNeMo-Curator owns ETL).
- No multi-GPU benchmarking (single T4 target; DDP wiring exists upstream if
  ever needed).
- No autoregressive-rollout timing (one-shot model only).

## References

- `examples/structural_mechanics/drop_test/README.md` — scenario, data prep, configs
- `examples/structural_mechanics/drop_test/inference.py` — worker being reused
- PhysicsNeMo-Curator drop_test ETL recipe — VTU generation
- Repo `AGENTS.md` — example conventions (README headings, markdownlint, license headers)

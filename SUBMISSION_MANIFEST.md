# Submission manifest

Where every submission artifact is, which requirement it serves, how to regenerate it, and its
status. Statuses and evidence kinds as in `docs/compliance_matrix.md` (SAMPLE / PSEUDO-GT /
GT-FREE / SYNTHETIC). **No physical benchmark exists: no iPhone was available.**
Verified engineering baseline: git tag `engineering-baseline-v1`.

## AVAILABLE NOW

| Artifact | Requirement | Path | Purpose | Reproduce | Status |
|---|---|---|---|---|---|
| README | D3 | `README.md` | install → weights → one command per tier → evaluate → reproduce | `scripts/clean_clone_audit.sh <dir>` | PARTIAL (clean-clone audit on the development Mac, `reports/repro_audit.md`) |
| Compliance matrix | D1 | `docs/compliance_matrix.md` | requirement → path → evidence → status | — | PASS |
| Capture protocol (Route 2) | C1, D2 | `docs/capture_protocol.md` | one-page operator protocol + evidence status of each rule | — | PARTIAL (not field-tested) |
| Device matrix | C2, D2 | `docs/device_matrix.md` | tier × hardware × honest accuracy | — | PARTIAL |
| Output schema | O8 | `schema/plan.schema.json` (from `src/roomscan/model.py`) | published JSON contract; every measurement has an interval | `uv run python -c "import json; from roomscan.model import Plan; print(json.dumps(Plan.model_json_schema(), indent=1))"` | PASS |
| LiDAR pipeline | C5, O1–O2, O6–O9 | `src/roomscan/lidar/` | Stray export → plan.json + plan.png | `uv run roomscan-lidar <stray_dir> out/` | PARTIAL |
| Photo pipeline | C3 | `src/roomscan/photo/` | per-room photo folders → plan (rooms unplaced) | `uv run roomscan-photo <dir> out/` | PARTIAL |
| Video pipeline | C4 | `src/roomscan/video/` | plain clip → plan (fragmented, unplaced) | `uv run roomscan-video <clip> out/` | PARTIAL |
| Damage / concealed rules / scope | O3–O5 | `src/roomscan/damage.py` | observations → regions, rule flags, scope items | `uv run roomscan-damage plan.json obs.json out.json` | PARTIAL (mechanism; no detector) |
| Input inspection | W1 | `src/roomscan/inspect.py` | format and convention checks on real files | `uv run roomscan-inspect <path>` | PASS (tested on sample + synthetic files) |
| Evaluator | D4, D5 | `src/roomscan/bench/` | site → execute pipelines → gates, coverage, figures; leakage guard | `uv run roomscan-bench benchmark/<site> --run runs/<id> --execute` | PASS (machinery) |
| Benchmark report | D5 | `reports/benchmark_report.md`, `reports/data/` | PSEUDO-GT / GT-FREE / SYNTHETIC results, timing, failures | `uv run python reports/build_reports.py --regenerate` | PARTIAL (no real benchmark) |
| Fix-loop bundle | F1, F2, D6 | `reports/fix_loop/` | declaration, before/after runs, diff | `reports/fix_loop/run.sh` | PARTIAL (synthetic cycle; real gate BLOCKED) |
| Technical report | D7 | `reports/technical_report.md` | ≤ 6-page engineering argument | — | PASS (document) |
| Reproduction audit | D3, D4, K1 | `reports/repro_audit.md`, `scripts/clean_clone_audit.sh` | clean clone → install → weights → intermediates → offline runs → evaluator → schema validation | `scripts/clean_clone_audit.sh <dir> [--weights-from <hf cache>]` | PASS on the development Mac |
| Environment recipes | D3, K2 | `envs/`, `scripts/setup_envs.sh`, `scripts/fetch_weights.py`, `docs/reproduction.md` | isolated envs + pinned weights | `scripts/setup_envs.sh mapanything`; `uv run python scripts/fetch_weights.py` | PASS |
| Tests | D4 | `tests/` | conventions, contract, smoke end-to-end, repeatability, quality, inputs, damage, dev site | `uv run pytest -q` (+ `ROOMSCAN_GPU_TESTS=1`) | PASS |
| Experiments e00–e27 | process, evidence | `experiments/`, `experiments/DEPENDENCIES.md` | every design decision's evidence, incl. negative results | per experiment README | PASS (as evidence) |
| Engineering log, handover | P1 | `ENGINEERING_LOG.md`, `HANDOVER.md` | reasoning and state | — | PASS |
| Development-capture plan | B0 (preparation) | `docs/development_capture_protocol.md`, `benchmark/_template_dev/` | 30-min device session with pre-registered readings | `uv run roomscan-bench benchmark/dev_<date> --run runs/dev_<date> --execute` | READY |

## BLOCKED BY PHYSICAL DEVICE

| Requirement | What is missing | Prepared |
|---|---|---|
| B1–B5 benchmark set (multi-room ×3 tiers, damage room, repeat captures, laser/tape GT) | captures and measurements | site format, evaluator, `docs/physical_benchmark_procedure.md` |
| G1 openings, G2 ceiling, G3 repeatability, G5 photo stitch, G6 photo walls, G7 video walls, G8 calibration | real GT | metrics implemented and tested |
| H1 head-to-head vs a consumer app | app export on 2 rooms | `Incumbent` table support |
| F1/F2 fix loop on the worst real gate | a real failing gate | synthetic cycle shipped (`reports/fix_loop/`) |
| D8 raw benchmark data | everything | — |
| Real-file validation (current Stray export, iPhone HEIC, MOV) | real files | `roomscan-inspect`, synthetic-file tests |
| Walk-in test readiness on real captures | an unseen real capture | all three tiers run cold |
| Field test of the capture protocol | a non-engineer following it | `docs/capture_protocol.md` |

## NOT IMPLEMENTED

- Photo-tier whole-property stitching (rooms are emitted unplaced; no working method found, e17).
- Continuous video tracking / video room placement (rooms per coherent window, unplaced).
- Damage detection (only the rule/scope mechanism exists); window detection; doors with leaves.
- Interval calibration (needs laser GT; intervals are `calibrated: false`).
- Route 1 (own iOS app).

## NOT CLAIMED

- Any laser/tape-verified accuracy at any tier.
- Any gate pass on the assessment's benchmark.
- That the photo or video tier measures rooms correctly on the supplied recordings (it does not).
- That PSEUDO-GT numbers (our own LiDAR as reference) or SYNTHETIC numbers are benchmark accuracy.
- That the capture protocol has been field-tested, or that 0.5× ultra-wide photos help (A/B pending).
- That the fix-loop example satisfies Part 4 (it is on synthetic data, declared retrospectively).

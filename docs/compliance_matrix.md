# Compliance matrix

Requirement (from the case study) → evidence path → command → what is demonstrated → what is
missing → status. Audited 2026-10-06 against the repository; supersedes all earlier versions.

**Status:** PASS = demonstrated in this repository as far as the requirement can be met without a
physical benchmark · PARTIAL = implemented, demonstrated only in part, or demonstrated on
non-benchmark data · BLOCKED = needs a physical iPhone capture, laser/tape GT or an app export
that does not exist · NOT STARTED.
**Evidence kinds:** SAMPLE = runs on the supplied Stray captures · PSEUDO-GT = scored against our own
LiDAR reconstruction (~3 % LiDAR/trajectory self-inconsistency; never benchmark accuracy) ·
GT-FREE = internal consistency · SYNTHETIC = rendered exact geometry (tests code, not sensors).
`R` = `uv run python reports/build_reports.py --regenerate` (regenerates `reports/benchmark_report.md`).

## Part 1 — capture route, tiers, device matrix

| # | Requirement | Evidence (path · command) | Demonstrated | Missing | Status |
|---|---|---|---|---|---|
| C1 | Capture route: Route 2 one-page stock-capture protocol (install, walk, duration, avoid, hand-off) | `docs/capture_protocol.md` | written; each rule tagged supported / hypothesis with evidence | never followed by a non-engineer on a device; install time not measured on a phone | PARTIAL |
| C2 | Device matrix: tier × hardware × honest accuracy | `docs/device_matrix.md` | hardware per tier; accuracy stated as PSEUDO-GT / GT-FREE / SYNTHETIC / unknown | real-device accuracy per tier | PARTIAL |
| C3 | Photo tier: 2–8 stills/room, any iPhone 15+, no depth/poses, one folder per room → **same stitched whole-property plan**, wider intervals | `src/roomscan/photo/` · `uv run roomscan-photo <dir> out` · e21, e23, e24 | runs end to end on simulated photo folders (SAMPLE frames, EXIF-only intrinsics); schema-valid plans; intervals ±0.4–1.3 m where geometry is unstable | rooms **not stitched** (`placed: false`); 0 % of walls within ±8 % (PSEUDO-GT); real iPhone photos untested | PARTIAL |
| C4 | Video tier: handheld walkthrough, any iPhone 15+ | `src/roomscan/video/` · `uv run roomscan-video <clip> out` · e20, e25, e24 | runs end to end on the sample clip with self-estimated field of view; fragmentation explicit | rooms unplaced; 0 % of walls within ±3 %, rooms −60…−100 % area by identity (PSEUDO-GT); real MOV untested | PARTIAL |
| C5 | LiDAR tier: depth + poses + intrinsics on Pro devices | `src/roomscan/lidar/` · `uv run roomscan-lidar <stray> out` · e18, e22, synthetic | 7/7 non-empty regions on the sample apartment (SAMPLE); exact synthetic rooms to ~2 mm (SYNTHETIC) | absolute accuracy (laser); captures without an upper-wall sweep give no rooms (flagged) | PARTIAL |
| C6 | Intervals widen honestly as sensor data thins | `src/roomscan/assemble.py`, `quality.py` · R §1, §6 | per-face named error terms; photo/video intervals wider than LiDAR; unstable geometry widened; flags for unreliable rooms | coverage 0–67 % vs 90 % on PSEUDO-GT; no calibration | PARTIAL |

## Part 2 — output contract (per capture)

| # | Requirement | Evidence | Demonstrated | Missing | Status |
|---|---|---|---|---|---|
| O1 | Dimensioned per-room plan: walls | `assemble.py`, tier `build.py` · R §1 | all tiers emit wall lengths with intervals; LiDAR consistent (GT-FREE) and exact on SYNTHETIC | real accuracy; photo/video wrong on PSEUDO-GT | PARTIAL |
| O1b | Ceiling height | `lidar/levels.py`, `cloudroom.py` | LiDAR per room (`not_observed` when the ceiling was not seen); SYNTHETIC −1.2 mm | laser; bimodal ceilings (one level reported) | PARTIAL |
| O1c | Floor area | `assemble.py` | every room, interval from face/scale terms | real accuracy | PARTIAL |
| O1d | Openings | `lidar/openings.py` | LiDAR doorways (lintel + jamb gap); SYNTHETIC doorway +1 cm (1 cm bins) | windows; doors with leaves; photo/video openings | PARTIAL |
| O2 | Stitched multi-room plan, correct adjacency, every room placed | `lidar/layout.py`, `build.py` · R §3 | LiDAR: rooms placed in one frame, doorway adjacency | photo and video rooms **not placed**; adjacency unscored against real GT | PARTIAL |
| O3 | Per-surface damage regions with class and metric extent | `src/roomscan/damage.py` · `uv run roomscan-damage plan.json obs.json out.json` · `tests/test_damage.py` | schema + mechanism: regions keyed to surfaces with area intervals | **no damage detector**; no staged-damage capture | PARTIAL (mechanism only) |
| O4 | Concealed-damage flags with the rule that fired | `damage.py` rules CD1–CD4 | rule id + surfaces + reason, tested | depends on O3 observations; rules unvalidated | PARTIAL (mechanism only) |
| O5 | Scope line items keyed to surfaces | `damage.py` | repair (region) and refinish (surface) items with quantity intervals | pricing/real scope; unvalidated | PARTIAL (mechanism only) |
| O6 | Confidence interval on every measurement | `src/roomscan/model.py` (validator), `schema/plan.schema.json` | schema rejects an observed value without an interval; all tiers comply | intervals uncalibrated (G8) | PASS (contract) |
| O7 | One command per capture | `roomscan-lidar`, `roomscan-photo`, `roomscan-video` · `roomscan-bench --execute` | each tier is one command; the evaluator runs them from raw input | on real iPhone files (BLOCKED) | PASS (SAMPLE/SYNTHETIC) |
| O8 | JSON to a published schema | `schema/plan.schema.json` · `tests/test_smoke_pipeline.py`, `scripts/clean_clone_audit.sh` | every produced plan validates against the schema file | — | PASS |
| O9 | Rendered plan | `src/roomscan/render.py` | `plan.png` for every tier, intervals, inferred walls dashed, unplaced noted | — | PASS |

## Part 2 — benchmark composition and gates (real benchmark required)

| # | Requirement | Evidence | Demonstrated | Missing | Status |
|---|---|---|---|---|---|
| B1 | Multi-room capture (≥ 3 rooms + connector) | `benchmark/_template/` | format, evaluator | the capture | BLOCKED |
| B2 | Furnished room with staged damage (2 classes) | — | — | the capture, damage GT | BLOCKED |
| B3 | Same rooms at all three tiers, photos as per-room folders | — | — | the captures | BLOCKED |
| B4 | One room captured twice at the same tier | `bench/evaluate.py` repeat groups | evaluator support | the captures | BLOCKED |
| B5 | Laser/tape GT on everything; raw data + measurements submitted | `benchmark/README.md` | GT format incl. partial tape | measurements | BLOCKED |
| G1 | Opening widths ≤ 2 cm on ≥ 85 %, misses and phantoms count | `bench/evaluate.py` | metric implemented and tested; SYNTHETIC doorway +1 cm | real openings | BLOCKED |
| G2 | Ceiling ≤ 1.5 cm per room; repeat spread ≤ 1 cm; biased vs unrepeatable | R §2 | GT-FREE half-to-half ceiling agreement (see report) | laser; two captures | BLOCKED |
| G3 | Repeatability: two captures agree ≤ 1 cm or 0.5 % per wall | R §2 | GT-FREE halves of one capture (not two captures) | two real captures | BLOCKED |
| G4 | Drift accountability: method stated + stitched footprint ablation ON/OFF; "poses as-is" fails | R §3 · e09, e19 | method: room-local registration of visits + common-yaw placement; ON/OFF footprint on the sample apartment (GT-FREE); negative results for pose graph and plane-anchored stitch | the ON/OFF ablation shows the stitched footprint barely changes (overlap 0.098 vs 0.093 m², yaw ≤ 0.9°): room **translations are ARKit's**, so placement is at risk of failing "poses as-is"; drift correction acts on room dimensions (per-room visit registration), not placement; no benchmark multi-room capture | PARTIAL (at risk) |
| G5 | Photo whole-property stitch: one plan, correct adjacency, no overlaps, footprint ±8 %, calibrated | e17, e27 | negative result (0/36 doorway links); chain test prepared | a working method | NOT STARTED (no method) / BLOCKED (data) |
| G6 | Photo walls ±8 % with calibrated intervals | R §1 | 0 % within gate (PSEUDO-GT) | — | BLOCKED (real data); failing on PSEUDO-GT |
| G7 | Video walls ±3 % | R §1 | 0 % within gate (PSEUDO-GT) | — | BLOCKED (real data); failing on PSEUDO-GT |
| G8 | Calibration scored at every tier | `bench/evaluate.py` (coverage), `bench/leakage.py` | coverage computed per tier; calibration sources must be named | laser GT; any calibration | BLOCKED |

## Parts 3–5 — head-to-head, fix loop, process

| # | Requirement | Evidence | Demonstrated | Missing | Status |
|---|---|---|---|---|---|
| H1 | LiDAR tier vs a consumer app on 2 rooms, beat/tie ≥ 70 % | `bench/gt.py` `Incumbent`, `bench/evaluate.py` | head-to-head table support | app export, rooms, GT | BLOCKED |
| F1 | Fix declaration: worst failing gate on own benchmark + number, root cause + evidence, fix + predicted number | `reports/fix_loop/DECLARATION.md` | a full diagnose → fix → before/after cycle on SYNTHETIC exact GT (segmentation border erosion) | the real benchmark's worst gate; declaration is retrospective | BLOCKED (benchmark) / PARTIAL (synthetic cycle) |
| F2 | Shipped fix, before/after regenerable, readable diff | `reports/fix_loop/` · `reports/fix_loop/run.sh` | before/after from one code version (`--seg-pad`), `fix.diff`, results | on a real gate | PARTIAL |
| P1 | Process evidence: incremental history | `git log` (tag `engineering-baseline-v1`) | exploration → experiments (incl. failures) → implementation → fixes → hardening, over 30+ commits | — | PASS |

## Deliverables and constraints

| # | Requirement | Evidence | Demonstrated | Missing | Status |
|---|---|---|---|---|---|
| D1 | Compliance matrix | this file | traceable rows | — | PASS |
| D2 | Capture route + device matrix | `docs/capture_protocol.md`, `docs/device_matrix.md` | see C1, C2 | field test | PARTIAL |
| D3 | README: fresh machine → running on a fresh capture < 15 min, one command per capture | `README.md`, `scripts/clean_clone_audit.sh` | clean-clone audit on the development Mac (see `reports/repro_audit.md`) | a truly separate machine; a fresh real capture | PARTIAL |
| D4 | Reproduction bundle regenerates every reported number from raw inputs; deterministic cache + live path | R; `docs/reproduction.md`; `experiments/DEPENDENCIES.md` | every report number from raw sample inputs; model outputs bitwise repeatable, LiDAR ≤ 1e-10 m (e26) | raw benchmark inputs (do not exist) | PARTIAL |
| D5 | Benchmark report: gates at all tiers, repeatability, head-to-head, timing | `reports/benchmark_report.md` | PSEUDO-GT / GT-FREE / SYNTHETIC results, timing, failures visible | real gates; head-to-head | PARTIAL |
| D6 | Fix-loop bundle | `reports/fix_loop/` | see F1/F2 | real gate | PARTIAL |
| D7 | Technical report ≤ 6 pages | `reports/technical_report.md` | architecture, tiers, drift, error budget, calibration status, fix loop, failure modes | real results | PASS (document) |
| D8 | Raw benchmark data (sensor logs, GT, app exports) | — | — | everything | BLOCKED |
| W1 | Walk-in: all three tiers ready to run cold on an unseen capture | three CLIs, `roomscan-inspect` | each tier runs cold on inputs it has not seen (sample, synthetic) | real iPhone inputs; photo/video accuracy | PARTIAL |
| K1 | Runs without calling our infrastructure; pretrained models disclosed | `envs/weights.yaml`; offline run in `reports/repro_audit.md` | offline (`HF_HUB_OFFLINE=1`) runs of all tiers; models, revisions, licences listed | — | PASS |
| K2 | Weights and large binaries fetched by script | `scripts/fetch_weights.py`, `scripts/setup_envs.sh` | pinned manifest, fetch + offline check | — | PASS |
| K3 | Mirrors, glass, wet-look surfaces, low light covered | `docs/capture_protocol.md` (avoid list, evidence table), technical report §6 | known effects documented (mirror bathroom breaks MapAnything scale, e16; glass LiDAR artefacts, e00) | no targeted captures or handling | PARTIAL |

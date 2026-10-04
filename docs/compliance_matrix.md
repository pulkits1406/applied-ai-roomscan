# Compliance matrix

Requirement → file path → artifact → status. Status values: `not started`, `exploring`,
`in progress`, `done (evidence: …)`. "Evidence needed" is what an evaluator must be able to
regenerate to accept the row.

Last updated: 2026-10-05 (phase 2: prototypes; all numbers on supplied data are pseudo-GT or GT-free).

## Capture route and tiers

| # | Requirement | Evidence needed | File path | Artifact | Status |
|---|---|---|---|---|---|
| C1 | Capture route: Route 1 app build **or** Route 2 one-page protocol | Protocol a non-engineer follows literally; install < 10 min | `docs/capture_protocol_hypotheses.md` | hypotheses with evidence | exploring (Route 2 provisional; final page blocked on device pre-check) |
| C2 | Device matrix: tier × hardware × honest accuracy | Measured per-tier error on benchmark | `docs/device_matrix.md` | table | not started |
| C3 | Photo tier: 2–8 stills/room, any iPhone 15+, no depth/poses → full contract | Runs on per-room folders | `experiments/e12_photo_scale`, `e13_multiview_photo` | scale studies | exploring (scale ≈ 6 % median / 15 % p90 per room, pseudo-GT) |
| C4 | Video tier: handheld walkthrough, any iPhone 15+ | Runs on a single clip | `experiments/e08_video_sfm`, `e11_imu_scale` | SfM + IMU-scale studies | exploring (IMU scale ±2 % on ARKit poses; RGB-only tracking fragments → main blocker) |
| C5 | LiDAR tier: depth + poses + intrinsics | Runs on a Stray export | `src/roomscan/lidar_pipeline.py` | v0 pipeline → Plan JSON | in progress (runs on `with_ceiling`; fails without upper-wall sweep) |
| C6 | Intervals widen honestly as data thins | Interval width ordered photo > video > LiDAR, with coverage measured | — | — | not started |

## Output contract (per capture, every tier)

| # | Requirement | Evidence needed | File path | Artifact | Status |
|---|---|---|---|---|---|
| O1 | Per-room plan: walls, ceiling height, floor area, openings | JSON + render | `src/roomscan/lidar_pipeline.py` | v0 (LiDAR only, no render) | in progress |
| O2 | Stitched multi-room plan, correct adjacency, no overlaps | Rendered whole-property plan | — | — | not started |
| O3 | Per-surface damage regions: class + metric extent | JSON regions keyed to surfaces | — | — | not started |
| O4 | Concealed-damage flags with the rule that fired | Rule id in JSON | — | — | not started |
| O5 | Scope line items keyed to surfaces | JSON | — | — | not started |
| O6 | Confidence interval on **every** measurement | Schema validation enforces it | `src/roomscan/model.py` | schema rejects observed values without interval | in progress (intervals uncalibrated) |
| O7 | One command per capture | README command | — | — | not started |
| O8 | JSON to a published schema | `schema/*.json` + validation test | `schema/plan.schema.json` | v0.1 | in progress |
| O9 | Rendered plan | PNG/SVG/PDF | — | — | not started |

## Gates

| # | Gate | Threshold | Evidence needed | File path | Status |
|---|---|---|---|---|---|
| G1 | Opening widths | ≤ 2 cm on ≥ 85 % of openings; misses + phantoms count | Per-opening table vs tape | — | not started |
| G2 | Ceiling height | ≤ 1.5 cm/room; repeat spread ≤ 1 cm; state biased vs unrepeatable | Per-room table vs laser | — | exploring (e04) |
| G3 | Repeatability | Same-tier repeat captures agree ≤ 1 cm or 0.5 % per wall | Per-wall diff table | `experiments/e14_lidar_v0`, `e15` | at risk: within-capture proxy 50 % pass, 4/7 rooms unstable topology |
| G4 | Drift accountability | Method stated + footprint ablation on/off; "poses as-is" fails | Ablation figure + numbers | `experiments/e09_drift_posegraph` | exploring (pose graph: negative result; room-local + plane-anchored chosen provisionally) |
| G5 | Photo-tier whole-property stitch | One plan, correct adjacency, no overlaps, footprint ±8 %, calibrated | Stitched photo-tier plan vs GT | — | not started |
| G6 | Photo-tier wall lengths | ±8 % with calibrated intervals | Per-wall table + coverage | — | exploring (e06) |
| G7 | Video-tier wall lengths | ±3 % | Per-wall table + coverage | — | not started |
| G8 | Calibration at every tier | Empirical coverage of stated intervals | Coverage table per tier | `src/roomscan/bench/evaluate.py` | harness computes coverage; v0 LiDAR intervals cover 32–77 % vs 90 % nominal (overconfident) |

## Benchmark set (we build it)

| # | Requirement | Evidence needed | File path | Status |
|---|---|---|---|---|
| B1 | Multi-room capture, ≥ 3 rooms + connector | Raw data + GT | `benchmark/` | not started (needs physical capture) |
| B2 | Furnished room with staged damage, ≥ 2 classes | Raw data + GT damage extents | `benchmark/` | not started (needs physical capture) |
| B3 | Same rooms at all three tiers (multi-room incl.; photo as per-room folders) | Three tier captures per room | `benchmark/` | not started (needs physical capture) |
| B4 | ≥ 1 room captured twice at same tier | Two captures | `benchmark/` | not started (needs physical capture) |
| B5 | Laser/tape GT on everything; raw sensor data submitted | Measurement sheet | `benchmark/ground_truth/` | not started (needs physical work) |

## Comparison, fix loop, process, deliverables

| # | Requirement | Evidence needed | File path | Status |
|---|---|---|---|---|
| H1 | Head-to-head vs consumer app on 2 rooms, LiDAR tier; beat/tie ≥ 70 % of shared dims | App name+version, its export, one table | `benchmark/incumbent/` | not started (candidate: magicplan free tier) |
| F1 | Fix declaration: worst gate + failing number, root cause + evidence, fix + predicted number | 1-page doc | `docs/fix_loop.md` | not started |
| F2 | Shipped fix, before/after regenerable, readable diff | Two runs + diff | — | not started |
| P1 | Incremental commit history | git log | — | in progress |
| D1 | README: fresh machine → running < 15 min, one command per capture | Timed clean install | `README.md` | not started |
| D2 | Reproduction bundle regenerates every number from raw inputs; cache deterministic + live path runs | Script + cache | `src/roomscan/bench/`, `benchmark/README.md` | `roomscan-bench` one command per site | in progress (harness done; pipeline execution step pending) |
| D3 | Benchmark report: all tiers, repeatability, head-to-head, timing | Report | — | not started |
| D4 | Technical report ≤ 6 pages | PDF | — | not started |
| D5 | Raw benchmark data: sensor logs, GT, app exports | Data bundle | — | not started |
| K1 | Runs without calling our infrastructure; weights fetched by script; pretrained models disclosed | Offline run | — | not started |
| K2 | Cover mirrors, glass, wet-look surfaces, low light | Failure-mode section + handling | — | exploring (glass shower + windows present in samples) |
| W1 | Walk-in: all three tiers ready to run cold on an unseen capture | Dry runs on unseen spaces | — | not started |

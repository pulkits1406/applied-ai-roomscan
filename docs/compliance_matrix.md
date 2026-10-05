# Compliance matrix

Requirement → file path → artifact → status. Status values: `not started`, `exploring`,
`in progress`, `done (evidence: …)`. "Evidence needed" is what an evaluator must be able to
regenerate to accept the row.

Last updated: 2026-10-05 (phase 4; all numbers on supplied data are pseudo-GT, GT-free or synthetic).

## Capture route and tiers

| # | Requirement | Evidence needed | File path | Artifact | Status |
|---|---|---|---|---|---|
| C1 | Capture route: Route 1 app build **or** Route 2 one-page protocol | Protocol a non-engineer follows literally; install < 10 min | `docs/capture_protocol_hypotheses.md` | hypotheses with evidence | exploring (Route 2 provisional; final page blocked on device pre-check) |
| C2 | Device matrix: tier × hardware × honest accuracy | Measured per-tier error on benchmark | `docs/device_matrix.md` | table | not started |
| C3 | Photo tier: 2–8 stills/room, any iPhone 15+, no depth/poses → full contract | Runs on per-room folders | `src/roomscan/photo/` (`roomscan-photo`), e21, e23, e24 | production frontend → Plan JSON (rooms unplaced) | in progress, at risk: runs end to end; scale ensemble 5.3 % median (e21) but room geometry wrong on simulated photos (0 % walls within ±8 %, e24) — joint poses need overlapping photos (e23); protocol untested on a device |
| C4 | Video tier: handheld walkthrough, any iPhone 15+ | Runs on a single clip | `src/roomscan/video/` (`roomscan-video`), e20, e25, e24 | production frontend → Plan JSON (fragmented, unplaced) | in progress, at risk: runs on a plain clip (FOV self-estimated); 3/18 windows coherent; rooms −60…−100 % area by identity (e24); rooms in tracked pieces fail (e25) |
| C5 | LiDAR tier: depth + poses + intrinsics | Runs on a Stray export | `src/roomscan/lidar/` (`roomscan-lidar`) | v1.1 pipeline → Plan JSON + PNG | in progress: 7/7 non-empty regions on `with_ceiling` (e22 ladder); synthetic two-room capture within 2 mm; captures without upper-wall sweep fail (flagged) |
| C6 | Intervals widen honestly as data thins | Interval width ordered photo > video > LiDAR, with coverage measured | `src/roomscan/assemble.py`, e24 | per-tier sigmas, inferred faces widen | exploring: widths ordered photo ≈ video > LiDAR; coverage 0–67 % (pseudo-GT, e24) — topology failures are not in the sigmas |

## Output contract (per capture, every tier)

| # | Requirement | Evidence needed | File path | Artifact | Status |
|---|---|---|---|---|---|
| O1 | Per-room plan: walls, ceiling height, floor area, openings | JSON + render | `src/roomscan/assemble.py`, `render.py`, tier `build.py` | all three tiers (openings: LiDAR doorways only, no windows) | in progress |
| O2 | Stitched multi-room plan, correct adjacency, no overlaps | Rendered whole-property plan | — | — | not started |
| O3 | Per-surface damage regions: class + metric extent | JSON regions keyed to surfaces | — | — | not started |
| O4 | Concealed-damage flags with the rule that fired | Rule id in JSON | — | — | not started |
| O5 | Scope line items keyed to surfaces | JSON | — | — | not started |
| O6 | Confidence interval on **every** measurement | Schema validation enforces it | `src/roomscan/model.py` | schema rejects observed values without interval | in progress (intervals uncalibrated) |
| O7 | One command per capture | README command | `roomscan-lidar`, `roomscan-photo`, `roomscan-video` | CLIs; `roomscan-bench --execute` runs them | done (evidence: e24 execution log, `tests/test_smoke_pipeline.py`) |
| O8 | JSON to a published schema | `schema/*.json` + validation test | `schema/plan.schema.json` | v0.1 | in progress |
| O9 | Rendered plan | PNG/SVG/PDF | `src/roomscan/render.py` | dimensioned PNG, intervals, inferred walls dashed, unplaced noted (all tiers) | in progress |

## Gates

| # | Gate | Threshold | Evidence needed | File path | Status |
|---|---|---|---|---|---|
| G1 | Opening widths | ≤ 2 cm on ≥ 85 % of openings; misses + phantoms count | Per-opening table vs tape | — | not started |
| G2 | Ceiling height | ≤ 1.5 cm/room; repeat spread ≤ 1 cm; state biased vs unrepeatable | Per-room table vs laser | `src/roomscan/lidar/levels.py`, e18, e22 | in progress: production code 50–60 % of rooms ≤ 1 cm half-to-half (GT-free, e22; e18 experiment code 75 %); bimodal ceiling (room 3) flips; synthetic −1.2 mm |
| G3 | Repeatability | Same-tier repeat captures agree ≤ 1 cm or 0.5 % per wall | Per-wall diff table | `experiments/e18_lidar_repeat`, `e22_lidar_small_rooms` | at risk: production code 65 % of walls within gate with half data (e22; e18 experiment code 73 %); inter-visit misregistration 30–160 mm raw (e22) |
| G4 | Drift accountability | Method stated + footprint ablation on/off; "poses as-is" fails | Ablation figure + numbers | `experiments/e09_drift_posegraph`, `e19_drift_stitch` | in progress: room-local measurement + yaw regularisation; translation stitch negative; ablation figure exists (GT-free) |
| G5 | Photo-tier whole-property stitch | One plan, correct adjacency, no overlaps, footprint ±8 %, calibrated | Stitched photo-tier plan vs GT | `experiments/e17_photo_stitch` | at risk: doorway photos 0/36 successful links; overlapping-chain protocol untested |
| G6 | Photo-tier wall lengths | ±8 % with calibrated intervals | Per-wall table + coverage | e21, e23, e24 | at risk: scale 64 % of runs within ±8 % (e21) but walls 0 % within gate on simulated folders (e24): registration within a room is the binding problem |
| G7 | Video-tier wall lengths | ±3 % | Per-wall table + coverage | e20, e25, e24 | at risk: 0 % within gate on the sample clip (pseudo-GT, e24); needs a protocol-following clip |
| G8 | Calibration at every tier | Empirical coverage of stated intervals | Coverage table per tier | `src/roomscan/bench/evaluate.py` | harness computes coverage per tier; pseudo-GT coverage 0–67 % vs 90 % (e24); calibration needs laser GT |

## Benchmark set (we build it)

| # | Requirement | Evidence needed | File path | Status |
|---|---|---|---|---|
| B1 | Multi-room capture, ≥ 3 rooms + connector | Raw data + GT | `benchmark/` | not started (needs physical capture) |
| B2 | Furnished room with staged damage, ≥ 2 classes | Raw data + GT damage extents | `benchmark/` | not started (needs physical capture) |
| B3 | Same rooms at all three tiers (multi-room incl.; photo as per-room folders) | Three tier captures per room | `benchmark/` | not started (needs physical capture) |
| B4 | ≥ 1 room captured twice at same tier | Two captures | `benchmark/` | not started (needs physical capture) |
| B5 | Laser/tape GT on everything; raw sensor data submitted | Measurement sheet | `benchmark/<site>/site.yaml` + `measurements/` | not started (needs physical work) |

## Comparison, fix loop, process, deliverables

| # | Requirement | Evidence needed | File path | Status |
|---|---|---|---|---|
| H1 | Head-to-head vs consumer app on 2 rooms, LiDAR tier; beat/tie ≥ 70 % of shared dims | App name+version, its export, one table | `benchmark/incumbent/` | not started (candidate: magicplan free tier) |
| F1 | Fix declaration: worst gate + failing number, root cause + evidence, fix + predicted number | 1-page doc | `docs/fix_loop.md` | not started |
| F2 | Shipped fix, before/after regenerable, readable diff | Two runs + diff | — | not started |
| P1 | Incremental commit history | git log | — | in progress |
| D1 | README: fresh machine → running < 15 min, one command per capture | Timed clean install | `README.md`, `docs/reproduction.md` | in progress: env recipes rebuilt from scratch (39 s / 33 s warm cache); full clean-machine run not timed |
| D2 | Reproduction bundle regenerates every number from raw inputs; cache deterministic + live path runs | Script + cache | `src/roomscan/bench/`, `scripts/prepare_intermediates.sh`, `experiments/DEPENDENCIES.md` | `roomscan-bench --execute` raw → pipeline → score | in progress: execute path done for all tiers; uncached video runs not bit-stable (e24) |
| D3 | Benchmark report: all tiers, repeatability, head-to-head, timing | Report | — | not started |
| D4 | Technical report ≤ 6 pages | PDF | — | not started |
| D5 | Raw benchmark data: sensor logs, GT, app exports | Data bundle | — | not started |
| K1 | Runs without calling our infrastructure; weights fetched by script; pretrained models disclosed | Offline run | `envs/weights.yaml`, `scripts/fetch_weights.py` | pinned manifest + offline `--check` | in progress: 8/8 weights pinned and verified; HF_HUB_OFFLINE end-to-end run not yet done |
| K2 | Cover mirrors, glass, wet-look surfaces, low light | Failure-mode section + handling | — | exploring (glass shower + windows present in samples) |
| W1 | Walk-in: all three tiers ready to run cold on an unseen capture | Dry runs on unseen spaces | — | not started |

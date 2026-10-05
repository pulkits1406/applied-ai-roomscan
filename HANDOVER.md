# HANDOVER — state of the project at compaction time

Written 2026-10-05 from the actual repository (commit `1b1e53e` on `master`, clean tree), the
experiment outputs and the engineering log. This file is authoritative for *state*; the reasoning
behind every item is in `ENGINEERING_LOG.md` (phases 1–3) and the per-experiment READMEs.

**Status vocabulary used below.** IMPLEMENTED = production code under `src/` that runs end to end ·
PARTIAL = production code that covers part of the requirement · EXPERIMENT ONLY = code under
`experiments/` that answers a question but is not a product path · NOT STARTED.
**Evidence labels.** FACT (verified, not an estimate) · MEASURED (number from our code on the
supplied data, GT-free consistency) · PSEUDO-GT (compared against our own LiDAR/ARKit
reconstruction) · HYPOTHESIS · PROVISIONAL (decision) · BLOCKED (needs a physical device or a
user decision).

> **The pseudo-GT caveat applies to every accuracy number in this repo.** There is no laser/tape
> ground truth yet. The supplied ARKit LiDAR depth and the ARKit trajectory disagree with each
> other by ~3 % (LiDAR reads ~3 % deeper than depth triangulated from ARKit poses, e08). Nothing
> below 3 % absolute scale can be established from the current data, and **no number in this
> repository is a benchmark (laser) result.**

---

## 1. Project / assessment summary

**What we build** (assessment PDF `Applied_AI_Case_Study.pdf`, gitignored, at repo root): a
pipeline from iPhone captures to a **stitched, dimensioned whole-property floor plan**
(magicplan/Polycam-like) with walls, ceiling height, floor area, openings, damage regions,
concealed-damage flags, scope line items, a confidence interval on **every** measurement, JSON to
a published schema, a rendered plan, one command per capture.

**Three mandatory input tiers, same output contract:** Photo (2–8 stills per room, one folder per
room, any iPhone 15+, *no depth, no poses*; must still stitch the whole property) · Video (a
handheld walkthrough clip, any iPhone 15+) · LiDAR (depth + poses + intrinsics, Pro iPhones).
Intervals must widen honestly as input thins; **calibration is scored at every tier and
"confident garbage on thin input caps your total score"**.

**Capture route:** Route 1 = own iOS app (TestFlight/dev build); Route 2 = named App Store app(s)
+ a one-page protocol a non-engineer follows *literally*.

**Hard gates:** opening widths ≤ 2 cm on ≥ 85 % of openings (misses *and* phantoms count) ·
ceiling height ≤ 1.5 cm per room, repeat spread ≤ 1 cm, report must say biased vs unrepeatable ·
repeatability: two same-tier captures agree within 1 cm or 0.5 % per wall · drift
accountability: method + an ablation of the stitched footprint with it on/off ("poses as-is" =
automatic fail) · photo tier: per-room folders → one stitched plan, correct adjacency, no
overlaps, footprint ±8 % with calibrated intervals · photo walls ±8 %, video walls ±3 %.

**Scoring:** walk-in test 30 % (evaluator captures an unseen space with their own iPhone 15+,
picks the tier on the day, follows our protocol; we run cold; laser-scored live) · **fix loop 25 %**
(declare worst failing gate + number, root cause + evidence, ship fix + predicted number;
before/after both regenerable; analysis without shipped fix = 0) · verified benchmark accuracy
15 % · compliance matrix 10 % · head-to-head vs a consumer app (beat/tie ≥ 70 % of shared
dimensions on 2 rooms, LiDAR tier) 10 % · capture-route quality 5 % · commit history 5 %.

**Engineering implications:** 55 % of the score is robustness on unseen captures + a measured
diagnose→repair loop, not feature count. The protocol directly drives walk-in input quality. The
fix loop needs a *real failing gate* on *our own laser benchmark* → the benchmark harness exists
already; the benchmark data does not. Over-confident intervals are penalised harder than wide
honest ones.

**Benchmark composition we must build:** a multi-room capture (≥ 3 rooms + a connector) at all
three tiers (photo as per-room folders); one furnished room with staged damage (≥ 2 classes); one
room captured twice at the same tier; laser/tape GT on everything; raw data submitted.

---

## 2. Repository state

```
.
├── HANDOVER.md                 this file
├── ENGINEERING_LOG.md          reasoning record, phases 1–3, every experiment with labels
├── README.md                   setup + run commands
├── pyproject.toml / uv.lock    uv project "zingly-scan"; groups: dev, ml (both default)
├── .gitignore                  raw data, zips, PDF, cache/, runs/, *.ply, *.npy, *.npz
├── src/roomscan/               production package (see §3)
│   ├── stray.py                Stray Scanner loader (verified conventions)
│   ├── geometry.py             2-D grid fusion/segmentation helpers (shared by v0 and v1)
│   ├── lidar_pipeline.py       LiDAR v0 prototype — FROZEN baseline, do not edit
│   ├── lidar/                  LiDAR v1 production pipeline (staged modules + CLI)
│   ├── model.py                output representation (pydantic) → schema/plan.schema.json
│   └── bench/                  benchmark site format, matching, gates, evaluator CLI
├── schema/plan.schema.json     published JSON schema (generated from model.py)
├── tests/                      13 tests (stray conventions, evaluator, synthetic walls)
├── benchmark/
│   ├── README.md               drop-in benchmark format + GT measurement conventions
│   ├── _template/site.yaml     example site (laser GT format)
│   └── pseudo_gt_stray_apartment/site.yaml   PSEUDO-GT site generated by e15 (v0 pipeline)
├── docs/
│   ├── compliance_matrix.md               requirement → path → status (see §7 for reconciliation)
│   ├── capture_protocol_hypotheses.md     protocol rules with evidence status (NOT the final protocol)
│   └── physical_benchmark_procedure.md    borrowed-iPhone session plan, captures grouped by purpose
├── experiments/e00…e21/        one dir per experiment; scripts regenerate out/; READMEs for e08, e13, e16–e21
└── (ignored) single_room/ single_scan_floor_only/ single_scan_with_ceiling/  raw Stray captures (unzipped)
    (ignored) *.zip, Applied_AI_Case_Study.pdf, cache/, runs/
```

**Raw data (FACT):** three Stray Scanner exports of **one apartment**, one device, one evening, no
GT/labels. `single_room` (37 s; *misnamed*: living/kitchen + bathroom; never looks above ~1.7 m),
`single_scan_floor_only` (115 s, whole apartment ~6 rooms + corridor, no upper walls/ceiling),
`single_scan_with_ceiling` (215 s, whole apartment, ceiling swept, revisits; the only capture the
v1 pipeline can segment). Zips (`*.zip`) and the extracted dirs sit at the repo root, gitignored.

**Generated / cached artifact locations (all gitignored, all regenerable):**

| Path | Size | Produced by | Used by |
|---|---|---|---|
| `cache/frames/<capture>/` | 223 MB | `experiments/e07_frame_cache/extract.py` | e08, e12, e13, e16, e17, e20 |
| `experiments/e02_fusion/out/*_vox2cm.ply` | 91 MB | `e02_fusion/fuse.py` | e03, e04, e10, e17 doorways, e12 run_moge |
| `experiments/e10_room_seg/out/*_labels.npy` | small | `e10_room_seg/segment.py`, `segment_v2.py` | e12 photo_sets, e16, e17 |
| `cache/e09/` | 4 MB | e09 pose-graph corrected poses | e09 ablation |
| `cache/e12/<row>.npy` | 35 MB | `e12_photo_scale/run_moge.py` (MoGe depth) | e12 cues, e16 MoGe comparison |
| `cache/e08/` | 5.7 GB | e08 COLMAP databases/models + MoGe | e08 only; deletable |
| `cache/e13/` | 7.9 GB | MapAnything isolated venv + source + weights (`hf/hub/blobs`, 4.6 GB) + MLX/t28 venvs | e13, e16, e17, e20 MapAnything |
| `cache/e16/`, `cache/e17/` | 63 / 15 MB | cached MapAnything per-run predictions | e16/e17 re-analysis |
| `cache/e20/` | 2.7 GB | isolated venv (LightGlue/kornia), features, matches, MoGe depth (all 818 with_ceiling keyframes), poses, COLMAP | e20 |
| `runs/v0/`, `runs/v1/`, `runs/pseudo_v0/` | <1 MB | LiDAR v0/v1 plans, e15 harness run | inspection, e15 |
| `~/.cache/huggingface/hub` | — | MoGe-2 ViT-L, Grounding DINO tiny, Depth Anything V2 metric indoor small | project venv |

**Entry points (VERIFIED working at `1b1e53e`):** `roomscan-lidar` → `roomscan.lidar.cli:main`;
`roomscan-bench` → `roomscan.bench.evaluate:main`; `python -m roomscan.lidar_pipeline` (v0).

---

## 3. CURRENT IMPLEMENTATION

### LiDAR — IMPLEMENTED (v1), PARTIAL against the contract
- **Entry point:** `uv run roomscan-lidar <stray_dir> <out_dir> [--placement arkit|arkit_yaw]` →
  `plan.json` (schema-valid), `plan.png` (dimensioned render), `debug.json`.
- **Stages** (`src/roomscan/lidar/`):
  1. `fusion.py` — per-frame world points with **camera-facing normals** computed on the depth grid
     (oriented toward the observing camera; grazing/edge normals rejected by |cos| > 0.15);
     `Selection` presets `ALL` (≤ 4.5 m) and `CLOSE` (≤ 2.5 m, |cos| > 0.5); voxel 2 cm.
  2. `rooms.py` — `prepare()`: fuse every 5th frame, `geometry.segment_rooms` (2-D 5 cm grid;
     barrier = vertical-normal points above `H_STRUCT` = 1.9 m above floor, so door lintels close
     doorways; interior bounded by the observed-floor footprint); camera→room label per frame;
     `visits()` = contiguous camera-in-room spans; `room_cloud()` fuses each visit, ICP-aligns
     visits to the longest one (point-to-plane, 5 cm then 2 cm), **keeps a visit unaligned
     (flagged) if ICP fails rather than dropping it**, and returns an `all` and a `close` cloud.
  3. `walls.py` (e18 variant V6) — Manhattan yaw from horizontal normals; wall-face candidates =
     density peaks per facing side whose support spans ≥ 1.2 m and reaches ≥ 2.0 m; room polygon =
     union of line-arrangement cells covered by the room's free space, grown across edges *not*
     covered by structural points facing into the room; wall positions = median of interior-facing
     points within ±5 cm in the 0.9 m…(ceiling − 0.1 m) band and the edge's own extent, taken from
     the `close` cloud when ≥ 60 points support it, else from `all`, else "unmeasured".
  4. `levels.py` — floor and dominant ceiling level from horizontal-normal points inside the
     polygon shrunk by 0.2 m; ceiling `not_observed` if < 300 points.
  5. `openings.py` — doorway detection = lintel present, no wall at 0.3–1.8 m, floor below,
     touching exactly two rooms; width = jamb-to-jamb gap in each room's own mid-height wall
     points (1 cm bins), averaged over the rooms that could measure it; falls back to the 5 cm
     grid width with a ±5 cm interval and confidence 0.3. **No windows** (glass defeats LiDAR).
  6. `layout.py` — placement `arkit` (raw) or `arkit_yaw` (default; rooms rotated about their
     centroid to the common Manhattan yaw); per-room `placement_sigma_m` = 0.033.
  7. `build.py` — assembles `Plan`: wall/floor/ceiling `Surface`s, `Room`s, `Opening`s,
     `Adjacency` from doorways, footprint, **`quality_flags`**; intervals UNCALIBRATED
     (`Interval.calibrated=False`) from a provisional budget: face position σ 7 mm, scale σ 0.4 %,
     level σ 5 mm, jamb σ 7 mm; method tag `lidar_v1_uncal`.
  8. `render.py` — dimensioned PNG (cm ± half-interval), room area/ceiling labels, doorways.
- **Output on supplied data (MEASURED):** `with_ceiling` → 5 rooms of 8 segmented regions (regions
  2 and 6 fail polygon, region 5 is outside), 2 doorways (one jamb-measured 0.75 m vs grid 0.79 m),
  r4 ceiling `not_observed`; 5 quality flags. `single_room` → 0 rooms + `upper_walls_not_observed`.
  `floor_only` → 1 merged region, ceiling not observed. Runtime ~36 s on `with_ceiling` (M4).
- **Reliable:** loader conventions; segmentation when upper walls were seen; wall positions in
  simple rectangular rooms (0–10 mm half-to-half, e18); ceilings from horizontal normals; quality
  flags. **Provisional:** topology (wall existence) in small/complex rooms; doorway widths (1 of 2
  measured); placement (yaw snap evidence is mixed); all intervals.
- **Known limitations:** needs every wall's top edge in view; small rooms (≈ 1–2 m²) and rooms
  with wardrobes/corridor merges are unstable; no windows; single dominant ceiling (no bulkhead
  handling); merged NW room + corridor (r1, 32 m²) because no lintel separates them; no damage.
- **Tests:** `tests/test_stray.py` (4: depth units, OpenCV T_wc via reprojection < 15 mm, world +y
  up, video row offset), `tests/test_lidar_walls.py` (2 synthetic: rectangle to 2 mm; an
  outward-facing full-height clutter face 5 cm from a wall does not move it).
- **v0 baseline (FROZEN):** `src/roomscan/lidar_pipeline.py` (+ `geometry.py` helpers). Kept
  byte-identical so a fix-loop "before" run stays regenerable. Lint shows 2 unused imports there —
  leave them.

### Photo — EXPERIMENT ONLY (no production code)
- Implemented as experiments only: per-room photo simulation from walkthrough keyframes
  (`e12_photo_scale/photo_sets.py`: |pitch| < 25°, median depth > 1.6 m, sharp, slow; grouped by
  e10 room labels); MoGe-2 per photo; MapAnything multi-view (isolated venv); door / camera-height
  cues; two-family ensemble analysis; stitching probe.
- **Current best scale strategy (PROVISIONAL, not implemented):** ensemble √(MoGe-2 × MapAnything
  raw × c), c ≈ 1.45 learned leave-one-room-out; fixed ≈ ±14–15 % interval (e21).
- **Stitching:** no working method. Doorway photos failed (e17, 0/36).
- **Output status:** no `Plan` is produced for photos; the schema can represent it (rooms with
  `shared:scale` error terms, wide intervals, `inferred`/`not_observed` statuses).
- **Known failures:** small rooms and a mirror-dominated bathroom (e16 room 6/7); see §4.

### Video — EXPERIMENT ONLY (no production code)
- Experiments: COLMAP-SIFT (e08), COLMAP + ALIKED/LightGlue, MoGe-depth RGB-D odometry
  (Open3D, PnP, PnP + pose graph), MapAnything windows (e20); IMU scale (e11, diagnostic only).
- **No tracker gives a continuous *and* accurate walkthrough** on the supplied clips (e20).
- **Scale approach:** none in production. Candidates measured: MoGe depth over many frames inside a
  tracked piece (±2–4 % per piece, e08); IMU (±1–2 % on dense ARKit poses, e11) — IMU excluded
  from the official video contract by user decision.
- **Output status:** no `Plan`. **Known failures:** fragmentation at fast close turns; drift when
  bridging them (§4, e20).

### Shared representation — IMPLEMENTED (`src/roomscan/model.py`, schema v0.1)
- Frame: property frame metres, z up; plan coords (x, y). ARKit→property via
  `ARKIT_TO_PROPERTY` in `geometry.py` (x, −z, y).
- `Interval{lo, hi, level=0.9, calibrated=False, calibration_ref}` — a calibrated interval must
  name its calibration run. `Measurement{value, unit, interval, status ∈ measured|inferred|
  not_observed, method, error_budget}` — an observed measurement **must** have a value and an
  interval containing it; `not_observed` carries value=None. `error_budget` = 1-σ terms by source;
  keys prefixed `shared:` are fully correlated across a room (photo-tier scale).
- `Surface{id, kind ∈ floor|ceiling|wall, room_id, plane, polygon, area, length?, height?,
  shared_with_room_id?}`; `Opening{id, kind ∈ door|window|doorway|unknown, wall_id, room_ids,
  width (clear jamb-to-jamb), height?, sill_height?, offset_along_wall?, detection_confidence}`;
  `Room{id, label, polygon, floor_area, perimeter, ceiling_height, wall_ids, opening_ids,
  placement_sigma_m}`; `Adjacency{room_a, room_b, via_opening_id, shared_wall_ids}`;
  `DamageRegion{surface_id, damage_class, region_uv, area, confidence}`;
  `ConcealedDamageFlag{rule_id, surface_ids, reason}`; `ScopeItem{surface_id, action, quantity}`;
  `Plan{schema_version, capture{tier,device,app,app_version,inputs}, rooms, surfaces, openings,
  adjacency, footprint_area, damage, concealed_damage_flags, scope, drift_correction,
  quality_flags}`. A model validator rejects dangling references.
- Regenerate the schema after editing `model.py`:
  `uv run python -c "import json; from roomscan.model import Plan; open('schema/plan.schema.json','w').write(json.dumps(Plan.model_json_schema(), indent=1))"`

### Evaluation — IMPLEMENTED (`src/roomscan/bench/`)
- **CLI:** `uv run roomscan-bench benchmark/<site> --run runs/<run_id>` expects
  `runs/<run_id>/<capture_id>/plan.json`; writes `runs/<run_id>/eval/<site>.{json,md}`. Running the
  pipelines over the captures is NOT wired in yet (`--execute` not implemented).
- **Site format:** `benchmark/<site>/site.yaml` (schema `bench/gt.py`): rooms with walls in cyclic
  order (clockwise from above, starting at the entry-opening wall; raw repeated readings, median
  used), ceiling readings, openings (`connects:` gives topology), damage, captures (tier, path,
  device, app, `repeat_group`, `room_map`), incumbent numbers. `gt_kind ∈ laser|tape|pseudo_lidar`.
- **Metrics (implemented):** room matching (labels / map / Hungarian on perimeter), wall matching
  by cyclic shift + reflection (a different wall count is reported as `topology_mismatch` and all
  walls unmatched = failures), wall abs/rel error, ceiling error, floor area, opening hit rate with
  misses + phantoms, adjacency correctness, room overlap area, footprint error, repeatability pairs
  per `repeat_group` (max(1 cm, 0.5 %) and ceiling ≤ 1 cm), interval coverage, head-to-head
  beat-or-tie, and `calibration_residuals` rows tagged with `gt_kind`.
- **Pseudo-GT representation:** `gt_kind: pseudo_lidar` → `pseudo_gt: true` in JSON and a
  PSEUDO-GT banner in the markdown; residual rows carry `gt_kind` so they cannot be used to
  calibrate. `benchmark/pseudo_gt_stray_apartment/` was generated by
  `experiments/e15_harness_on_samples/make_pseudo_site.py` from the **v0** pipeline (visit0 plan as
  reference, visit1 scored) — it exercises the harness, it is not evidence of accuracy.
- **True-GT evaluation (future):** fill `benchmark/<site>/site.yaml` from laser readings
  (`benchmark/README.md` conventions), drop raw captures under `captures/`, run each tier's CLI into
  `runs/<id>/<capture_id>/plan.json`, run `roomscan-bench`. Calibration then fits interval widths
  from `calibration_residuals` (only `laser`/`tape` rows).

---

## 4. EXPERIMENT HISTORY

Each row: purpose · data · approach · result · conclusion · reuse? / do-not-repeat?
Full detail: `ENGINEERING_LOG.md` (dated entries) and experiment READMEs.

| ID | Purpose / approach | Key result | Conclusion | Reuse / do not repeat |
|---|---|---|---|---|
| e00 | Survey format (all 3 captures) | depth uint16 mm 256×192, no zeros; conf {0,1,2}; RGB 1920×1440 HEVC VFR ~46 fps, phone held portrait; per-frame fx varies ~2 %; `camera_matrix.csv` = last frame only | Stray format facts | Done; facts in `stray.py` docstring |
| e01 | Pose convention: depth reprojection between frame pairs | OpenCV-axes camera-to-world: 6–8 mm median residual; ARKit-axes reading 200–430 mm; frame offset 0 best | **Stray poses are T_wc with OpenCV camera axes, world +y up** (matches Stray source `q_WA*q_AC`) | Final. Do not re-derive; regression test exists |
| e02 | Fuse depth to voxel clouds + top-down/height views | `single_room` is partial (living+bath); other two are whole-apartment multi-room | Data understanding; PLYs feed later experiments | Re-run `fuse.py` to regenerate ignored PLYs |
| e03 | Drift: revisit residual vs time gap; cross-capture ICP | short-baseline noise 6–12 mm; with_ceiling revisits 60–120 s: median 28 mm, p90 47 mm; cross-capture residual spatially structured (east rooms 40–60 mm) | Drift is real and structured, not registration noise | Metrics reusable (GT-free) |
| e04 | Floor/ceiling height maps | ceilings ≈ 2.35 / 2.95 / 3.07 m per room; one floor region −25 mm (wet room); floor_only "ceiling" 2.09 m = furniture tops | Per-room ceilings mandatory; must detect "not observed" | Done |
| e05 | RGB↔depth frame alignment | video lacks first frame: **decoded video frame j ↔ odometry/depth row j+1** (PTS gap pattern 1713/1713; edge correlation peak) | Encoded as `VIDEO_ROW_OFFSET=1` | Final; regression test exists |
| e06 | MoGe-2 single-image metric scale vs LiDAR (75 frames) | FOV unknown: 9.8 % median, p90 23.6 %; FOV given: 6.4 % / 21.2 %, bias +4.5 % (LiDAR/pred); shape AbsRel 1.3 % | Shape is good, scale is the problem | Done |
| e07 | Keyframe cache (every 5th row, 960×720, metadata) | `cache/frames/` | Shared input for photo/video experiments | Re-run to regenerate |
| e08 | Video: COLMAP (pycolmap 4.2.1, SIFT) on RGB only | 7–14 pieces; longest continuous ≤ 13.5 s; largest with_ceiling model ATE 1.47 m; global mapper positions wrong; MoGe scale per piece ±2–4 %; **pseudo-GT self-inconsistency ~3 %** | COLMAP fragments on this footage | Do not re-run plain SIFT COLMAP hoping for continuity |
| e09 | Drift correction: submap pose graph v1/v2 (Open3D) | v1: cross-capture median better 21→16 mm but revisit p90 46→72 mm, floor_only worse; v2 (strict loops, yaw-only): one loop moved poses 44 cm, worse | **Generic pose graph on top of ARKit: negative**. Per-room rigid ICP brings 7/8 rooms to 11–15 mm → drift ≈ rigid per room | Do not retry generic pose graphs without new information |
| e10 | Room segmentation | v1 (height bands) failed (low corridor ceiling, furniture); **v2: vertical-normal barrier above 1.9 m (lintels close doors) + observed-floor domain → 8 regions on with_ceiling**; 0 rooms on captures without upper walls | Segmentation needs upper-wall observations | v2 is in `geometry.segment_rooms`; labels regenerable |
| e11 | Video scale from IMU (ARKit trajectory with scale/rotation discarded) | without lever arm: −4…−44 % biased; **with camera–IMU lever arm (≈ 8 cm): ±2 % at 0.4–0.6 s filter**; 3 fps keyframes: −2.5…+3.6 %; on real COLMAP pieces (6–13 s): −26…+19 % | IMU works on long dense trajectories only; IMU excluded from official video contract | Diagnostic only (Sensor Logger). Do not put in production video path |
| e12 | Photo scale cues (MoGe-2, 359 photo-like frames, 6 rooms) | per-room bias −0.1…−15.7 %; K photos don't help (|err| ~6 % at K=1…8, ~62 % within ±8 %); camera-height cue: 21 % height error → **refuted**; door cue: zero-shot Grounding DINO boxes mostly not doors → **fails**; oracle doors ~5 %; DAv2-metric-indoor +44 % bias and per-room bias corr 0.75 with MoGe | More photos, camera height, zero-shot doors, second similar model: all fail to remove bias | Do not repeat these cues as implemented |
| e13 | MapAnything (Apache ckpt) room 1 only | raw −28…−35 %; focal predicted ~35 % short; focal-corrected 1.5–5.7 % (room 1) | Looked promising — **superseded by e16** | Do not cite e13 alone |
| e14 | LiDAR v0 pipeline + visit-to-visit proxy | 7 rooms; 50 % of dims within max(1 cm, 0.5 %); topology varies between visits | Baseline for e18 and the fix loop | v0 frozen |
| e15 | Harness on supplied data (pseudo-GT site from v0 visit0) | visit1: 4/7 rooms topology mismatch; matched walls 13 mm median; v0 intervals cover 32–77 % vs 90 % nominal | Harness works; v0 intervals ~3× overconfident | Reusable demo of the harness |
| e16 | Does MapAnything + true intrinsics generalise? (300 runs, 6 rooms, K 2–8) | focal-corrected: rooms 1/3/4/6 1–7 %, room 8 10–12 %, **room 7 (mirror bathroom) +31…+44 %**; f_pred/f_true = 0.654 ± 0.037 regardless of true FOV (crop probe 48.5/37.3/25.4° → ~68–70°); held-out constant beats EXIF correction | **MapAnything assumes ~70° FOV + a size prior; "focal correction" ≈ ×1.5 constant; does not generalise** | Do not use MapAnything EXIF-corrected depth as a metric source on its own |
| e17 | Photo stitching via doorway photos (MapAnything joint runs + SIFT) | **0/36** successful links (< 0.5 m, < 15°); even all-camera diagnostic fit leaves B 0.4–1.2 m off; SIFT link above control floor for 1/4 pairs | Single doorway photos insufficient | Next hypothesis: overlapping doorway chain (untested) |
| e18 | LiDAR repeatability variants V0–V6 (chunk halves, fixed topology) | wall Δ median/p90 and % within gate: V0 10.2/36.7 mm 57 %; V1 (aggregate, unoriented) 11.2/49.8 44 %; V2 (oriented normals) 11.0/27.6 65 %; V3 regularise 62 %; V4 outermost peak 19.5/71.0 31 %; V5 close-range only 7.5/25.7 61 % but topology fails; **V6 9.8/33.5 73 %**; ceilings 75 % ≤ 1 cm (V2+). Visit-split halves: 28/28 failed (no upper walls per half) | Oriented normals + close-range positions help; aggregation without orientation hurts; regularisation/outermost-peak hurt; topology is coverage-hungry | V6 is production. Do not repeat V1/V3/V4 |
| e19 | Drift ablation: raw ARKit vs plane-anchored stitch (visit-split placements, shared shapes) | corners median/p90/max: raw 54.8/96.8/115.8 mm; yaw snap 66.0/84.4/93.8; doorway constraints measurable 0/4; shared-wall stitch (common thickness) 69.2/104.6/120.0 and moved a room ~19 cm; per-wall thickness 90.5/…; shared-wall gap spread is circular (dropped) | **Plane-anchored translation stitch: negative**; yaw snap inconclusive | Do not use common-wall-thickness stitching |
| e20 | Video continuity (with_ceiling 215 s) | SIFT 14 pieces, 20 s-window ATE 4.4 cm; LightGlue 1 piece/65 % but ATE 2.97 m; depth odometry 1 piece/100 % but ATE 2.6–3.0 m, end drift 9–15 m, long-range err 44–59 %; PnP cut at failed links: longest 62 s, long-range 7.2 %; MapAnything windows (single_room) long-range 41 %; **1.8 % of links fail, in ≤ 1.2 s bursts, at fast turns (53 vs 25 °/s) close to surfaces (0.86 vs 1.59 m)** | No continuous + accurate tracker; failures are capture-behaviour-driven | Do not re-run MapAnything windows on with_ceiling (aborted on memory, 1/41 windows) |
| e21 | Photo two-family ensemble + disagreement intervals (e16 runs, LORO) | MoGe 7.5/19.1 %, 54 % within ±8 %, 7 catastrophic; MA×c 5.2/23.7 %, 27 catastrophic; **ensemble 5.3/14.0 %, 64 %, 0 catastrophic**; fails rooms 6 and 7; fixed ±14 % → 85 % coverage; disagreement-adaptive intervals no better (82 %) | Best photo strategy found; ±8 % at risk | Reuse as the photo-scale baseline |

---

## 5. CURRENT MEASURED RESULTS

| Quantity | Value | Label | Source |
|---|---|---|---|
| Stray pose convention | T_wc, OpenCV camera axes, world +y up, metres | FACT | e01, `stray.py`, Stray source |
| Video ↔ depth frame mapping | decoded video j ↔ row j+1 | FACT | e05 |
| Short-baseline LiDAR depth consistency | 6–12 mm median | MEASURED | e01, e03 |
| Revisit disagreement after 60–120 s (with_ceiling) | 28 mm median / 47 mm p90 | MEASURED | e03 |
| Per-room rigid alignment between captures | 11–15 mm median in 7/8 rooms | MEASURED | e09 per_room_rigid |
| LiDAR vs ARKit-trajectory scale self-consistency | ~3 % (ratio 1.009–1.04) | MEASURED | e08 |
| LiDAR wall-position repeatability (V6, half data, given topology) | 9.8 mm median, 33.5 mm p90, 73 % within max(1 cm, 0.5 %) | MEASURED (GT-free) | e18 |
| Same, rectangular rooms | 0–10 mm per wall | MEASURED (GT-free) | e18 |
| Ceiling repeatability (V2/V6) | 75 % of rooms ≤ 1 cm half-to-half | MEASURED (GT-free) | e18 |
| LiDAR absolute accuracy (walls, ceilings, openings) | unknown | BLOCKED (laser GT) | — |
| v0 interval coverage vs pseudo-GT | 32–77 % (nominal 90 %) | PSEUDO-GT | e15 |
| Room placement disagreement, raw ARKit (corners) | 54.8 / 96.8 / 115.8 mm (median/p90/max) | MEASURED (GT-free) | e19 |
| Photo scale, MoGe-2 (FOV given), per run | 7.5 % median, 19.1 % p90 | PSEUDO-GT | e21 |
| Photo scale, ensemble MoGe × MA·c | 5.3 % median, 14.0 % p90, 64 % within ±8 %, 0 > 25 % | PSEUDO-GT (LORO) | e21 |
| Photo ensemble fixed ±14 % interval coverage | 85 % | PSEUDO-GT (LORO) | e21 |
| MapAnything predicted focal ratio | 0.654 ± 0.037 (FOV ~70° vs 48.5°) | MEASURED | e16 |
| Photo stitching via doorway photos | 0/36 | PSEUDO-GT eval | e17 |
| Video: best accurate local tracking | 4.4 cm ATE over 20 s windows, but 14 pieces | PSEUDO-GT | e08/e20 |
| Video: continuous trajectory accuracy | ≥ 2.6 m ATE, 44–59 % long-range error | PSEUDO-GT | e20 |
| Video tracking failure rate | 1.8 % of links, bursts ≤ 1.2 s | MEASURED | e20 blind.json |
| IMU scale on ARKit trajectories (lever arm) | ±2 % | PSEUDO-GT | e11 |
| Photo ±8 % achievable without a reference object | only in "typical" rooms | HYPOTHESIS | e21 |
| A protocol-following video tracks continuously | — | HYPOTHESIS / BLOCKED | e20 |

---

## 6. CURRENT DECISIONS

| Decision | Status | Evidence | Rejected alternatives | Would change if |
|---|---|---|---|---|
| Python ≥ 3.12 (env 3.13.14) + uv; `ml` group default; MoGe pinned to commit `b942f00bdc2a` | FINAL | MoGe `main` is MoGe-3 with CUDA-only deps | pip/conda | — |
| Stray loader conventions (T_wc OpenCV, +y up, mm depth, row j+1 video) | FINAL | e01, e05, tests | ARKit-axes reading | Stray v1.4 export differs (check first real capture) |
| Common `Plan` representation; every measurement has interval + status; `quality_flags` | FINAL (additive changes only) | used by LiDAR v1 + evaluator | per-tier schemas | a tier output cannot be represented |
| Pseudo-GT only for consistency; never presented as benchmark; residual rows tagged | FINAL | ~3 % self-inconsistency | treating LiDAR as GT | laser GT arrives (then calibrate) |
| Capture Route 2 (stock apps) leaning: Stray (LiDAR), stock Camera (photo/video), magicplan (incumbent) | PROVISIONAL | no iPhone/dev account; Stray format verified; Stray refuses non-LiDAR phones | Route 1 own app ($99, TestFlight, Swift) | Route 2 cannot meet a gate that ARKit/RoomPlan would |
| Photo reference object NOT mandatory | PROVISIONAL (user) | user instruction | mandatory A4/card | photo ±8 % proven unreachable on real benchmark |
| Official video = plain handheld clip; IMU/Sensor Logger diagnostic only | PROVISIONAL (user) | user instruction; e11 shows IMU would help | video+IMU production path | user revisits interpretation |
| LiDAR geometry = e18 V6 (oriented normals, per-room ICP keeping unaligned visits, close-range wall positions, horizontal-normal levels) | PROVISIONAL | e18 | V1/V3/V4/V5 | laser GT shows bias/other winner |
| LiDAR drift = room-local measurement + `arkit_yaw` placement + per-room placement σ; `arkit` kept for the on/off ablation | PROVISIONAL | e09, e19 | generic pose graph, plane-anchored translation stitch | two real captures per protocol show stitch gains |
| Photo scale = MoGe-2 × MapAnything·c ensemble, fixed wide interval | PROVISIONAL (not implemented) | e16, e21 | MoGe alone, MA EXIF-corrected, camera height, zero-shot door cue, DAv2 | real photos behave differently; a new cue appears |
| Photo stitching = unsolved; next test = overlapping doorway chain | HYPOTHESIS | e17 | single doorway photo | dev capture result |
| Video = accurate local pieces + room measurement within pieces + slow-turn protocol | PROVISIONAL (not implemented) | e20 | depth odometry, LightGlue COLMAP, MapAnything windows | protocol-following clip tracks continuously |
| GPU: one model process at a time; bf16 MapAnything; release MPS cache; pause on free pages < 50k or pageouts > 5000/30 s (not on swap level) | FINAL (operational) | e13/e16/e20 swap thrash, agent stall | parallel GPU jobs; swap-threshold guard | more RAM |
| v0 pipeline frozen | FINAL | fix-loop before/after must be regenerable | editing v0 | — |

---

## 7. CURRENT ASSESSMENT STATUS (reconciled with the repository)

`docs/compliance_matrix.md` is mostly accurate; this table supersedes it where they differ
(differences marked ★). Statuses: PASS / PARTIAL / EXPLORING / BLOCKED / NOT STARTED.

| Requirement | Status | Evidence path |
|---|---|---|
| LiDAR tier runs on a Stray export | PARTIAL | `src/roomscan/lidar/`, `runs/v1/` (5/8 rooms; needs upper-wall coverage) |
| Photo tier | EXPLORING (no production path) | `experiments/e12…e17, e21` |
| Video tier | EXPLORING (no production path) | `experiments/e08, e11, e20` |
| Common output contract / schema | PARTIAL | `src/roomscan/model.py`, `schema/plan.schema.json` (LiDAR emits; photo/video don't) |
| One command per capture ★ | PARTIAL (LiDAR only) | `roomscan-lidar` |
| Rendered plan | PARTIAL (LiDAR) | `src/roomscan/lidar/render.py`, `runs/v1/*/plan.png` |
| Stitched multi-room plan, adjacency ★ | PARTIAL (LiDAR: placed rooms + doorway adjacency, incomplete) | `lidar/layout.py`, `lidar/build.py` |
| Openings / G1 ★ | EXPLORING (doorways only, 1 jamb-measured; no windows; no GT) | `lidar/openings.py` |
| Ceiling height / G2 | EXPLORING (75 % ≤ 1 cm GT-free; bias unknown) | `lidar/levels.py`, e18 |
| Repeatability / G3 | EXPLORING, at risk (73 % positions; topology unstable) | e18 |
| Drift ablation / G4 | PARTIAL (method stated + GT-free ablation figure; translation stitch negative) | e09, e19 (`out/footprint_v2_*.png`) |
| Photo stitch / G5 | EXPLORING, at risk (0/36) | e17 |
| Photo walls / G6 | EXPLORING, at risk (64 % within ±8 %, pseudo-GT) | e21 |
| Video walls / G7 | EXPLORING, at risk | e20 |
| Calibration / G8 / C6 | NOT STARTED (machinery ready, intervals uncalibrated) | `bench/evaluate.py` `calibration_residuals` |
| Damage / concealed flags / scope (O3–O5) | NOT STARTED (schema ready) | `model.py` |
| Benchmark set (B1–B5) | BLOCKED (physical) | `benchmark/README.md`, `docs/physical_benchmark_procedure.md` ★ GT lives in `site.yaml` + `measurements/`, not `benchmark/ground_truth/` |
| Benchmark evaluator | PASS (machinery; tested) | `src/roomscan/bench/`, `tests/test_bench.py` |
| Head-to-head (H1) | BLOCKED (physical; magicplan free export unverified) | `bench/gt.py` `Incumbent` |
| Fix loop (F1/F2) | NOT STARTED (needs a real failing gate) | v0 frozen as "before" candidate |
| Capture route (C1) | EXPLORING (Route 2 provisional; final page not written) | `docs/capture_protocol_hypotheses.md` |
| Device matrix (C2) | NOT STARTED | — (`docs/device_matrix.md` does not exist) |
| Clean-machine README < 15 min (D1) | NOT STARTED (README exists; never timed; isolated venvs not scripted) | `README.md` |
| Reproduction bundle (D2) | PARTIAL (evaluator; no pipeline-execution step; weights not fetched by script) | `bench/` |
| Benchmark report (D3), technical report ≤ 6 pages (D4) | NOT STARTED | — |
| Raw benchmark data (D5) | BLOCKED (physical) | — |
| Offline / weights fetched by script (K1) | NOT STARTED (HF `from_pretrained` on first use) | — |
| Mirrors/glass/wet/low light (K2) | EXPLORING (mirror bathroom breaks photo scale; glass in samples) | e16, e00 |
| Walk-in readiness (W1) | NOT STARTED (LiDAR only, protocol not final) | — |
| Process evidence (P1) | PASS so far (17 incremental commits) | `git log` |

---

## 8. CURRENT FILES THAT MUST NOT BE LOST

| Path | Why |
|---|---|
| `src/roomscan/stray.py` | verified Stray conventions (pose axes, video offset) — hard-won facts |
| `src/roomscan/lidar/` (all modules) | LiDAR v1 production pipeline |
| `src/roomscan/lidar_pipeline.py`, `src/roomscan/geometry.py` | frozen v0 baseline (fix-loop "before") + shared segmentation |
| `src/roomscan/model.py`, `schema/plan.schema.json` | output contract |
| `src/roomscan/bench/` | evaluator, gates, residual export |
| `tests/test_stray.py`, `tests/test_bench.py`, `tests/test_lidar_walls.py` | regression for conventions, gates, wall extraction |
| `ENGINEERING_LOG.md` | reasoning and every experiment's evidence |
| `docs/capture_protocol_hypotheses.md`, `docs/physical_benchmark_procedure.md`, `docs/compliance_matrix.md` | protocol evidence, capture plan, requirement tracking |
| `benchmark/README.md`, `benchmark/_template/site.yaml` | GT format the physical session must follow |
| `experiments/e18_lidar_repeat/out/*.json`, `README.md` | LiDAR variant evidence |
| `experiments/e19_drift_stitch/out/ablation_v2_*.json`, `footprint_v2_*.png` | drift ablation (gate G4 evidence) |
| `experiments/e16_photo_scale_generalization/out/runs.jsonl`, `summary.json`, `per_room_table.md` | 300 MapAnything/MoGe runs; e21 reads runs.jsonl |
| `experiments/e21_photo_ensemble/out/summary.json` | photo strategy evidence |
| `experiments/e17_photo_stitch/out/*` | stitching negative result |
| `experiments/e20_video_tracking/out/*_eval.json`, `table.md`, `*_blind.json` | video tracking evidence and failure anatomy |
| `experiments/e11_imu_scale/out/*.json` | IMU lever-arm result |
| `experiments/e12_photo_scale/out/*.json`, `experiments/e08_video_sfm/out/*` | scale-cue and COLMAP evidence |
| `cache/e13/hf/hub/blobs` (ignored, 4.6 GB) | MapAnything weights (≈ 18 min download) |
| `cache/e13/.venv`, `cache/e13/map-anything` (ignored) | MapAnything env + source at `3d10cf7` |
| `cache/e20/.venv` (ignored) | LightGlue/kornia env — **no setup recipe in repo** |
| `cache/e20/moge/single_scan_with_ceiling/` (ignored) | MoGe depth for all 818 keyframes (~25 min GPU) |
| `cache/e16/` (ignored) | cached MapAnything per-run predictions |
| `pyproject.toml`, `uv.lock`, `.gitignore` | environment + what stays out of git |
| `single_*.zip` (ignored, repo root) | the only raw data |

---

## 9. ENVIRONMENT / REPRODUCTION

- **Machine:** Apple M4, 10 cores, 16 GB RAM, macOS 26.7.1, no CUDA; MPS available. uv 0.12.23.
- **Project env:** `.venv` (Python 3.13.14). numpy 2.5.3, scipy 1.18.1, open3d 0.20.0, opencv 5.0.0
  (both `opencv-python` and `-headless` installed — `ml` group pulls the non-headless one; works),
  torch 2.14.1 (MPS), transformers 5.18.0, pydantic 2.13.5, shapely 2.1.2, moge 2.0.0 (git
  `b942f00bdc2a`), pycolmap 4.2.1.
- **Models:** MoGe-2 `Ruicheng/moge-2-vitl-normal`; Grounding DINO `IDEA-Research/grounding-dino-tiny`;
  Depth Anything V2 `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`; MapAnything
  `facebook/map-anything-apache` (repo v1.1.4 commit `3d10cf7`, HF sha `00f9c24`, Apache-2.0);
  ALIKED + LightGlue (cvg) in `cache/e20/.venv`.
- **Isolated envs:** `cache/e13/.venv` (Python 3.12.13, torch 2.14.1, mapanything 1.1.4,
  opencv-headless 4.10.0.84 — mapanything pins it, conflicting with the project's 5.x; recipe in
  `experiments/e13_multiview_photo/README.md`, requires `cache/e13/map-anything` source checkout).
  `cache/e20/.venv` (Python 3.12.13, torch 2.14.1, lightglue 0.0, kornia 0.8.3; recipe not
  recorded).
- **Known conflicts:** pycolmap and torch cannot be imported in the same process (duplicate
  OpenMP runtime → `OMP: Error #15`); e08/e20 run MoGe in a separate process. cv2 and PyAV both
  bundle libavdevice (objc duplicate-class warnings; harmless). MoGe prints an MPS autocast
  warning (harmless).
- **Memory:** MapAnything fp32 peak ≈ 7.7 GB, bf16 ≈ 4.2–4.6 GB, load spike ≈ 6.4 GB; MoGe ViT-L ≈
  1–2 GB; the user's browser often holds several GB. Run one GPU model at a time. Pause on free
  pages < 50 000 or pageouts > 5000 per 30 s — **not** on swap level (swap stays high after spikes).

**Commands (verified at `1b1e53e`):**
```
uv sync                                                    # project env incl. ml group
uv run pytest -q                                           # 13 passed
uv run roomscan-lidar single_scan_with_ceiling runs/v1/single_scan_with_ceiling_lidar
uv run python -m roomscan.lidar_pipeline single_scan_with_ceiling runs/v0/with_ceiling_lidar   # v0 baseline
uv run roomscan-bench benchmark/pseudo_gt_stray_apartment --run runs/pseudo_v0                  # PSEUDO-GT demo
```
**Regenerating ignored intermediates (order matters on a fresh clone):**
```
# raw data: each zip extracts to a hash-named folder; the code expects these repo-root names:
#   single_room.zip -> c00a170fe1/ -> single_room/
#   single_scan_floor_only.zip -> 1a8384c3f6/ -> single_scan_floor_only/
#   single_scan_with_ceiling.zip -> c7d28f72c6/ -> single_scan_with_ceiling/
uv run python experiments/e02_fusion/fuse.py               # *_vox2cm.ply (e03, e04, e10, e17)
uv run python experiments/e07_frame_cache/extract.py       # cache/frames/
uv run python experiments/e10_room_seg/segment_v2.py       # *_v2b_h*_labels.npy + *_v2_grid.json
uv run python experiments/e12_photo_scale/photo_sets.py    # out/photo_sets.json
uv run python experiments/e12_photo_scale/run_moge.py      # cache/e12 (≈ 10 min MPS)
```
**Key experiment commands:**
```
uv run python experiments/e18_lidar_repeat/run_fixed_topology.py single_scan_with_ceiling V0,V1,V2,V3 2.0
uv run python experiments/e18_lidar_repeat/run_fixed_topology.py single_scan_with_ceiling V6 2.0
uv run python experiments/e19_drift_stitch/stitch_v2.py single_scan_with_ceiling V2
uv run python experiments/e21_photo_ensemble/analyze.py    # CPU only, reads e16 runs.jsonl
uv run python experiments/e11_imu_scale/lever_arm.py
uv run python experiments/e15_harness_on_samples/make_pseudo_site.py
# MapAnything (isolated): see experiments/e16_photo_scale_generalization/README.md (cache/e13/.venv/bin/python)
# Video: see experiments/e20_video_tracking/README.md and each script's docstring
```

---

## 10. GIT STATE

- Branch `master` (main branch name in the repo metadata is `main`, but all work is on `master`).
- HEAD `1b1e53e` "Phase 3 log, protocol hypotheses, compliance and physical procedure updates";
  working tree clean before this handover.
- History (17 commits): loader/conventions → exploration e00–e06 → log + matrix → ml group →
  schema + evaluator → LiDAR v0 + harness → frame cache/drift/segmentation → photo/video scale →
  phase-2 log → schema extensions → e18 → LiDAR v1 → e19 → e16/e17/e21 → e20 → phase-3 log.
- Intentionally untracked/ignored: raw captures and zips, PDF, `cache/`, `runs/`, `*.ply`,
  `*.npy`, `*.npz`, `.venv`.
- Incomplete work: none in progress; no background processes. Phase 3 is closed.
- Commit rule (user's global instructions): **Claude does not run git write commands** unless the
  user explicitly asks in that session; otherwise print the commands for the user.

---

## 11. OPEN QUESTIONS / BLOCKERS

### Can be answered locally
| Question | Why it matters | Evidence so far | Next action |
|---|---|---|---|
| Can photo and video tiers emit a `Plan` through the shared back end? | Every tier must produce the contract; evaluator can then score them | photo/video are experiments only | build photo-tier v0 (per-room MoGe×MA ensemble → room polygon → Plan with `shared:scale`) and score it on the pseudo-GT site |
| Why do regions 2 and 6 fail polygon extraction? | Small rooms (WC/lobby) are common; missing rooms fail adjacency and footprint | e18: fail even with all data; few wall lines | debug with `experiments/e14_lidar_v0/debug_rooms.py`-style plots on V6 |
| Window detection (glass) for openings | G1 counts misses; LiDAR cannot see glass | none | RGB detector (window frames) projected onto wall planes; needs design |
| Can a video piece measure a room? | Video may only need per-room accuracy, not a continuous walk | SIFT pieces: 4 cm window ATE | room measurement within the largest piece covering each room, scaled by MoGe; score against v1 LiDAR plan |
| Footprint ablation as an actual stitched-plan figure for G4 | the gate asks for the footprint on/off | e19 figure uses visit halves | render `--placement arkit` vs `arkit_yaw` plans side by side |
| Pipeline execution in the evaluator (`--execute`) and weight-fetch script | D1/D2/K1 | not built | add after photo/video CLIs exist |

### Requires physical device
| Question | Why | Evidence | Next action |
|---|---|---|---|
| Does wall topology stabilise when every wall's top edge is seen? | LiDAR repeatability gate | e18 topology failures with partial coverage | dev capture (see §12) |
| Does a slow-turning, ≥ 1 m-from-wall video track continuously? | video tier viability | e20 failure anatomy | dev capture |
| Does an overlapping doorway chain let photo folders stitch? | G5 | e17 negative for single photos | dev capture |
| Absolute LiDAR accuracy, bias vs repeatability, opening widths, calibration | all gates, fix loop | none | laser benchmark session |
| Stray v1.4 output parity; real photo EXIF/HEIC; Sensor Logger format; magicplan free export | loaders, head-to-head | research only | dev capture / pre-check |

### Requires external decision (user)
| Question | Why | Evidence | Recommendation |
|---|---|---|---|
| Is an optional scale cue acceptable in the photo protocol (e.g. "include one door in each room's photos", or an optional reference object)? | photo ±8 % at risk | e21: 64 % within ±8 %; rooms 6/7 fail | decide after real photos are measured |
| Repeatability tolerance reading: max(1 cm, 0.5 %) vs stricter | gate pass/fail | evaluator uses the looser reading | keep, state it in the report |
| Route 1 reconsideration | only if Route 2 can't meet a gate | none yet | revisit after the dev capture |

---

## 12. PHYSICAL CAPTURE PLAN

Authoritative plan: `docs/physical_benchmark_procedure.md` (captures grouped by purpose with id
prefixes `dev_`, `bm_`, `rep_`, `inc_`, `walk_`, `fail_`). Summary:
- **Equipment:** one borrowed iPhone Pro with LiDAR (15 Pro or newer preferred; 12–14 Pro fine
  for development — record the model), ±1.5 mm laser meter (e.g. Bosch GLM 40), 5 m tape,
  masking tape, laptop for AirDrop/USB. No Apple Developer account.
- **Apps (free):** Stray Scanner (LiDAR), stock Camera (photo HEIC with EXIF; plain video),
  magicplan Starter (incumbent), Sensor Logger (diagnostic only).
- **Ground truth:** walls corner-to-corner at ~1 m, 3 readings; ceilings ≥ 3 points per room;
  openings clear jamb-to-jamb at mid-height; damage extents; into `site.yaml` per
  `benchmark/README.md`; photograph measurement sheets.
- **Benchmark (`bm_`):** whole property at all three tiers (photo as per-room folders, plus the
  overlapping-doorway-chain variant), damage room. **Repeatability (`rep_`):** whole property +
  damage room again (LiDAR), one room again at photo/video. **Incumbent (`inc_`):** magicplan on 2
  rooms incl. the damage room, keep native export. **Walk-in proxy (`walk_`):** a second person
  follows only the one-page protocol. **Failure mode (`fail_`):** fast LiDAR walk without
  upper-wall sweep; mirror-dominated bathroom photos.
- **Export:** unzipped Stray folders, original HEIC (AirDrop "All Photos Data" on / USB "Keep
  Originals"), original .MOV, Sensor Logger zip, magicplan export, measurement-sheet photos.

**FIRST short development capture (~30 min) is meant to validate, before any laser session:**
1. LiDAR topology stability when every wall's top edge is in view (e18 risk);
2. continuous video tracking with slow turns kept ≥ 1 m from walls (e20 hypothesis);
3. photo stitching with an overlapping doorway chain (e17 hypothesis);
4. real-file formats: Stray v1.4 export, HEIC/EXIF focal, .MOV timing, magicplan free export.

---

## 13. WHAT SHOULD HAPPEN NEXT (prioritised)

1. **Photo-tier v0 production path into the shared Plan.**
   - Why: a whole tier has no output; the walk-in may pick photos; the evaluator can't score it.
   - Prerequisite: none (e16 caches, e21 analysis).
   - Output: `src/roomscan/photo/` + CLI (`roomscan-photo <folders> <out>`): per-room MoGe-2 ×
     MapAnything·c scale, room polygon from per-room geometry, `shared:scale` error terms, fixed
     wide intervals, `quality_flags`; rooms unplaced (stitching unsolved, flagged).
   - Success: schema-valid Plans for simulated room folders; pseudo-GT wall error ≈ e21 numbers.
   - Files: e12 `photo_sets.py`, e16 `e16lib.py`, e21 `analyze.py`.
2. **Video-tier v0: room measurement within tracked pieces.**
   - Why: accurate local pieces exist; a continuous walk may not be needed for per-room dims.
   - Prerequisite: e20 caches (features, matches, MoGe depth).
   - Output: experiment then `src/roomscan/video/`: per room, the best covering piece, MoGe-scaled,
     walls via the LiDAR wall code on MoGe point clouds → Plan.
   - Success/failure: pseudo-GT wall error per room vs v1 LiDAR plan; report pieces/coverage
     honestly. Files: e20 `rgbd_odom.py`, `colmap_lg.py`, `common.py`.
3. **Small-room topology failures (regions 2 and 6) in LiDAR v1.**
   - Why: missing rooms break adjacency, footprint and repeatability.
   - Output: diagnosis + fix measured with `run_fixed_topology.py` and `roomscan-lidar`.
   - Success: rooms 2/6 produce polygons without degrading e18 V6 numbers.
4. **Short development capture (physical; user borrows a device).**
   - Why: decides the three biggest open hypotheses (§12). Prerequisite: device. Output: `dev_`
     captures + verdicts logged. Success: each hypothesis supported or refuted with numbers.
5. **Reproduction hardening: weight-fetch script, isolated-env recipes (e20 missing), `--execute`
   in the evaluator, timed clean-machine README run.**
   - Why: D1/D2/K1 and walk-in cold runs. Output: `scripts/fetch_weights.*`, env recipes,
     README timing. Success: fresh clone → LiDAR plan in < 15 min.
6. **Laser benchmark session → calibration → pick the worst failing gate for the fix loop**
   (v0 is the frozen "before" candidate). Blocked on hardware.
7. Lower priority until geometry is credible: windows (RGB), damage detection + concealed rules +
   scope, technical report, device matrix.

---

## 14. THINGS NOT TO DO

| Don't | Why |
|---|---|
| Treat Stray poses as ARKit camera axes | 200–430 mm reprojection error (e01); loader is correct |
| Pair video frame j with depth row j | off by one (e05) |
| Present any number here as benchmark accuracy | pseudo-GT ~3 % self-inconsistent; no laser data |
| Re-run generic submap pose graphs over ARKit | e09 v1/v2 negative |
| Use plane-anchored translation stitching with a common wall thickness | e19: worse, moved a room ~19 cm |
| Use "shared-wall gap spread" as a drift metric | circular — it is what the stitch optimises |
| Aggregate visits without camera-oriented normals | e18 V1 worse than baseline |
| Drop visits that fail ICP | removes upper-wall coverage; all walls vanish (e18) |
| Outermost-peak wall positions, notch/merge regularisation | e18 V4/V3 negative |
| Expect more photos per room to fix scale | per-room bias dominates at every K (e12/e16) |
| Camera-height or zero-shot Grounding-DINO door-height scale cues | refuted / not deployable (e12) |
| Second monocular model of the same family (DAv2) as a bias cancel | correlated per-room bias r = 0.75, +44 % bias (e12) |
| Use MapAnything EXIF "focal correction" as metric depth | it is a ×1.5 constant; fails in mirror room (e16) |
| Rely on single doorway photos for stitching | 0/36 (e17) |
| Plain SIFT COLMAP or LightGlue COLMAP for a continuous walkthrough | fragments / wrong geometry (e08, e20) |
| Depth-odometry bridging as a "continuous" solution | 9–15 m drift, 44–59 % long-range error (e20) |
| MapAnything windows on long walkthroughs | least accurate; its load starved the 16 GB machine twice (e20) |
| Put IMU/Sensor Logger in the production video path | user decision: video = plain clip |
| Run two GPU models at once, or gate on swap level | swap thrash; agent stalls; swap stays high after spikes |
| Edit `src/roomscan/lidar_pipeline.py` (v0) | fix-loop "before" must remain regenerable |
| Run `uv sync` without the `ml` group configured | it once removed torch mid-run (now `default-groups = ["dev","ml"]`) |
| Delete caches without checking §8 | MapAnything weights/env and MoGe depth cost 20–40 min to rebuild |

---

## 15. HANDOVER QUALITY CHECK

1. What are we building? → §1. 2. What is implemented? → §3 (LiDAR v1, schema, evaluator).
3. How do I run it? → §9. 4. Experiments done? → §4. 5. What did they prove? → §4–5.
6. What failed? → §4, §14. 7. Architecture decisions? → §6. 8. Still uncertain? → §11.
9. Blocked by hardware? → §11, §12. 10. Next work? → §13. 11. Which files matter? → §8.
12. What must not be overwritten? → §8 (and v0 frozen, §14). 13. Pseudo-GT vs benchmark? → header
caveat, §5 labels, §3 Evaluation.

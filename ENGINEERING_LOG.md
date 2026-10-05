# Engineering Log

Reasoning record for the Applied AI Engineer case study: what the assessment requires,
what the data contains, what was tried, what was decided and why. Newest material is
appended within each section; dated entries live in §6.

---

## 1. What the assessment requires (read from `Applied_AI_Case_Study.pdf`)

### 1.1 Product
A pipeline from iPhone sensors to a **stitched, dimensioned whole-property floor plan**
(magicplan / Polycam-like), plus damage regions and scope line items.

Per capture, the "Round 1 contract":
- dimensioned per-room plan: walls, ceiling height, floor area, openings
- stitched multi-room plan with correct adjacency
- per-surface damage regions with class and metric extent
- concealed-damage flags with the rule that fired
- scope line items keyed to surfaces
- a confidence interval on **every** measurement
- one command per capture, JSON to a published schema, rendered plan

### 1.2 Input tiers (all mandatory, same output contract)
| Tier | Input | Device | Accuracy gate |
|---|---|---|---|
| Photo | 2–8 stills/room, one folder per room, no depth/poses | any iPhone 15+ | wall lengths ±8%, footprint ±8%, must stitch multi-room |
| Video | handheld walkthrough clip | any iPhone 15+ | wall lengths ±3% |
| LiDAR | depth + poses + intrinsics | Pro-class | Round-1 gates below |

Intervals must "widen honestly as sensor data thins". **Calibration is scored at every
tier; "confident garbage on thin input caps your total score."** → interval coverage is a
first-class metric, not decoration.

### 1.3 Capture route (choose one)
- Route 1: own iOS app (TestFlight / dev build, installable < 10 min).
- Route 2: named off-the-shelf app + one-page protocol a non-engineer follows *literally*.

### 1.4 Gates (LiDAR tier unless stated)
| Gate | Threshold |
|---|---|
| Opening widths | ≤ 2 cm on ≥ 85 % of openings; missed and phantom openings both count as misses |
| Ceiling height | ≤ 1.5 cm per room; spread across repeat captures ≤ 1 cm; report must say biased vs unrepeatable |
| Repeatability | two same-tier captures agree within 1 cm or 0.5 % per wall |
| Drift accountability | state drift handling + ablation of stitched footprint on/off; "poses as-is" = automatic fail |
| Photo-tier stitch | per-room folders → one stitched plan, correct adjacency, no overlaps, footprint ±8 % with calibrated intervals |

### 1.5 Benchmark set (we build it; composition fixed)
- ≥ 1 multi-room capture: ≥ 3 rooms + a connector
- ≥ 1 furnished room with staged damage, ≥ 2 damage classes
- same rooms at all three tiers (multi-room included; photo tier as per-room folders)
- ≥ 1 room captured twice at the same tier
- laser/tape ground truth on everything; raw data + measurements submitted

### 1.6 Other parts
- **Head-to-head (10 %)**: on 2 benchmark rooms, our LiDAR output vs one consumer app
  (named, versioned, export submitted); beat or tie on ≥ 70 % of shared dimensions.
- **Fix loop (25 %)**: declare worst gate + failing number, root cause + evidence, fix +
  predicted number; ship it; before/after both regenerable; readable diff.
- **Process (5 %)**: commit history must show incremental work.
- **Walk-in test (30 %)**: evaluator captures an unseen space with their iPhone 15+, picks
  the tier on the day, follows our route literally; pipeline runs cold; laser-scored live.

### 1.7 Deliverables
Compliance matrix · capture route + device matrix · README (fresh machine → running in
< 15 min, one command per capture) · reproduction bundle (deterministic cache allowed only
if the live path also runs) · benchmark report (all tiers, repeatability, head-to-head,
timing) · fix-loop bundle · technical report ≤ 6 pages · raw benchmark data.

Constraints: handheld consumer capture only; any pretrained model/dataset/API with
disclosure; **everything runs without calling our infrastructure**; weights fetched by
script; cover mirrors, glass, wet-look surfaces, low light.

### 1.8 Reading the scoring like an evaluator
- 55 % of the score (walk-in 30 % + fix loop 25 %) is about **robustness on unseen data**
  and **diagnose→repair discipline**, not feature breadth.
- The walk-in tier is chosen on the day → all three tiers must run cold, end-to-end, on
  an unseen capture made by a stranger following our protocol. Protocol clarity directly
  drives walk-in accuracy.
- Calibration penalty is asymmetric: an over-confident wrong answer is worse than a wide
  honest interval. Intervals must be derived from measured error, not guessed.
- The fix loop needs a *genuinely failing* gate in our own benchmark plus a fix that
  moves it. Building the benchmark/eval harness early is a prerequisite for 25 % of the
  score, and the harness has to be regenerable.

---

## 2. What the supplied data actually contains

(each item is established by an experiment in §6; outputs under `experiments/*/out/`)

**Format: Stray Scanner export, one device, one evening, one apartment.**

| Item | Finding | Evidence |
|---|---|---|
| Depth | uint16 PNG, **millimetres**, 256×192, no zero pixels (ARKit `sceneDepth`), range 0.24–7.9 m | e00 |
| Confidence | uint8 {0,1,2}; 84–93 % of pixels are level 2; low-confidence pixels are far ones (median 3–4 m) | e00 |
| RGB | HEVC 1920×1440, sensor-landscape although the phone was held **portrait**; variable frame rate (60 Hz ticks with ~30 % skipped; mean ~46 fps) | e00, contact sheets |
| Frame alignment | odometry row *i* ↔ depth *i*; **decoded video frame *j* ↔ row *j*+1** (video lacks its first frame; confirmed by PTS gap-pattern match 1713/1713 and RGB/depth edge correlation peaking at −1) | e05 |
| Pose | camera-to-world, **OpenCV camera axes** (x right, y down, z fwd), world = ARKit gravity-aligned **+y up**, metres. ARKit-axes interpretation gives 200–400 mm reprojection residual; OpenCV gives 6–8 mm. Matches Stray source (`q_WA * q_AC`, 180° about x). | e01 |
| Intrinsics | per-frame, for 1920×1440; **fx varies ~2 % within a capture** (1581–1618, autofocus); `camera_matrix.csv` = last frame only; distortion columns empty | e00 |
| IMU | ~100 Hz, accel in g, gyro in rad/s, same clock as odometry | e00 |
| Ground truth / labels | **none** | survey |

**Content: three scans of the same apartment** (same fridge, sofa, shower visible in all)

| Capture | Duration / path | Coverage | Ceiling seen |
|---|---|---|---|
| `single_room` (misnamed) | 37 s / 14 m | living/kitchen + bathroom (partial apartment) | no |
| `single_scan_floor_only` | 115 s / 54 m | whole apartment, ~6 rooms + cross-shaped corridor | no |
| `single_scan_with_ceiling` | 215 s / 100 m | whole apartment, revisits, staircase seen (not climbed) | **yes** |

**Measured data quality**
- Short-baseline depth consistency (same surface, frames < 2 s apart): median 6–12 mm.
  This is the noise floor; it is already comparable to the 1 cm gates, so single-frame
  measurements cannot meet the gates: aggregation over many frames / plane fitting is required.
- **Drift is real and measurable.** Revisits 60–120 s apart in `with_ceiling`: median 28 mm,
  p90 47 mm (vs 12 mm short-baseline). Cross-capture rigid registration leaves residuals that
  are spatially structured: ~10 mm in western rooms, 40–60 mm in eastern rooms. A rigid
  transform cannot absorb it → it is drift, not registration error.
- Gravity tilt between captures after ICP: 0.43° (`single_room` vs others), 0.05° between
  the two whole-apartment scans.
- **Ceiling height varies by room** (clusters ≈ 2.35 m, 2.9–3.0 m, 3.05 m): per-room ceiling
  estimation is mandatory; bulkheads/soffits exist.
- **Floor is not one plane**: one region sits ~25 mm lower (likely a wet-room step);
  interior floor cells otherwise within ±10 mm.
- Hard surfaces present: glass shower enclosure, windows with reflections, glossy tiles,
  evening artificial light → realistic failure cases for the "mirrors / glass / low light"
  constraint.

**What this data can and cannot be used for**
- Can: loader/format verification, drift study, stitching/room-segmentation development,
  cross-capture consistency, and — importantly — **pseudo-ground-truth for the photo and video
  tiers**: RGB frames from the video, stripped of depth/poses, simulate those tiers, and the
  LiDAR geometry is the reference.
- Cannot: accuracy against a laser (no GT), opening-width gate, damage (none staged),
  benchmark compliance (we must build our own set).

---

## 3. Assumptions and open uncertainties

| # | Item | Status |
|---|---|---|
| U1 | Are the supplied captures representative of what the evaluator will produce at the walk-in? They follow *their* capture, not our protocol. | Unknown; our protocol defines walk-in input |
| U2 | Absolute accuracy of ARKit LiDAR + VIO without GT. Literature/forum claims ~1–2 % on distances; unverified. | Needs laser GT (physical work) |
| U3 | Whether "Repeatability ≤ 1 cm or 0.5 % per wall" means the *looser* of the two (natural reading: 0.5 % of a 4 m wall = 2 cm). | Assume looser-of-two; state explicitly in report |
| U4 | Whether stock iPhone Camera video carries any IMU track. Research says no; check on a real clip. | Needs a sample clip from the user |
| U5 | Which consumer app's free tier really exports dimensions (magicplan "Starter" claims PDF/CSV export). | Needs on-device check |
| U6 | Whether Stray Scanner v1.4 export differs from the samples (v1.4 adds distortion params). | Check with a fresh capture |
| U7 | Evaluator's machine (CPU/GPU) for the cold walk-in run. | Assume no CUDA; design for Apple Silicon/CPU |

---

## 4. Approaches considered

Candidates per subsystem, with what the evidence so far says. "Test" = experiment needed
before committing.

| Subsystem | Candidates | Evidence so far | Leaning |
|---|---|---|---|
| Capture route | R1 own app (ARKit/RoomPlan, $99 dev account, TestFlight review ~24 h); R2 Stray Scanner (LiDAR) + stock Camera (photo/video) | Stray: free, format verified here, **records nothing on non-LiDAR phones**. RoomPlan accuracy undocumented by Apple. R1 adds a Swift app to build, test and support at the walk-in. | **R2** unless the user wants R1 (question Q2) |
| LiDAR geometry | (a) fuse raw depth+poses; (b) RoomPlan output; (c) TSDF mesh | (a) noise floor 6–12 mm/frame, drift 2–6 cm over minutes → needs aggregation + drift correction | (a) with plane fitting; test TSDF vs raw points |
| Drift | as-is (auto-fail); revisit ICP + pose graph; plane-anchored (Manhattan walls / shared floor) correction; per-room local frames | Drift measured (e03); revisit and cross-capture residual are usable **GT-free drift metrics** for the ablation | pose graph over keyframe submaps with revisit edges + plane constraints; test |
| Room segmentation | protocol (one recording per room); automatic from free-space map (doorway narrowing, distance-transform watershed); ceiling-height discontinuities | Ceiling-height map already separates rooms visually (e04) | automatic on a continuous walk; test on samples |
| Layout | Manhattan/plane RANSAC on wall points → 2D polygon; RoomFormer-style learned | Walls are crisp in the top-down density (e02) | classical plane fitting first |
| Ceiling height | per-room floor + ceiling plane fit; histogram peak | Distinct per-room levels found (e04); needs ceiling sweep in protocol (floor_only has none) | per-room plane fit; protocol must require ceiling sweep |
| Openings | gaps in wall planes (free space through wall); RGB open-vocab detector (door/window) projected onto wall plane | Glass windows: LiDAR sees through / reflects (e00 contact sheets) | geometric gaps + RGB detection fused; test |
| Photo tier | MapAnything (Apache ckpt, MLX port); DA3; MoGe-2 per image + multi-view consistency | MoGe-2 per-image scale error 6.4 % median / 21 % p90 with known FOV, +4.5 % bias (e06). Literature: unposed multi-view scale ~13–16 %. Shape is good (AbsRel 1.3 %). | shape from network, scale from fused cues; test MapAnything on room-like stills |
| Photo stitch | shared views through doorways; opening matching between rooms (width + position puzzle); protocol "doorway photo" | none yet | protocol-assisted doorway photos + opening matching; test |
| Video tier | COLMAP 4 global mapper + metric-depth scale; MapAnything/DA3 on keyframes | same scale problem; many frames should average random scale error but not the bias | test both on sample video vs LiDAR pseudo-GT |
| Scale cues | metric-depth median over many views; door-height prior; EXIF focal; reference object (A4/card) | +4.5 % bias is systematic → averaging alone cannot fix it | needs benchmark; assessment-interpretation question Q4 |
| Damage | SAM 3 / Grounding DINO / OWLv2 open-vocab + VLM classify; YOLO fine-tuned on BD3 | no damage in samples | open-vocab + rules; lowest priority (not gated) |
| Intervals | per-tier empirical error model (conformal on benchmark residuals), propagated per measurement | none yet | needs GT data |

## 5. Decisions

Tentative unless marked final. Each one names the evidence it rests on.

| # | Decision | Evidence | Reversible? |
|---|---|---|---|
| D1 (final) | Stray poses are T_wc with OpenCV camera axes; world +y up; depth in mm; video frame j ↔ row j+1 | e01, e05, Stray source | n/a (fact) |
| D2 | Raw data never enters git; experiments write to `experiments/eNN_*/out/`, large binaries ignored | repo hygiene | yes |
| D3 | Python + uv; core deps light; ML deps in a separate `ml` group | M4/16 GB dev box, evaluator hardware unknown | yes |
| D4 | Use the supplied LiDAR captures as **pseudo-GT harness** for photo/video tiers (frames stripped of depth/poses) before any physical capture exists | e06 shows it works and is informative | yes |
| D5 | MoGe pinned to the last MoGe-2 commit (`b942f00`); `main` is MoGe-3 with CUDA-only deps | install failure on darwin | yes |

---

## 6. Dated entries

### 2026-10-04 — Repository survey
- Repo: the assessment PDF, three zips and their extracted directories. `.gitignore`
  excludes the zips, the extracted dirs and the PDF (so raw data never enters git).
- The three directories share one layout: `camera_matrix.csv`, `odometry.csv`, `imu.csv`,
  `rgb.mp4`, `depth/NNNNNN.png`, `confidence/NNNNNN.png`. This is the **Stray Scanner**
  (Stray Robots) iOS export format; zip root folders are Stray-style hash IDs
  (`c00a170fe1`, `1a8384c3f6`, `c7d28f72c6`).
- **No ground truth, no annotations, no labels, no photos-only or video-only captures**
  are supplied. The PDF itself says "We provide no captures" — these samples are an
  illustration of a capture format, not a benchmark.
- All three captures share one monotonic clock (device uptime seconds) and were recorded
  back-to-back on one device: `single_room` 65575.6–65612.8 s (37 s),
  `single_scan_floor_only` 65621.6–65736.4 s (115 s), `single_scan_with_ceiling`
  65764.2–65979.2 s (215 s). → plausibly the **same room scanned three times**, which
  would make them usable as an informal repeatability set. To be verified geometrically.
- Dev machine: Apple M4, 16 GB, no CUDA. Python env via `uv` (Python 3.13 venv).

### 2026-10-04 — e00 data survey (`experiments/e00_data_survey`)
- Per-capture JSON in `out/survey.json`; contact sheets `out/contact_*.jpg`.
- Depth mm/uint16/256×192, confidence {0,1,2}, 1:1 rows across odometry/depth/confidence,
  per-frame intrinsics vary ~2 %. Video declares N frames but decodes N−1.
- Visual: one apartment in all three captures; phone held portrait; glass shower, windows,
  evening light; `with_ceiling` views a staircase.

### 2026-10-04 — e01 pose convention (`experiments/e01_pose_convention`)
- Hypothesis going in: ARKit camera axes (−z forward). **Wrong.** Reprojection residual of
  depth between frame pairs: ARKit-axes 200–430 mm, OpenCV-axes 6–8 mm, world-to-camera
  readings 270–670 mm. Frame offset scan −3..+3: minimum at 0 in all captures.
- Loader (`src/roomscan/stray.py`) corrected; `tests/test_stray.py` asserts the convention
  numerically (median residual < 15 mm) so a regression cannot pass silently.

### 2026-10-04 — e02 fusion (`experiments/e02_fusion`)
- Fused every 5th frame, conf=2, ≤ 4 m, 2 cm voxels. Top-down density shows crisp walls.
- `single_room` is a partial apartment; the two `single_scan_*` captures are whole-apartment
  multi-room scans. Floor peak at y ≈ −1.4 to −1.5 m (camera at 0) → world +y up.

### 2026-10-04 — e03 drift (`experiments/e03_drift`)
- Revisit residual (pairs viewing the same surface, ≥ 1 s apart, ≤ 0.4 m and ≤ 20° apart):
  `with_ceiling` 0–2 s median 11.6 mm → 60–120 s median 27.7 mm / p90 46.5 mm;
  `floor_only` 60–120 s median 9.3 mm / p90 38.9 mm.
- Cross-capture FPFH+RANSAC → point-to-plane ICP (gravity-aligned walls/furniture band):
  RMSE ~12 mm on inliers; overlap distance median 13–19 mm, p90 41–84 mm. Residual map is
  spatially structured (east rooms 40–60 mm) → drift, not registration noise.
- Consequence: the 1 cm repeatability gate and the drift-accountability gate both require
  explicit drift handling; the two metrics above can serve as GT-free ablation measures.

### 2026-10-04 — e04 height maps (`experiments/e04_heights`)
- 0.3 m grid; floor = median of points within 15 cm of the 2nd-percentile height; ceiling =
  10 mm histogram peak above floor + 2.1 m.
- Ceiling clusters 2.27–2.43 m, 2.95 m, 3.07 m → per-room ceilings, bulkheads.
- Floor: interior within ±10 mm; one region −25 mm (wet room?). Edge cells contaminated by
  furniture/wall bases → plane fits must use interior floor only.
- `floor_only` reports "ceiling" 2.09 m: that is furniture tops, not ceiling → a pipeline must
  detect "ceiling not observed" and refuse (or widen the interval), not report furniture.

### 2026-10-04 — e05 RGB/depth sync (`experiments/e05_rgb_depth_sync`)
- PTS gap pattern of the decoded video matches odometry timestamp gaps exactly when shifted
  by one row (1713/1713, 5248/5249, 9741/9743 vs ~40 % unshifted). RGB-edge/depth-edge
  correlation peaks at the same pairing (0.31–0.34 vs 0.16–0.19 at the naive pairing).
- First tally in sync.py had an argmax bug (always reported offset −3); means were correct.
  Fixed and re-run: per-frame argmax picks −1 in 53/60, 52/60, 51/60 sampled frames.

### 2026-10-04 — e06 monocular metric scale (`experiments/e06_mono_scale`)
- MoGe-2 ViT-L (`Ruicheng/moge-2-vitl-normal`, MPS, 1.6 s/frame at 960×720), 75 frames across
  the three captures, upright-rotated via pose gravity (checked visually), compared to LiDAR on
  confidence-2 pixels.
- Scale ratio LiDAR/pred: FOV unknown median 1.087, |err| median 9.8 %, p90 23.6 %;
  **FOV given** median 1.045, |err| median 6.4 %, p90 21.2 %. Per-capture median bias 2.7–9 %.
  Shape after per-frame scaling: AbsRel 1.3 %. Predicted FOV ≈ 50.1° vs true 48.5°.
- Reading: geometry shape is nearly free; metric scale is the photo-tier problem. A single
  photo's scale is not good enough for ±8 % with calibrated intervals; multi-photo aggregation
  reduces the random part, not the ~4–9 % bias. Caveat: these are walk-through frames (many
  close-ups of walls/floor), not deliberate room photos — re-test on stills taken per protocol.

---

## Phase 2 (2026-10-05): evidence-driven prototyping

Constraint from the user: no iPhone and no Apple Developer account for now. Everything below
runs on the three supplied Stray captures. **Labels used:** [FACT] verified fact ·
[MEASURED] number from an experiment on supplied data · [PSEUDO-GT] measured against our own
LiDAR reconstruction, not a laser · [HYPOTHESIS] engineering hypothesis · [DECISION-PROV]
provisional decision · [BLOCKED] needs physical capture.

### e07 — shared keyframe cache
Every 5th row of each capture, 960×720 JPEG + per-frame K, T_wc, upright rotation, pitch,
LiDAR median depth, sharpness, angular speed (`cache/frames/`, regenerable, gitignored).

### e11 — video-tier metric scale from the IMU
- Model: s·p̈ − g + R b − R A(t) r = R f (scale s, gravity g, accel bias b, **camera–IMU lever
  arm r**), same low-pass on both sides; camera–IMU rotation and time offset estimated from
  gyro vs trajectory angular rate (residual 0.011–0.015 rad/s; offset 0 to −5 ms).
- [MEASURED] Without lever arm, scale biased low 4–44 %; errors-in-variables attenuation
  explains part (forward/reverse regressions bracket) but both stayed low → model error.
- [MEASURED] With lever arm (estimated 4–10 cm, x ≈ 8 cm consistently), at 9 fps poses and a
  0.4–0.6 s filter, forward/reverse bracket 1 within ±2 % on all three captures.
- [MEASURED] At 3 fps keyframes (SfM-like) the geometric-mean error is −2.5 … +3.6 % with no
  noise and 1–4 % with 5–10 mm jitter; shortest capture (37 s) worst.
- [HYPOTHESIS] Video + IMU can reach ~1–2 % scale if SfM poses are dense (≥ 10 fps) and the
  clip is ≥ 60 s. [BLOCKED] needs a real Sensor Logger recording (timestamp format, IMU rate).
- [FACT] (research) Sensor Logger (free) records video + accelerometer + gyro on any iPhone on
  one clock; native Camera video carries no IMU. Interpretation risk: video+IMU is
  "visual-inertial", which may not count as "a handheld walkthrough clip".

### e12 — photo-tier metric scale (MoGe-2, FOV given), 359 photo-like frames, 6 rooms [PSEUDO-GT]
- Photo-like = |pitch| < 25°, median depth > 1.6 m, sharp, slow; grouped by segmented room.
- [MEASURED] Per-photo size error: signed median −5.3 %, |err| median 5.8 %, p90 17 %.
- [MEASURED] K photos per room (median of per-photo scales, 200 draws/room): |err| median
  6.6 / 6.9 / 5.6 / 6.4 / 6.2 / 6.2 % for K = 1/2/3/4/6/8; within ±8 %: 57–65 % of rooms.
  **More photos do not help**: per-room bias ranges −0.1 % … −15.7 % (small rooms worst).
- [MEASURED] Leave-one-room-out bias correction (K=4): within ±8 % rises to 78 %, p90 11 %.
- [MEASURED] Camera-height cue (lowest horizontal plane in the lower image): single-image
  height error 21 % median; fusing it makes results worse (55 % within ±8 %). Operator's own
  camera height varies ±9 cm (6 %). → **refuted** as a scale cue.
- [MEASURED] Door-height cue (Grounding DINO "a door." + wall plane fitted around the box):
  zero-shot boxes are mostly not full doors (LiDAR heights 0.86–3.12 m) → deployable version
  fails (per-room errors up to 64 %). Oracle (true doors only): ~5 % per photo, per-room within
  ±6 % (2.1 m prior). Doors here are 2.00–2.23 m → prior spread alone is ~±5 %.
- [HYPOTHESIS] Without a reference object, single-model monocular scale lands at ~6 % median /
  ~15 % p90 per room. The ±8 % gate with *calibrated* intervals then needs either (a) a second
  independent scale source whose bias is uncorrelated (multi-model ensemble, MapAnything), or
  (b) a protocol-level metric cue. Honest 90 % intervals would be roughly ±15 %.

### e10 — room segmentation [MEASURED, visual check only]
- v1 (height bands): corridor's low ceiling filled the "header" band; furniture removed free
  space → wrong. v2: barrier = **vertical** surfaces (normals) above 1.9 m; door lintels close
  doorways; interior bounded by the observed-floor footprint → 8 regions on `with_ceiling`,
  clean rectangles for closed rooms, insensitive to the 1.6–2.05 m threshold.
- [MEASURED] `floor_only` / `single_room`: vertical points above 1.7 m are < 1 % → 0 rooms.
  → [DECISION-PROV] the protocol must include an upper-wall + ceiling sweep.

### e09 — drift correction ON vs OFF (GT-free metrics)
- v1: submap pose graph (2.5 s submaps, ICP loops, Open3D LM). [MEASURED] cross-capture median
  21.4 → 16.4 mm (better) but revisit p90 46 → 72 mm and `floor_only` revisit median
  9.3 → 21.3 mm (worse).
- v2 (degenerate-loop rejection, consistent odometry information, yaw-only corrections):
  fitness ≥ 0.5 rejected almost all loops; one accepted loop moved `floor_only` poses 44 cm.
  [MEASURED] worse. → **negative result**: a generic pose graph on top of ARKit is not
  reliably better than ARKit poses.
- Per-room rigid test: after global alignment of `floor_only` onto `with_ceiling`, per-room ICP
  brings median disagreement to 11–15 mm in 7/8 rooms (room shifts 2–16 cm, yaw ≤ 1.8°).
  [MEASURED] drift ≈ rigid per room, accumulating between rooms.
- [DECISION-PROV] Drift strategy: **room-local geometry** (each room measured within one
  visit, so drift cancels out of wall-to-wall distances) + **plane-anchored stitching** (rooms
  placed via shared walls/doorways and a common Manhattan yaw). Ablation for the gate: stitched
  footprint with ARKit placement only vs with plane-anchored stitch. Not yet implemented.

### Representation + benchmark harness
- `src/roomscan/model.py` → `schema/plan.schema.json`. Every user-facing number is a
  `Measurement` (value, unit, 90 % interval, status, method, error budget). Unobserved
  quantities are `not_observed`, never guessed (e.g. `floor_only` ceiling). Damage regions,
  concealed-damage flags and scope items reference `Surface.id`, so damage attaches to stable
  geometry later. Cross-references are validated.
- `src/roomscan/bench/` + `benchmark/README.md` + `benchmark/_template/site.yaml`: tape/laser
  GT as per-room cyclic wall lists, openings (clear jamb-to-jamb width), ceilings (≥3 points),
  topology, damage, capture manifest with `repeat_group`, incumbent numbers.
  `uv run roomscan-bench benchmark/<site> --run runs/<id>` computes every gate, repeatability,
  interval coverage, head-to-head; pseudo-GT sites get a PSEUDO-GT banner. 8 tests pass.

### e14 — LiDAR pipeline v0 (`src/roomscan/lidar_pipeline.py`)
- fuse → segment → per room: Manhattan yaw, wall lines (density peaks with support reaching
  ≥ 2 m), arrangement-cell polygon grown across edges without structural support, floor and
  ceiling levels from all visits, geometric doorways (lintel present, no mid-height wall).
- [MEASURED] `with_ceiling`: 7 rooms, consistent yaw 26.8–28.8°, ceilings 2.28–3.08 m,
  4 doorways (0.79, 0.88, 0.88, 0.51 m), ~50 s on M4. Schema-valid Plan.
- [MEASURED] Within-capture repeatability proxy (room walls from visit 1 vs visit 2):
  median |Δ| 16 mm, 50 % within max(1 cm, 0.5 %). Rigid drift cancels in these distances, so
  this is **method noise** (furniture faces vs wall faces, partially observed walls).
  [HYPOTHESIS] Averaging all visits after per-room rigid alignment + per-wall plane fitting
  will reduce it. Fix-loop candidate.
- Intervals are an explicit **uncalibrated** placeholder (`*_uncal_v0`) until laser GT exists.
- [MEASURED] Second monocular model (Depth Anything V2 Metric-Indoor Small, no FOV input):
  +44 % per-photo bias (unusable as-is), and per-room biases correlate with MoGe-2's at
  r = 0.75 → biases are **scene-driven, not model-specific**; a multi-model ensemble does not
  cancel them (K=4 ensemble 16 % within ±8 % vs MoGe alone 64 %). Negative result.
- [MEASURED] IMU scale on **real COLMAP poses** (e08 output, `imu_on_sfm.py`): largest models
  cover only 6–15 s of `single_room` (fragmented), Sim3 ATE 12 mm … 1.1 m; `with_ceiling` model
  ATE 3.1 m (broken). Scale error 16–200 %. → the IMU method is sound on good poses (e11) but
  the **bottleneck is RGB-only tracking** on this footage (white walls, blur, close-ups).
  [HYPOTHESIS] video tier needs a robust learned tracker (MapAnything/VGGT-class poses) or a
  protocol that yields trackable footage (slower, wider views); test before committing.

### e13 — MapAnything (Apache ckpt `facebook/map-anything-apache`, repo v1.1.4) [PSEUDO-GT, room 1 only]
- Run by a subagent in an isolated venv (`cache/e13/.venv`; mapanything pins
  opencv-python-headless 4.10, conflicting with the project's 5.x). Stopped early under memory
  pressure on the 16 GB M4 (swap thrash); 153 runs, room 1 only, K = 1…6 (K = 8: ≤ 2 runs).
- [MEASURED] Raw metric scale: |err| median 28–35 %, always too small (LiDAR/pred ≈ 1.45–1.53);
  0/76 multi-view runs within ±8 %. Passing intrinsics barely helps (median 1.9 pp).
- [MEASURED] Cause: predicted FOV ≈ 70° vs true 48.5° (focal −33…−37 %), even on centre crops;
  given intrinsics move the output focal only ~6 % (upstream issue #93 reports the same).
- [MEASURED] Focal-corrected (depth × f_true / f_pred, f_true from EXIF in deployment):
  |err| median 3.9 / 5.7 / 4.1 / 3.0 / 1.5 % and p90 15.7 / 11.7 / 8.2 / 7.2 / 4.6 % for
  K = 1/2/3/4/6. Shape AbsRel 0.05–0.07. Runtime 1.2–10 s per run (K = 1–6) on MPS, ~40 s load.
- Caveat: room 1 is MoGe-2's *easiest* room (bias −3 %), so this does not yet show MapAnything
  beating MoGe where MoGe fails (rooms 6, 8: −11…−16 %). [HYPOTHESIS] multi-view consistency +
  EXIF focal reduces per-room bias; must be re-run on rooms 3/4/6/7/8 before any decision.
- Stitching probe (mixed-room subsets) not run.
- Operational: on this 16 GB machine, MapAnything (4.9 GB weights) + other GPU jobs exhaust
  memory → run one model at a time; affects walk-in hardware assumptions (see U7).

### e08 — video tier: COLMAP (pycolmap 4.2.1, CPU, SIFT) on RGB only [PSEUDO-GT]
- [MEASURED] Tracking fragments: `single_room` 7 models (largest 78 frames, ATE 17 mm);
  `with_ceiling` 14 models, longest continuous run ≤ 13.5 s, largest model ATE 1.47 m with
  per-20 s scale varying 0.80–1.14. Breaks where the phone turns > ~80°/s within 0.2–0.7 m of
  a blank wall. Default SIFT gives ~429 keypoints/frame on these walls (threshold lowered → 5.8k).
  Global mapper registers more frames but its positions are wrong (ATE 0.5–3.2 m).
- [MEASURED] Inside unbroken runs: ~9 Hz poses, jitter median 1.4–5 mm (p90 ≤ ~10 mm).
- [MEASURED] MoGe-2 scale per SfM piece (`single_room`): −2.5 / +1.9 / −2.2 / −1.8 / +3.5 %
  (one failed piece +45.6 %). Pooled convergence 0.6 % median at 50 frames is optimistic
  (pieces' errors cancel); a single reconstruction keeps its own ±2–4 %.
- [MEASURED] Pseudo-GT self-consistency: LiDAR depth vs depth triangulated from ARKit poses +
  true intrinsics differ by ~3 % (ratio 1.009–1.04, median 1.03). **Pseudo-GT cannot resolve
  scale errors below ~3 %.** This also means ARKit LiDAR and ARKit trajectory scale may disagree
  at the percent level → [BLOCKED] needs laser GT; directly relevant to LiDAR-tier 1 cm gates.
- [MEASURED] IMU scale on corrected per-piece SfM poses (6–13 s pieces, `imu_on_sfm_v2.py`):
  errors −26 … +19 %, best pieces 2–5 %. Consistent with e11: the IMU method needs long
  continuous trajectories (≥ ~60 s); COLMAP does not deliver them on this footage.
- [HYPOTHESIS] The video tier's binding constraint is **continuous RGB-only tracking**, not the
  scale estimator. Candidates to test next: learned trackers (MapAnything/VGGT-class on
  keyframe windows, MASt3R-SLAM-class), plus protocol (slower turns, ≥ 1 m from walls, wider
  framing). With continuous tracking, both MoGe-over-many-frames (±2–4 %) and IMU (±1–2 %)
  become viable scale sources.
- Operational: pycolmap and torch conflict on OpenMP in one process → separate processes.

---

## Phase 3 (2026-10-05): attack the three highest-value risks

User constraints for this phase: no iPhone / developer account; video contract = plain clip (IMU
stays a diagnostic); no mandatory reference object. GPU work runs one model at a time.

### e18 — LiDAR repeatability: room-local aggregation + robust planes
**Hypothesis.** The 16 mm within-capture method noise (e14) comes from (a) using one visit's
points, (b) furniture faces and the far side of walls being accepted as wall faces, (c) unstable
topology; aggregation of aligned visits + interior-facing robust planes + topology regularisation
pushes it toward the 1 cm gate.
**Variants** (`experiments/e18_lidar_repeat/roomgeo.py`): V0 baseline (e14 logic); V1 all visits,
per-room ICP to the longest visit; V2 V1 + camera-oriented normals (a face only supports the room
it faces) + robust offsets from points 0.9 m–ceiling within the edge's own extent; V3 V2 + merge
same-facing lines < 12 cm + absorb notches < 15 cm.
**Design iterations (kept as evidence):**
1. Visit split (A = even visits, B = odd): [MEASURED] 28/28 half-geometries failed — most halves
   have **no observations above 2 m**, so no wall passes the structural-support test
   (`out/split_visit_*`). A split by whole visits is harsher than two real captures, each of which
   would include the sweep.
2. Chunk split (2 s chunks alternating inside every visit): V1–V3 still failed in most rooms.
   Cause: aligned aggregation **dropped visits whose ICP check failed**, and those included the
   ones carrying the ceiling sweep. [DECISION-PROV] keep unaligned visits (flagged) instead of
   dropping them — observation selection must not remove coverage.
3. With that fix, independent halves still mostly fail to find all walls (rooms 2, 4, 6, 8: too few
   wall lines). [MEASURED] **Topology (wall existence) is coverage-hungry**: on this capture,
   half of the frames is not enough upper-wall evidence for most rooms. → protocol consequence:
   every wall's top must be in view at least once per capture (not just "sweep the ceiling").
4. Therefore the study is split into **topology stability** (above: fails with half coverage) and
   **position repeatability given topology** (`run_fixed_topology.py`: topology from all data,
   each half ICP-aligned onto the full room cloud, every wall re-measured per variant).
**Results** (`experiments/e18_lidar_repeat/README.md`, `out/fixed_topology_*.json`) [MEASURED, GT-free]:
position repeatability given topology, 2 s chunk halves (each ~50 % of the data):

| Variant | wall Δ median / p90 | walls within max(1 cm, 0.5 %) | ceilings ≤ 1 cm |
|---|---|---|---|
| V0 baseline | 10.2 / 36.7 mm | 57 % | 50 % (one 110 mm wrong-level pick) |
| V1 aggregate (unoriented) | 11.2 / 49.8 mm | 44 % | 25 % |
| V2 + oriented normals, band, extent | 11.0 / 27.6 mm | 65 % | 75 % |
| V3 + merge/notch regularisation | 11.5 / 44.2 mm | 62 % | 75 % |
| V4 outermost peak | 19.5 / 71.0 mm | 31 % | 75 % |
| V5 close-range only (≤ 2.5 m, ≤ 60°) | 7.5 / 25.7 mm (3 rooms; 4 topology failures) | 61 % | 0 % (ceilings beyond range) |
| **V6** V2 topology + close-range positions | **9.8 / 33.5 mm** | **73 %** | 75 % |

**Interpretation.** Aggregating visits only helps once normals are camera-oriented (V1 < V0 < V2):
without orientation, furniture backs and the far faces of shared walls compete with the true
face. Close-range, near-normal observations measure walls better (V5) but cannot carry topology
or ceilings, hence the hybrid V6. Regularisation (V3) and outermost-peak (V4) are negative
results. Per room, rectangular rooms (3, 4, 8) are already 0–10 mm per wall with half the data;
complex rooms (1: corridor merged with a room; 7: wardrobes) hold the 33–45 mm residuals; small
rooms 2 and 6 fail topology even with all data.
**Confidence/limitations.** One capture, ~5 rooms, halves share drift (removed by ICP), no GT.
**Decision** [DECISION-PROV]: production LiDAR geometry = V6 (oriented fusion, per-room ICP
aggregation that keeps unaligned visits, V2 topology, close-range wall positions with fallback,
horizontal-normal floor/ceiling). Topology stability is the open LiDAR risk; it needs full
upper-wall coverage → protocol rule strengthened from "sweep the ceiling" to "every wall's top
edge in view at least once".

### e19 — drift: raw placement vs plane-anchored stitch [MEASURED, GT-free]
**Hypothesis.** Common Manhattan yaw + shared-wall + doorway constraints remove inter-room drift
from the stitched footprint. **Method.** Shared room shapes; two placements of every multi-visit
room from disjoint visit halves; disagreement after one rigid fit (`experiments/e19_drift_stitch/`).
**Result.** Raw ARKit: corners 54.8 / 96.8 / 115.8 mm (median/p90/max). Yaw snap: 66.0 / 84.4 /
93.8. Doorway constraints: 0 of 4 doorways measurable from both sides. Shared-wall translation
stitches: worse (69–90 mm median); one common wall thickness moved a room ~19 cm. A circular metric
(shared-wall gap spread, which the stitch optimises) was caught and dropped.
**Interpretation.** On this data the plane-anchored translation stitch is a **negative result**;
yaw regularisation is inconclusive (better tails, worse median). Drift does not reach room
measurements because they are room-local (e18), so the drift story is: *room-local measurement
(drift-immune dimensions) + yaw regularisation for plan regularity + an honest per-room placement
uncertainty (σ ≈ 33 mm from these halves)*. **Confidence.** Low–moderate: 5 rooms, one session.
**Decision** [DECISION-PROV]: production layout = `arkit_yaw` (chosen for regular, parallel walls
and smaller worst case, not for a measured median gain); `arkit` kept selectable for the gate's
on/off ablation. Revisit with two real captures per protocol ([BLOCKED]).

### LiDAR production core v1 (`src/roomscan/lidar/`)
Stages: `fusion` (camera-oriented normals, observation selection) → `rooms` (segmentation,
visits, per-room ICP keeping unaligned visits) → `walls` (V6) → `levels` (horizontal-normal floor/
ceiling) → `openings` (lintel doorways; jamb-gap width per room) → `layout` → `build` (Plan with
floor/ceiling surfaces, uncalibrated intervals with explicit budgets) → `render`. CLI:
`uv run roomscan-lidar <stray_dir> <out_dir>` (36 s on `with_ceiling`).
[MEASURED] `with_ceiling`: 5 of 8 labels produce rooms (2 and 6 fail topology, 5 is outside); 2
doorways, one jamb-measured (0.75 m vs coarse 0.79 m); r4 ceiling reported `not_observed` (135
points). v0 (`lidar_pipeline.py`) is kept unchanged as the regenerable baseline.
Schema: `Interval.calibrated` / `calibration_ref` added; `error_budget` keys prefixed `shared:`
denote room-correlated terms (photo-tier scale); evaluator emits `calibration_residuals` tagged by
gt_kind so pseudo-GT rows can never feed a calibration. Tests: 13 pass (incl. synthetic wall
extraction with an outward-facing full-height clutter face).

### e16 — does MapAnything + true intrinsics generalise? [PSEUDO-GT] (subagent, `experiments/e16_photo_scale_generalization/`)
**Hypothesis.** e13's room-1 result (focal-corrected MapAnything 1.5–5.7 % median) holds across rooms.
**Method.** Rooms 1,3,4,6,7,8 × K 2–8 × 10 subsets (300 runs), MoGe-2 on identical subsets.
**Result.** [MEASURED] Focal-corrected MA per room (median |err|): 1.3–6.6 (1), 2.1–5.0 (3),
2.5–6.5 (4), 2.8–6.6 (6), **31–44 (7)**, 10–12 (8) %; catastrophic runs 48/300 (45 in room 7).
MoGe-2: 1.2–2.7 (7) but 13–18 (6), 11–12.5 (8) %; catastrophic 1/300.
**Cause (Part B).** f_pred/f_true = 0.654 ± 0.037 over 1380 views, independent of the true FOV
(crop probe: true 48.5/37.3/25.4° → predicted 70.5/67.9/67.6°); per-run correlation of log depth
ratio and log focal ratio 0.04. MA effectively assumes ~70° FOV and a scene-size prior; "focal
correction" ≈ a ×1.5 constant. MA pose scale is room-dependent (baseline ratio 0.72…3.15).
Held-out constant (learned on 1/3/4, applied to 6/7/8) beats per-view EXIF correction
(8.0 vs 11.4 % median). **Interpretation.** No evidence that EXIF intrinsics give MA metric
scale; the improvement is a fixed prior that fails where the scene departs from it (mirrors).
**Decision.** Do not use MA's EXIF-corrected depth as a metric source on its own.

### e21 — two-family ensemble [PSEUDO-GT, LORO]
[MEASURED] √(MoGe × MA·c): median 5.3 %, p90 14.0 %, 64 % within ±8 %, **0 catastrophic**
(MoGe 7, MA 27–49). Fails in rooms 6 (both low) and 7 (mirror; MA high). Fixed ±14 % intervals
give 85 % coverage; disagreement-adaptive intervals do not improve calibration; per-room
coverage in the failing rooms 32–66 %. **Interpretation.** The strongest defensible photo-scale
strategy found without a reference object is the two-family ensemble with honest ≈ ±15 % (90 %)
intervals; it meets ±8 % in typical rooms but not in small/mirror-dominated rooms, and nothing
observable so far predicts those failures. [DECISION-PROV] photo tier = ensemble + wide
calibrated-later intervals; flag that the ±8 % gate is at risk without an additional metric cue.

### e17 — photo-folder stitching without hidden information [PSEUDO-GT eval only] (subagent)
**Hypothesis.** A doorway photo (standing in A's doorway looking into B) lets independent room
folders be linked. **Method.** Joint MapAnything on 4 A + 0/1/2 doorway + 4 B photos; success =
median B camera error < 0.5 m and yaw < 15° after a Sim(3) fit on A's cameras only (fit on all
cameras reported as a diagnostic). SIFT inlier counts as a cheap link signal.
**Result.** [MEASURED] 0/36 runs succeed; doorway photos change B's error by ≤ 0.45 m and do not
fix orientation; even the all-camera diagnostic leaves B 0.4–1.2 m off (35–100° for 2 pairs).
SIFT: doorway→B links clearly above the control floor for 1/4 pairs, and that pair was linked
anyway via near-duplicate frames. **Interpretation.** Single doorway photos are insufficient; the
joint feed-forward reconstruction itself is wrong across rooms. [HYPOTHESIS, untested] a short
overlapping chain through each doorway (≈ 3 photos with ≥ 50 % overlap of the same door frame:
inside A, in the doorway, inside B looking back) — fits within 2–8 photos/room but must be
tested on real photos. Caveat: walkthrough frames are not real stills; doorway frames for pair
3-4 were physically inside room 4.
- Schema: `Plan.quality_flags` added (concrete limitation: `single_room`, captured without an
  upper-wall sweep, produced an empty plan with no explanation). v1 now reports e.g.
  `upper_walls_not_observed`, regions without geometry, unobserved ceilings, walls placed from
  topology only — the walk-in operator gets a re-capture instruction instead of a silent gap.
  [MEASURED] `single_room` → 0 rooms + 2 flags; `floor_only` → 1 merged region (ceiling not
  observed); `with_ceiling` → 5 rooms + 5 flags.

### e20 — video tier: can any RGB-only tracker give a continuous walkthrough? [PSEUDO-GT = ARKit]
**Hypothesis.** The video tier's binding constraint is tracking continuity (e08); a learned
matcher, feed-forward windows or depth-based odometry yields one accurate walkthrough.
**Method** (subagent; `experiments/e20_video_tracking/README.md`): COLMAP with ALIKED+LightGlue;
MapAnything 16-frame windows chained by Sim(3); pseudo-RGB-D odometry on MoGe-2 depth (Open3D,
PnP with fallback ladder, PnP + pose graph); failure-anatomy analysis. Plain RGB only, no IMU.
**Result** [MEASURED], `with_ceiling` (215 s): COLMAP-SIFT 14 pieces (largest 40 %) but 20 s-window
ATE 4.4 cm; COLMAP-LightGlue 1 piece covering 65 % but ATE 2.97 m; depth odometry 1 piece / 100 %
but ATE 2.6–3.0 m, end drift 9–15 m, long-range distance error 44–59 % median; PnP cut at failed
links: 10 pieces, longest 62 s, long-range error 7.2 % median. MapAnything windows (`single_room`):
41 % long-range error. **Failure anatomy:** only 1.8 % of links fail, in ≤ 1.2 s bursts, during
fast turns (53 vs 25 °/s) close to surfaces (0.86 vs 1.59 m); bridging them with fallback motion
models adds 4–13° each, and small per-link rotation errors (0.3°) accumulate over ~900 links.
**Interpretation.** No candidate is both continuous and accurate. Continuity is cheap (depth
odometry) but accuracy needs real correspondences; the walkthrough breaks at a few identifiable
moments caused by capture behaviour, not by the scene. **Confidence.** Moderate on the diagnosis
(consistent across methods), low on numbers (one recording not made per our protocol).
**Decision** [DECISION-PROV]: video tier = accurate local pieces (COLMAP/LightGlue or PnP) with
room-level measurement inside pieces + protocol rule *turn slowly and keep ≥ 1 m from walls while
turning*; drop MapAnything windows (least accurate, and its load starved the 16 GB machine — not
retried). [BLOCKED] whether a protocol-following clip tracks continuously needs a real capture.
Operational: the subagent stalled (stream watchdog) during a MapAnything load under memory
pressure; its saved outputs were complete for every run except MapAnything on `with_ceiling`.

---

### 2026-10-05 — Handover at session compaction
`HANDOVER.md` created at the repository root as the authoritative state snapshot (commit
`3b4dd57`, clean tree): implemented vs experiment-only subsystems, the full experiment table with
negative results, measured numbers with evidence labels, current decisions, a reconciled
assessment status (it corrects the compliance matrix where code had moved ahead: one command per
capture, stitched LiDAR plan, openings are PARTIAL/EXPLORING; GT lives in `site.yaml`, not
`benchmark/ground_truth/`), files that must not be lost, environment and regeneration order for
ignored intermediates (e02 PLY → e10 labels → e12), the prioritised next-work queue, and a
do-not-repeat list. Reproduction gaps recorded there: `cache/e20/.venv` has no setup recipe;
MapAnything/LightGlue live in isolated venvs outside `uv.lock`; weights download on first use.

---

## Phase 4 (2026-10-05): from experiments to a three-tier product path

User priorities for this phase: (1) LiDAR robustness incl. the two dropped rooms, (2) a real photo
frontend, (3) a real video frontend, (4) one shared representation/evaluator/render, (5)
reproducibility fixed as part of the implementation. Still no iPhone / developer account.

### Reproducibility recipes (subagent, commit 63470ba) [FACT]
`envs/{mapanything,video}/requirements.txt` (pinned freezes of the e13/e20 venvs) and
`scripts/setup_envs.sh` rebuild both isolated envs under `.envs/`; verified from scratch (39 s /
1.5 GB, 33 s / 845 MB with a warm uv cache; smoke imports pass, MPS available). `envs/weights.yaml`
+ `scripts/fetch_weights.py --check` find all 8 pretrained weights offline with sha256 checks.
`experiments/DEPENDENCIES.md` maps every experiment's inputs and producers;
`scripts/prepare_intermediates.sh` regenerates ignored intermediates in order (dry-run verified,
real steps not re-executed). [FACT] `import pycolmap, torch` in one process aborts (OMP Error #15)
in either order → subprocess isolation stays; `roomscan.envs.python_for()` resolves interpreters.
Unpinnable: DINOv2 hub code (pinned to commit `7764ea0f` by content match), ALIKED URL (sha256 only).

### e22 — LiDAR regions 2 and 6 [MEASURED, GT-free] (`experiments/e22_lidar_small_rooms/README.md`)
**Hypotheses considered:** insufficient upper-wall evidence, thresholds, segmentation, filtering,
partial observations, scene effects. **Finding:** two general causes plus one fragility.
1. *Inter-visit misregistration beyond ICP range* (region 2): two visits 22 cm apart along one
   axis (u-faces agree to 1 cm, both v-faces offset +22/+23 cm), one seeing only < 1.52 m and the
   other only 1.3–2.29 m. ICP's 5 cm radius cannot bridge it; merged unaligned, every wall becomes
   two half-height faces that each fail the extent test. Face agreement over all rooms: raw visit
   disagreement 30–160 mm — larger than e03's 28 mm revisit median suggested.
2. *Missing evidence* (region 6): one wall never observed above 1.31 m in any visit.
3. *Knife-edge topology test*: with better registration, room 3's main wall failed by 1 cm of
   p98 height (1.99 vs 2.00 m).
**Fix tested on all rooms** (production code, e18 halves protocol): verified re-registration
everywhere recovers room 2 but loses room 3 and drops within-gate walls 65 → 50 %; inferring
missing sides everywhere recovers 2 and 6 but grows correct rooms (room 7 +1.9 m²). **Trade-off
recorded.** [DECISION-PROV] LiDAR **v1.1**: per-room escalation ladder applied only when topology
fails (v1.0 → re-registered visits → inferred sides), each step flagged; inferred faces σ 10 cm
with `status: inferred`, topology-only faces σ 3 cm (v1.0 gave them 7 mm — overconfident).
[MEASURED] 7/7 non-empty regions → rooms (was 5/7), doorways 2 → 4; rooms that already worked are
unchanged; `--no-ladder` reproduces v1.0 geometry exactly. Production-code repeatability (e18
protocol): v1.0 65 % walls / 50 % ceilings within gate vs e18 experiment-code V6 73 % / 75 % —
the production numbers are the ones to quote (difference: half construction; room 3's bimodal
ceiling flips 104 mm between halves).

### Shared plan assembly (`src/roomscan/assemble.py`)
Every tier now reduces to the same inputs (room polygon, per-face σ and observed/inferred status,
relative scale σ, ceiling ± σ, openings); intervals of lengths, areas and perimeter come from a
finite-difference Jacobian w.r.t. face offsets plus the scale term (reported as `shared:scale`).
Schema change (additive, justified by the photo tier): `Room.placed` (False = polygon in the room's
own frame, position carries no information).

### Synthetic capture + smoke test; segmentation border bug [FACT/MEASURED]
`src/roomscan/synthetic.py` renders an exact two-room Stray export (4.0×3.0 and 2.5×3.0 m,
0.8 m doorway with lintel, 2.5 m ceiling). **It exposed a real bug:** `geometry.segment_rooms`
closes the floor domain with a 0.6 m `binary_closing`, and scipy treats outside the grid as
empty, so rooms near the scan border were eroded (0.27 m at the low edges, 0.53 m at the high
edges; v0 pads 0.3 m / 0 m). With the 0.5 m room crop that removed the north walls of both
synthetic rooms (z-extent 2.5/2.0 m instead of 3.0, walls "inferred"). Fix: padding parameters
(v0 defaults kept; v0 plan values verified identical) and v1 pads 1 m. Synthetic result after the
fix [MEASURED]: walls −2.0…−2.2 mm, ceiling −1.2 mm, areas −0.1 %, doorway +1 cm (1 cm-bin
quantisation: a fix-loop candidate). On `with_ceiling` the fix is a no-op (≤ 1 mm). The smoke test
(`tests/test_smoke_pipeline.py`) runs synthetic capture → `roomscan-bench --execute` →
`roomscan.lidar.cli` subprocess → scored plan, without the unshipped sample data.

### e23 — photo frontend [PSEUDO-GT] (`experiments/e23_photo_frontend/README.md`)
Production `src/roomscan/photo/` consumes only per-room folders of stills with EXIF (integer
35 mm-equivalent focal). [MEASURED] MapAnything's joint poses decide everything: with
view-diverse low-overlap photos (6 at ~60°) the relative rotation error is 44–66° median in 3 of 6
rooms (flips of ~175° elsewhere) and the fused room is incoherent (camera heights 1.4–2.6 m);
with overlapping sweeps (consecutive ≥ 22°, ~50 % overlap) it is 2–7° in 5 of 7 rooms and the
clouds are coherent — but e24's wall-level scoring shows r3's matching area (8.86 vs 8.93 m²) is a
coincidence (walls 5.73 × 1.58 vs 3.0 × 3.0 m). Single photos cannot measure a room (26/36 give
no region). r8's photos show no floor → `no_floor`. [HYPOTHESIS, needs a device] protocol for the
photo tier: 0.5× ultra-wide landscape, 5–6 overlapping photos turning on the spot, floor and
ceiling edge in each — 2–8 portrait main-camera photos cannot both cover 360° and overlap.
[DECISION-PROV] photo frontend = MoGe-2 geometry placed by MapAnything poses + e21 ensemble scale
+ shared room step; rooms unplaced; σ scale 8.7 % (shared), faces 3 cm / inferred 15 cm.

### e25 — video frontend [PSEUDO-GT] (`experiments/e25_video_rooms/README.md`)
[MEASURED, negative] Measuring rooms inside e20/e08 tracked pieces fails even with the true
intrinsics: PnP-on-MoGe pieces drift ~25 cm per 20 s and smear walls (r3 81 m² vs 8.9 m²);
COLMAP-SIFT pieces mix frames from across the walk; their GT-free scale is off −12…−30 %.
[DECISION-PROV] video frontend v0 = the clip as a sequence of 12 s windows of 8 frames, each
measured like an overlapping photo set, accepted only when GT-free coherent (camera heights above
the reconstructed floor agree within 15 cm), unplaced, consecutive same-size windows merged.
FOV is estimated from the frames (MoGe median: 777.5 px vs true ≈ 795 px, −2.5 %).
[MEASURED] `with_ceiling` clip: 18 windows → 3 coherent rooms (9 incoherent, 3 no floor, 3 no
region/polygon), 543 s on M4. Fragmentation is explicit in `quality_flags`.

### e24 — one evaluator path for all three tiers [PSEUDO-GT] (`experiments/e24_three_tier_eval/README.md`)
`roomscan-bench --execute` runs raw captures through each tier's production CLI (subprocess per
capture) and scores them against `benchmark/pseudo_gt_lidar_v11` (our LiDAR v1.1 plan; never
scored against itself). [FACT] All 5 captures (2 independent LiDAR, 2 photo sets, 1 video)
execute and produce schema-valid flagged plans. [PSEUDO-GT] Walls within tier gates: photo 0 %
(±8 %), video 0 % (±3 %); photo median wall error 272–680 %; video rooms by true identity −60 to
−100 % area; independent LiDAR captures without an upper-wall sweep: 1 merged room each, 18 %
wall / 23–35 % area error, as predicted (e10/e18), and flagged. Interval coverage 0–67 % vs 90 %
nominal: sigmas describe measurement noise, not topology/registration failure.
**Evaluation-integrity corrections:** (1) size-only room matching paired video rooms with the wrong
rooms (reported 13 % area error; true −60…−100 %) → matches now carry `method`
(map/label/perimeter) and size-only matches are flagged identity-unverified; (2) labelled rooms
whose label is not a scored room are no longer size-matched; (3) the photo-sweep r3 area match
(1.2 %) was a coincidence of a wrong shape (5.73 × 1.58 vs 3.0 × 3.0 m) — wall-level scores are
the honest view. Video window results are not numerically stable across uncached runs.

### 2026-10-05 — Phase 4 close and handover update
Phase-4 completion targets met: LiDAR v1.1 with an explained/fixed small-room failure (e22) and a
segmentation border fix (synthetic); photo and video frontends emitting shared-schema plans; one
evaluator path (`--execute`) for all tiers (e24); isolated envs and weights reproducible from
recipes. Accuracy of the photo/video frontends on the supplied data is far from the gates
(pseudo-GT) and the open questions behind that are capture-behaviour questions → the short
`dev_` capture is now the highest-value next step (HANDOVER §12–13). Protocol hypotheses doc
updated with the e23/e25 photo and video rules. `HANDOVER.md` rewritten for the new state.

---

## Phase 5 (2026-10-06): prepare the development capture

User direction: no new architecture; fix determinism first, make uncertainty respond to weak
geometry, make loaders ready for real iPhone files, prepare a 30-minute development experiment
and its evaluation, protect the benchmark from tuning leakage, then stop. Phase-4 record carried
forward: photo/video 0 % of walls within gate (pseudo-GT, e24); size-only room matching had
paired video rooms with the wrong rooms; the supplied captures are not evidence for the proposed
future protocol.

### e26 — determinism [FACT/MEASURED] (`experiments/e26_determinism/README.md`)
**Correction of phase 4:** the "two uncached video runs differ" result was not nondeterminism.
MoGe-2 and MapAnything outputs are **bitwise identical** across the two processes in all 19 view
sets; re-running the room step on run 1's outputs reproduces run 1 with the old segmentation
padding and run 2 with the new one — the padding fix landed while run 1's process was running.
LiDAR repeat runs: structure identical, max numeric difference 2.3e-11 m (Open3D summation order).
[DECISION] every plan records `provenance` (commit, dirty flag, libraries, parameters); the
evaluator warns on mixed/dirty code; `bench/compare.py` judges repeatability materially
(tolerance), `tests/test_repeatability.py` (+ opt-in GPU test) guards it.
The real issue is **geometric sensitivity**: the same model outputs give 10.4 or 0.07 m² for one
window depending on an irrelevant parameter.

### Uncertainty from geometry quality (`src/roomscan/quality.py`) [DECISION-PROV, uncalibrated]
Per room, the wall step is re-run under fixed irrelevant perturbations (yaw ±1°, two seeded point
halves); per face the RMS offset → `faces:stability` (excess over the measurement σ); a face
missing from a perturbed outline counts 0.5 m; a run with no outline at all is *topology fragility*
(flag), not a positional error; photo/video add `faces:registration` = camera-height spread;
rooms with a floor-area half-width > 50 % are flagged `geometry_unreliable`. Terms propagate
separately (named error-budget entries). [MEASURED] LiDAR rooms 1–4, 6, 7 unchanged (< 2 mm),
r8 `topology_fragile` (the half-density run lost it; first version of the rule gave it ±58 cm,
corrected); photo intervals ±10 cm → ±0.4–1.3 m; video v0 0.07 ± 0.65 m² (flagged). Coverage
on pseudo-GT barely moves (LiDAR floor_only 0 → 40 %; photo unchanged): consistently wrong rooms
are reported by flags, not hidden by width. No coverage was manufactured.

### Real-file input contracts [FACT where tested; BLOCKED where device-only]
`pillow-heif` added: HEIC with EXIF orientation 6 + 35 mm focal loads upright with the right focal.
MOV rotation as iPhones signal it (tkhd matrix; FFmpeg 7 no longer writes the old tag) → our reader
matches the player view at 0/90/180/270. Stray: depth and RGB sizes now read from the export
(supplied data unchanged; a 192×144-depth synthetic export reconstructs to 1 cm).
`roomscan-inspect` re-runs the e01 pose-convention and e05 video-offset checks on any Stray export
(supplied data: OpenCV axes 3.9 mm vs ARKit axes 81 mm) and reports lens/focal/orientation for
photos and codec/bit depth/rotation/focal tags for videos. [BLOCKED] Apple HEIC `irot` + EXIF
handling, 0.5× focal/distortion, MOV focal tags, 10-bit HDR decode, current Stray export format.

### Development capture + evaluation (`docs/development_capture_protocol.md`)
Literal ~30-minute checklist for questions A (LiDAR protocol ×2 vs natural), B (slow-turn vs
natural video), C (1× spread vs 0.5× overlapping photos, + doorway chain), with the reading of
each result **pre-registered**. Evaluator: site `purpose` with prefixed capture ids; benchmark
sites reject `dev_` captures; `bench/leakage.py` (tune / calibrate / benchmark_report);
partial tape GT; wall-match ambiguity; per-room scale ratio vs shape error; phantom rooms;
question/variant tables; debug diagnostics (video window continuity, photo registration spread,
LiDAR ladder steps); predicted-vs-reference figure. e27 `chain_eval.py` (placement of room B from
photos vs the LiDAR reference; exact on a self-check; `B_not_separated` on simulated frames).
Ultra-wide stays an A/B test, not a protocol requirement.

---

## Submission preparation (2026-10-06)

User direction: no new research; make the work submission-ready, reproducible, auditable and honest;
keep physically blocked items explicitly blocked.

- **Baseline frozen:** tag `engineering-baseline-v1` on `279abb6` after verification (38 tests incl.
  the opt-in GPU test; all three tiers and `roomscan-bench --execute` uncached from raw sample inputs,
  metrics identical to e24/e26; every plan validates against `schema/plan.schema.json`).
- **Hygiene:** no secrets, keys or weights tracked; machine-specific absolute paths removed from
  tracked experiment outputs; the evaluator and report builder now record repo-relative paths only;
  `jsonschema` declared (was transitive); `scripts/clean_clone_audit.sh` made executable.
- **Damage interfaces:** `src/roomscan/damage.py` (`roomscan-damage`): observations → regions with
  area intervals, concealed-damage rules CD1–CD4 (flag names the rule and surfaces), scope items keyed
  to surfaces; tested. There is no detector and no damage benchmark; status stays mechanism-only.
- **Reports from machine output only:** `reports/build_reports.py --regenerate` recomputes every
  result from raw inputs and renders `reports/benchmark_report.md` from `reports/data/*.json`.
  [GT-FREE] **Drift ON/OFF finding:** the production placement (`arkit_yaw`) changes the stitched
  footprint negligibly (room overlap 0.098 vs 0.093 m², per-room yaw ≤ 0.9°); room translations
  are ARKit's. Drift handling measurably acts on room dimensions (per-room visit registration), not
  on placement → the drift gate's "poses as-is" rule is at risk; compliance updated to say so.
- **Fix loop audited:** Part 4 on a real benchmark gate is BLOCKED. Shipped and regenerable cycle:
  segmentation border erosion on the exact synthetic capture (walls −499/−1004 mm → −1.9…−2.2 mm,
  areas −16.7/−33.6 % → −0.12/−0.16 %, real capture unchanged ≤ 0.36 mm), before/after from one code
  version via `roomscan-lidar --seg-pad`; declaration marked retrospective (`reports/fix_loop/`).
- **Documents:** `docs/capture_protocol.md` (one page + evidence status), `docs/device_matrix.md`,
  rewritten `docs/compliance_matrix.md`, evaluator-oriented `README.md`,
  `reports/technical_report.md` (~1.8 k words), `SUBMISSION_MANIFEST.md`, `reports/repro_audit.md`.
- **Correction carried into docs:** e25's "two uncached runs differed" is annotated with the e26
  finding (different code, not nondeterminism).

### 2026-10-06 — Publication: history rewrite
Before the first push, commit-message co-author trailers were removed and the author/committer
identity was set to `Pulkit Sharma <pulkitsharma14062002@gmail.com>` (GitHub: `pulkits1406`) on every commit (`git filter-branch`; dates, order, messages
otherwise and all file trees unchanged; final tree byte-identical). Commit hashes changed: the
old → new mapping is in `docs/commit_hash_mapping.md`, and hashes cited in tracked files were
updated to their rewritten equivalents (same trees). The baseline tag was recreated on the
rewritten baseline commit with its original date and message. A bundle of the pre-rewrite history
is kept outside the repository.

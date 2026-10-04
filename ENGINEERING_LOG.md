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

# HANDOVER — state of the project (end of phase 5: development capture prepared)

Written 2026-10-06 from the repository at commit `48418fe` (+ this handover commit) on `master`,
the experiment outputs and `ENGINEERING_LOG.md`. This file is authoritative for *state*; the
reasoning is in the log (phases 1–5) and the per-experiment READMEs. Previous versions: `2b53124`
(end of phase 4), `8328541` (end of phase 3).

**Where the project stands:** all three tiers run raw capture → scored plan; LiDAR v1.1 builds all
7 regions of the sample apartment; photo and video geometry is **not reliable** on the supplied
recordings (0 % of walls within gate, pseudo-GT), and those recordings are not evidence about the
proposed future protocol. **The next decision comes from the short development capture**
(`docs/development_capture_protocol.md`) — development data, not the benchmark.

**Status vocabulary.** IMPLEMENTED = production code under `src/` that runs end to end ·
PARTIAL = production code covering part of the requirement · EXPERIMENT ONLY = code under
`experiments/` · NOT STARTED. **Evidence labels.** FACT · MEASURED (GT-free consistency) ·
PSEUDO-GT (vs our own LiDAR/ARKit) · SYNTHETIC (exact rendered geometry) · HYPOTHESIS ·
PROVISIONAL (decision) · BLOCKED (needs a device or a user decision).

> **The pseudo-GT caveat applies to every accuracy number on the supplied data.** No laser/tape
> GT exists. ARKit LiDAR depth and the ARKit trajectory disagree by ~3 % (e08). Nothing below 3 %
> absolute scale can be established, and **no number in this repository is a benchmark result.**
> The only exact numbers are SYNTHETIC (a rendered capture), which test the code, not the sensor.

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
├── HANDOVER.md / ENGINEERING_LOG.md / README.md
├── pyproject.toml / uv.lock     uv project "zingly-scan"; groups dev+ml default; CLIs below
├── .gitignore                   raw data, zips, PDF, cache/, runs/, .envs/, .weights/, *.ply, *.npy, *.npz
├── src/roomscan/
│   ├── stray.py                 Stray loader (verified conventions)
│   ├── geometry.py              2-D grid helpers shared by v0 and v1 (segment_rooms gained padding
│   │                            parameters; v0 defaults unchanged, v0 values verified identical)
│   ├── lidar_pipeline.py        LiDAR v0 — FROZEN fix-loop baseline
│   ├── lidar/                   LiDAR v1.1: fusion, rooms (+room_cloud_rows), register (new),
│   │                            roomgeo (new: escalation ladder), walls (Rules), levels, openings,
│   │                            layout, build, cli
│   ├── photo/                   photo frontend: images (EXIF), inference (MoGe in process,
│   │                            MapAnything via ma_worker subprocess), reconstruct (ensemble scale),
│   │                            build, cli
│   ├── video/                   video frontend: frames (decode/orient), build (windows), cli
│   ├── cloudroom.py             SHARED: gravity-aligned metric cloud -> room (LiDAR seg/walls/levels)
│   ├── assemble.py              SHARED: RoomInput -> Plan with Jacobian-propagated intervals
│   ├── render.py                SHARED: plan PNG (all tiers; inferred dashed; unplaced noted)
│   ├── envs.py                  isolated-interpreter resolution for subprocess steps
│   ├── synthetic.py             exact synthetic two-room Stray capture (smoke tests; depth size configurable)
│   ├── quality.py               face stability under fixed irrelevant perturbations -> sigma terms (phase 5)
│   ├── provenance.py            code version / dirty flag / libraries recorded in every plan (phase 5)
│   ├── inspect.py               roomscan-inspect: format report + convention checks for real files (phase 5)
│   ├── model.py                 output schema 0.1 (+ additive Room.placed)
│   └── bench/                   site format (purpose, partial tape GT), matching (method-tagged, ambiguity),
│                                gates, evaluator (--execute), compare (material repeatability), leakage
│                                (tuning guard), diagnostics (debug.json), figures (walls vs reference)
├── schema/plan.schema.json      generated from model.py
├── tests/                       37 tests (+1 opt-in GPU) incl. smoke, repeatability, quality, inputs, dev site
├── envs/                        mapanything/, video/ pinned requirements + python-version; weights.yaml
├── scripts/                     setup_envs.sh, fetch_weights.py, prepare_intermediates.sh
├── benchmark/                   README (purposes + leakage rules), _template (bm), _template_dev (first dev
│                                session), pseudo_gt_stray_apartment (v0, old), pseudo_gt_lidar_v11 (e24)
├── docs/                        compliance_matrix, capture_protocol_hypotheses, physical_benchmark_procedure,
│                                reproduction.md, development_capture_protocol.md (phase 5)
└── experiments/e00…e27, DEPENDENCIES.md (input/producer map for every experiment)
```

**Raw data (FACT):** three Stray exports of one apartment (`single_room`, `single_scan_floor_only`,
`single_scan_with_ceiling`), zips at repo root (gitignored); unzip mapping in §9.

**Generated / cached artifacts (gitignored, regenerable):**

| Path | Size | Produced by | Used by |
|---|---|---|---|
| `.envs/mapanything`, `.envs/video` | 2.3 GB | `scripts/setup_envs.sh` | production photo/video (mapanything); e20 scripts (video) |
| `cache/e13/hf` | 4.6 GB | MapAnything weights (HF) | production photo/video; e13/e16/e17/e20 |
| `~/.cache/huggingface` | — | MoGe-2 (+ e12-only models) | production photo/video |
| `cache/frames/<capture>/` | 223 MB | `e07_frame_cache/extract.py` | e12, e13, e16, e17, e20, e23, e25 |
| `cache/photo_sim/single_scan_with_ceiling_K6`, `_K8_sweep` | small | `e23/make_photo_folders.py` | photo captures in `benchmark/pseudo_gt_lidar_v11` |
| `cache/e22/` | ~50 MB | `e22/dump_rooms.py` | e22 diagnose |
| `cache/e20/`, `cache/e08/` | 2.7 / 5.7 GB | e20 / e08 | e25 (tracked-piece study) |
| `runs/e22/`, `runs/e23/`, `runs/e24/`, `runs/e25/` | small + model caches | CLIs | e22–e25 (reference plan: `runs/e24/lidar_v11/plan.json`) |
| e02 PLY, e10 labels, e12 cache | — | see `experiments/DEPENDENCIES.md` | older experiments |

**Entry points (VERIFIED at `48418fe`):** `roomscan-lidar`, `roomscan-photo`, `roomscan-video`,
`roomscan-bench` (`--execute`), `roomscan-inspect`, `python -m roomscan.lidar_pipeline` (v0).

---

## 3. CURRENT IMPLEMENTATION

### Shared back end — IMPLEMENTED
```
LiDAR (Stray) ─ fusion → rooms/register → roomgeo ─┐
Photo folders ─ MoGe + MapAnything → reconstruct ───┼→ cloudroom* → assemble → Plan JSON → render
Video clip   ─ frames → windows → (photo machinery) ┘                      └→ roomscan-bench
(* LiDAR uses lidar.walls/levels directly; photo/video go through cloudroom, which calls the same code)
```
- `assemble.py`: tiers supply room polygon + per-face σ and status (measured/inferred), relative
  scale σ (one factor per room, reported `shared:scale`), ceiling ± σ, openings. Lengths, areas,
  perimeter, wall areas get intervals from a finite-difference Jacobian w.r.t. face offsets plus
  scale. Inferred faces make dependent walls and the area `status: inferred`. All intervals
  `calibrated: false`.
- `model.py` additive change: `Room.placed` (False = room frame only, position meaningless).

### Phase-5 additions (all tiers)
- **Uncertainty from geometry quality** (`quality.py`): per face, named error-budget terms
  `faces:measurement` / `faces:inferred` / `faces:stability` (RMS offset under yaw ±1° and two
  seeded point halves; missing face = 0.5 m) / `faces:registration` (photo/video camera-height
  spread). A perturbed run without any outline → `topology_fragile` flag (not a positional error).
  Floor-area half-width > 50 % → `geometry_unreliable`. Deterministic; uncalibrated.
- **Provenance:** `Plan.provenance` (git commit, dirty flag, library versions, parameters).
- **Real files:** HEIC via pillow-heif; MOV rotation from the tkhd matrix; Stray depth/RGB sizes
  read from the export; `roomscan-inspect` re-runs the e01/e05 convention checks.
- **Determinism (e26):** model outputs bitwise identical across processes; LiDAR repeat ≤ 2.3e-11 m.

### LiDAR — IMPLEMENTED (v1.1), PARTIAL against the contract
- `uv run roomscan-lidar <stray_dir> <out> [--placement arkit|arkit_yaw] [--no-ladder]`.
- v1.1 vs v1.0: (1) per-room **escalation ladder** (`lidar/roomgeo.py`) only for rooms whose
  topology fails: v1.0 → visits re-registered by `lidar/register.py` (per-axis face-profile
  cross-correlation hypotheses verified by vertical-surface overlap, then ICP) → missing sides
  inferred at the observed-floor boundary; each step is a quality flag; (2) face σ by evidence:
  measured 7 mm, topology-only 3 cm, inferred 10 cm (status inferred); (3) segmentation grid
  padded 1 m (`rooms.SEG_PAD_M`): the 0.6 m closing eroded rooms at the scan border (found by the
  synthetic test); (4) plan assembly through `assemble.py`.
- **Output (MEASURED):** `with_ceiling` → 7 of 7 non-empty regions → rooms (v1.0: 5), 4 doorways
  (v1.0: 2); r6 has 1 inferred wall; r2 needed re-registration; 6 quality flags. 36–40 s on M4.
  `--no-ladder` reproduces v1.0 geometry exactly. Synthetic two-room capture: walls −2 mm,
  ceiling −1.2 mm, doorway +1 cm (1 cm-bin quantisation).
- **Limitations:** needs every wall's top edge in view (captures without it → 1 merged room,
  flagged); inter-visit misregistration is common (30–160 mm raw, e22) and only repaired when a
  room fails; single dominant ceiling (room 3 bimodal); no windows; intervals uncalibrated.

### Photo — IMPLEMENTED (v0 frontend), accuracy not established
- `uv run roomscan-photo <folder_of_room_folders> <out> [--mode moge|ma]`. Inputs: images only;
  EXIF orientation + integer 35 mm-equivalent focal (`photo/images.py`; HEIC needs pillow-heif).
- Pipeline: one MoGe-2 load for all rooms → MapAnything for all rooms in one subprocess
  (`photo/ma_worker.py`, isolated env, bf16 backbone) → `reconstruct.room_cloud`: scale =
  √(C_MA·r) applied to MapAnything units (= e21 ensemble, C_MA 1.455), MoGe geometry placed with
  MapAnything poses, gravity from camera up-axes + normals → `cloudroom.room_from_cloud` (robust
  floor level, padded segmentation, observed then inferred walls) → `assemble`, rooms unplaced and
  laid out side by side.
- σ (provisional): scale 8.7 % shared (e21 ±14.3 % at 90 %), faces 3 cm / inferred 15 cm,
  levels 3 cm + scale.
- **Works end to end** (e24: 127 s for 6 folders). **Accuracy on simulated folders: 0 % of walls
  within ±8 %** (PSEUDO-GT); joint poses need overlapping photos (e23).

### Video — IMPLEMENTED (v0 frontend), accuracy not established
- `uv run roomscan-video <clip> <out> [--rotate auto|0|90|180|270] [--fps 2]`. Reads frames and
  timestamps only; orientation from container metadata (Stray `rgb.mp4` detected → 90°).
- Pipeline: keyframes at 2 fps → FOV from MoGe on 16 frames (median) → 12 s windows × 8 frames →
  photo machinery per window → **GT-free coherence gate** (camera heights above the reconstructed
  floor: std ≤ 15 cm, min ≥ 0.5 m) → room per coherent window; consecutive same-size windows
  merged; rooms unplaced and unlabelled, time spans in `quality_flags` (`vK_time_span`),
  `video_fragmented` and `intrinsics_estimated` flags. σ: photo σ + 3 % intrinsics.
- **Output (MEASURED/PSEUDO-GT):** `with_ceiling` clip: 3/18 windows coherent → 3 rooms; by true
  identity −60…−100 % area. 479–543 s uncached, 38 s cached. Uncached runs are not bit-stable.

### Evaluation — IMPLEMENTED
- `uv run roomscan-bench benchmark/<site> --run runs/<id> [--execute [--force]]`: with
  `--execute`, every capture in `site.yaml` runs through `python -m roomscan.<tier>.cli` in its own
  process (raw → pipeline → plan → score). Capture paths resolve relative to the site dir, else
  the repo root.
- Room matching: explicit `room_map` → label → (unlabelled rooms only) Hungarian on perimeter.
  Every match carries `method`; size-only matches are reported
  `rooms_matched_by_size_only (identity unverified)`.
- Unplaced rooms are excluded from overlap and fail the photo stitch gate with a reason.
- `gt_kind ∈ laser | tape | pseudo_lidar | synthetic`; residual rows tagged.
- Sites: `benchmark/pseudo_gt_lidar_v11` (current; LiDAR v1.1 plan of `with_ceiling`, r6
  excluded, scored captures = 2 independent LiDAR captures, 2 photo sets, the video) and the old
  v0-based `pseudo_gt_stray_apartment`.

### Reproducibility — IMPLEMENTED (not yet timed on a clean machine)
- `scripts/setup_envs.sh mapanything|video|all`: rebuilds the isolated envs from `envs/*` pins
  (verified fresh: 39 s / 1.5 GB, 33 s / 845 MB with warm uv cache; ~1 GB more cold).
- `scripts/fetch_weights.py [--check]`: 8 models pinned (`envs/weights.yaml`), offline check.
- `scripts/prepare_intermediates.sh [--gpu] [--dry-run]`: raw → ignored intermediates in order.
- `experiments/DEPENDENCIES.md`: inputs, producers, env and GPU need of every experiment.
- `docs/reproduction.md`: fresh clone → run, env-resolution order, offline use, conflicts.
- pycolmap + torch in one process aborts (OMP Error #15, verified) → subprocesses only; no env hack.

---

## 4. EXPERIMENT HISTORY

e00–e21: unchanged — see the phase-3 table in git (`git show 8328541:HANDOVER.md`, §4) and the
log. Summary of what still matters: Stray conventions final (e01, e05); per-room drift ~rigid
(e09); segmentation needs upper walls (e10); MoGe/MapAnything scale facts (e06, e12, e13, e16);
two-family ensemble 5.3 % / 14 % (e21); single doorway photos 0/36 (e17); no continuous accurate
RGB tracker (e20); V6 wall rules (e18); plane-anchored stitch negative (e19).

| ID | Purpose / approach | Key result | Conclusion | Reuse / do not repeat |
|---|---|---|---|---|
| e22 | Why LiDAR drops regions 2 and 6 | r2: two visits 22 cm apart along one axis with disjoint height bands → doubled half-height faces fail the extent test; r6: a wall never seen above 1.31 m; raw visit disagreement 30–160 mm in all rooms; topology test knife-edge (1.99 vs 2.00 m). Re-registration everywhere: 65 → 50 % walls within gate, loses room 3; inference everywhere grows correct rooms | **Escalation ladder only on failure** (v1.1): 7/7 rooms, working rooms unchanged | Do not apply re-registration or inference to every room |
| e23 | Photo frontend on simulated folders | MapAnything relative rotation 44–66° wrong with view-diverse photos, 2–7° with overlapping sweeps; single photos cannot measure rooms; r3 area match was a coincidence (walls 5.73 × 1.58 vs 3 × 3 m) | Overlap is required; geometry inside a partial sweep still wrong | Do not use view-diverse low-overlap sets as the photo contract |
| e24 | One evaluator path, all tiers, `--execute` | 5/5 captures run raw → scored; photo/video walls 0 % within gate; coverage 0–67 %; size-only matching had paired video rooms wrongly | Integration done; accuracy not; evaluator identity fixes | Score walls, not areas; check identity |
| e25 | Video: rooms in tracked pieces; windows | PnP pieces smear walls (r3 81 vs 8.9 m²); SIFT pieces scale −12…−30 %, fragmented; windows: 3/18 coherent, rooms −60…−100 % by identity | Video v0 = windows with coherence gate; capture behaviour is the limit | Do not fuse MoGe depth over e20 PnP/SIFT pieces |
| e26 | Determinism + geometric sensitivity | phase-4 "nondeterminism" was a code change mid-run; MoGe/MapAnything bitwise identical across processes (19/19); LiDAR ≤ 2.3e-11 m; same model outputs give 10.4 vs 0.07 m² under an irrelevant parameter | provenance in every plan; stability term in the error budget | Never compare runs without matching provenance |
| e27 | Doorway-chain placement evaluation (prepared) | self-check exact; simulated frames `B_not_separated` | ready for the dev capture's C3 set | Not production stitching |
| synthetic | Exact two-room Stray capture | found the segmentation border-erosion bug; after the fix walls −2 mm, ceiling −1.2 mm, doorway +1 cm | Smoke test + regression for LiDAR geometry | Keep in CI (`tests/test_smoke_pipeline.py`) |

---

## 5. CURRENT MEASURED RESULTS

| Quantity | Value | Label | Source |
|---|---|---|---|
| Stray conventions | T_wc OpenCV axes, +y up, video row j+1 | FACT | e01, e05 |
| LiDAR/trajectory scale self-consistency | ~3 % | MEASURED | e08 |
| LiDAR v1.1 rooms on `with_ceiling` | 7/7 non-empty regions (v1.0 5/7); doorways 4 (v1.0 2) | MEASURED | e22, e24 |
| LiDAR wall repeatability, production code, half data | 65 % within max(1 cm, 0.5 %), 10.1 / 41.9 mm median/p90 (v1.0 rooms); ladder incl. recovered rooms 60 % | MEASURED (GT-free) | e22 |
| Same, e18 experiment code (V6) | 73 %, 9.8 / 33.5 mm | MEASURED (GT-free) | e18 — production numbers are the ones to quote |
| LiDAR ceilings repeat ≤ 1 cm, production | 50–60 % of rooms (room 3 bimodal, 104 mm flip) | MEASURED (GT-free) | e22 |
| Raw inter-visit face disagreement within a room | 30–160 mm | MEASURED | e22 |
| LiDAR on exact synthetic rooms | walls −2.0…−2.2 mm, ceiling −1.2 mm, area −0.1 %, doorway +1 cm | SYNTHETIC | `synthetic.py` |
| Independent LiDAR captures without upper-wall sweep vs v1.1 reference | 1 merged room each; 18 % wall / 23–35 % area error; coverage 0 % | PSEUDO-GT | e24 |
| Photo scale, ensemble | 5.3 % median, 14.0 % p90 | PSEUDO-GT (LORO) | e21 |
| Photo joint-pose rotation error, view-diverse vs overlapping | 44–66° vs 2–7° (median, failing/working rooms) | PSEUDO-GT | e23 |
| Photo walls within ±8 % (simulated folders) | 0 % (both selections) | PSEUDO-GT | e24 |
| Video FOV self-estimate | 777.5 px vs ≈ 795 px (−2.5 %) | PSEUDO-GT | e25 |
| Video coherent windows / rooms | 3/18 → 3 rooms; −60…−100 % area by identity; v0 now 0.07 ± 0.65 m², flagged unstable/unreliable | PSEUDO-GT | e24, e25, e26 |
| Interval coverage, all tiers | 0–67 % vs 90 % nominal (LiDAR floor_only 0 → 40 % with the stability term; photo/video unchanged: errors exceed honest widening) | PSEUDO-GT | e24, e26 |
| Run-to-run repeatability | models bitwise identical; LiDAR ≤ 2.3e-11 m | MEASURED | e26 |
| Photo interval half-widths with quality terms | ±0.4–1.3 m (was ±0.1 m) | MEASURED | e26 |
| Runtimes on M4 | LiDAR 6–40 s; photo 127 s / 6 folders; video 479–543 s / 215 s clip | MEASURED | e24 |
| Isolated env rebuild | 39 s + 33 s (warm cache), 2.3 GB | MEASURED | reproduction agent |

---

## 6. CURRENT DECISIONS

| Decision | Status | Evidence | Rejected alternatives | Would change if |
|---|---|---|---|---|
| Stray conventions; uv; MoGe pinned | FINAL | e01, e05 | — | Stray v1.4 differs |
| Common `Plan` + shared assembly (Jacobian intervals, inferred status, `shared:scale`, `Room.placed`) | FINAL (additive changes only) | used by all three tiers | per-tier schemas | a tier output cannot be represented |
| Pseudo-GT never presented as benchmark; scored captures never include the reference capture | FINAL | ~3 % self-inconsistency | — | laser GT |
| Evaluator: identity by map/label; size-only matches flagged | FINAL | e24 wrong pairings | Hungarian for all rooms | — |
| LiDAR v1.1 = v1.0 + escalation ladder only on topology failure; segmentation padded 1 m | PROVISIONAL | e22, synthetic | re-registration / inference for all rooms (trade-off recorded) | a real repeat capture shows registration helps everywhere |
| LiDAR placement `arkit_yaw`, σ 33 mm | PROVISIONAL | e19 | plane-anchored stitch | two protocol captures |
| Photo frontend = MoGe geometry on MapAnything poses + ensemble scale; rooms unplaced | PROVISIONAL | e21, e23 | MapAnything depth geometry (`--mode ma`), single-photo rooms | device photos with overlap behave differently |
| Photo protocol hypothesis: overlapping photos (0.5× ultra-wide, landscape, turning on the spot), floor + ceiling edge visible | HYPOTHESIS | e23 | view-diverse photos | dev capture |
| Video frontend = 12 s overlapping windows with a GT-free coherence gate; fragmentation explicit | PROVISIONAL | e20, e25 | rooms in tracked pieces (e25), depth-odometry bridging (e20) | a protocol-following clip tracks continuously |
| Video FOV from the frames (no metadata, no IMU) | PROVISIONAL (user: plain clip) | e25 −2.5 % | Sensor Logger/IMU | user reinterprets |
| Isolated envs for MapAnything (+ LightGlue) via subprocess; pycolmap never with torch | FINAL (operational) | OMP #15 verified | KMP_DUPLICATE_LIB_OK hack | upstream fix |
| Interval widening from measured geometric sensitivity (quality.py), never a blanket factor; uncalibrated | PROVISIONAL | e26 | multiplying intervals; refusing all photo/video rooms | laser calibration |
| Site purposes + leakage guard: dev data answers questions, never benchmark evidence or silent calibration | FINAL | user instruction | — | — |
| Ultra-wide (0.5×) photos: A/B test only, not a protocol requirement | PROVISIONAL (user) | e23 hypothesis | mandating 0.5× now | dev capture + device independence |
| One GPU model resident at a time (one MoGe load per run, MapAnything batched) | FINAL (operational) | memory dips to 4–25k free pages during MapAnything | per-window model loads | more RAM |
| v0 frozen | FINAL | values verified identical after the geometry.py change | — | — |

---

## 7. CURRENT ASSESSMENT STATUS (reconciled; `docs/compliance_matrix.md` updated in this phase)

| Requirement | Status | Evidence path |
|---|---|---|
| LiDAR tier | PARTIAL (7/7 regions; needs upper-wall coverage; intervals uncalibrated) | `src/roomscan/lidar/`, e22, e24 |
| Photo tier | PARTIAL (runs end to end; accuracy far from gate on simulated input) | `src/roomscan/photo/`, e23, e24 |
| Video tier | PARTIAL (runs end to end; fragmented; accuracy far from gate on the sample clip) | `src/roomscan/video/`, e25, e24 |
| Common output contract | PASS for representation (all tiers emit schema-valid plans) | `model.py`, `assemble.py` |
| One command per capture (O7) | PASS (machinery) | three CLIs, `--execute`, smoke test |
| Rendered plan (O9) | PARTIAL (all tiers; no windows/damage) | `render.py` |
| Stitched plan / adjacency (O2) | PARTIAL (LiDAR only; photo/video unplaced) | `lidar/layout.py` |
| Openings / G1 | EXPLORING (LiDAR doorways; +1 cm bin bias on synthetic; no windows) | `lidar/openings.py` |
| Ceiling / G2 | EXPLORING (production 50–60 % ≤ 1 cm GT-free) | e22 |
| Repeatability / G3 | AT RISK (65 % production, half data) | e22 |
| Drift ablation / G4 | PARTIAL | e19 |
| Photo stitch / G5 | AT RISK (no method; overlap-chain untested) | e17, e23 |
| Photo walls / G6, video walls / G7 | AT RISK (0 % within gate, pseudo-GT) | e24 |
| Calibration / G8, C6 | EXPLORING (quality-driven widening; coverage 0–67 %; calibration guard in place) | e24, e26 |
| Development capture (B0) | READY, blocked on a borrowed iPhone | `docs/development_capture_protocol.md` |
| Damage O3–O5 | NOT STARTED | — |
| Benchmark set B1–B5, head-to-head H1 | BLOCKED (physical) | `docs/physical_benchmark_procedure.md` |
| Fix loop F1/F2 | NOT STARTED (needs a real failing gate; v0 frozen; candidates: doorway bin bias, ceiling bimodality, registration) | — |
| D1 README < 15 min | PARTIAL (recipes verified; not timed end to end) | `docs/reproduction.md` |
| D2 reproduction bundle | PARTIAL (`--execute` done; runs materially identical, e26) | `bench/evaluate.py` |
| K1 offline / weights by script | PARTIAL (manifest + check; offline end-to-end not run) | `envs/weights.yaml` |
| Device matrix C2, reports D3/D4, walk-in W1 | NOT STARTED | — |

---

## 8. CURRENT FILES THAT MUST NOT BE LOST

Everything listed in the phase-3 handover (§8 of `git show 8328541:HANDOVER.md`) still applies.
Added this phase:

| Path | Why |
|---|---|
| `src/roomscan/{assemble,cloudroom,envs,render,synthetic}.py`, `photo/`, `video/`, `lidar/{register,roomgeo}.py` | the three-tier product path |
| `tests/test_smoke_pipeline.py`, `tests/test_assemble.py` | end-to-end and propagation regression |
| `envs/`, `scripts/`, `docs/reproduction.md`, `experiments/DEPENDENCIES.md` | environment + data reconstruction |
| `benchmark/pseudo_gt_lidar_v11/site.yaml` | current pseudo-GT site (regenerate with e24 `make_site.py`) |
| `experiments/e22…e25/` (scripts, READMEs, `out/`) | evidence for every phase-4 decision |
| `.envs/` (ignored, 2.3 GB) | rebuildable with `scripts/setup_envs.sh` (~1–2 min warm) |

---

## 9. ENVIRONMENT / REPRODUCTION

- Machine: Apple M4, 16 GB, macOS 26.7.1, MPS. uv 0.12.23. Project env Python 3.13.14 (torch
  2.14.1, open3d 0.20, opencv 5.0, moge `b942f00bdc2a`, pycolmap 4.2.1).
- Isolated envs (Python 3.12.13): `.envs/mapanything` (mapanything `3d10cf7`, opencv-headless
  4.10), `.envs/video` (lightglue `eb42fee2`, kornia 0.8.3). Resolution: `$ROOMSCAN_PY_<NAME>` →
  `.envs/<name>/bin/python` → legacy `cache/e13|e20/.venv`. MapAnything weights:
  `$ROOMSCAN_HF_HOME_MAPANYTHING` (default `cache/e13/hf`).
- Memory: MapAnything load spike ≈ 6.4 GB; free pages dropped to 4–25k during runs without
  sustained pageouts. Guard: pause on free pages < 50k **and** rising pageouts; never two GPU
  models at once.

```
uv sync && scripts/setup_envs.sh all && uv run python scripts/fetch_weights.py --check
uv run pytest -q                                                    # 20 passed (~8 s)
# raw data: single_room.zip -> c00a170fe1/ -> single_room/ (etc., see scripts/prepare_intermediates.sh)
uv run roomscan-lidar single_scan_with_ceiling runs/e24/lidar_v11                 # 40 s
uv run python experiments/e23_photo_frontend/make_photo_folders.py single_scan_with_ceiling 6
uv run python experiments/e23_photo_frontend/make_photo_folders.py single_scan_with_ceiling 8 "" sweep
uv run python experiments/e24_three_tier_eval/make_site.py runs/e24/lidar_v11/plan.json
uv run roomscan-bench benchmark/pseudo_gt_lidar_v11 --run runs/e24/three_tier --execute   # ~13 min
uv run python experiments/e24_three_tier_eval/video_identity.py
uv run python experiments/e22_lidar_small_rooms/repeat.py single_scan_with_ceiling v1_0,ladder
uv run python -m roomscan.lidar_pipeline single_scan_with_ceiling runs/v0/with_ceiling_lidar  # v0
```
Older experiments: regenerate ignored inputs with `scripts/prepare_intermediates.sh` (order and
producers in `experiments/DEPENDENCIES.md`).

---

## 10. GIT STATE

- Phase-5 commits: `46871d4` e26 determinism + provenance + quality terms → `49a1a4a` real-file
  inputs + roomscan-inspect → `4fbed2b` dev-site evaluator + leakage guard → `48418fe` dev capture
  protocol + e27 + docs → handover commit.
- Branch `master`. Phase-4 commits: `244f7e5` reproducibility recipes → `5b25c37` LiDAR v1.1 +
  e22 + assemble → `8570bb4` photo/video frontends, `--execute`, synthetic smoke test → `8f98342`
  e24 evaluation + evaluator identity fixes → handover commit. 22 commits before the handover.
- Ignored on purpose: raw data, zips, PDF, `cache/`, `runs/`, `.envs/`, `.weights/`, `*.ply`,
  `*.npy`, `*.npz`.
- No background processes running. Working tree clean after the handover commit.
- Git rule: the user authorised commits for this phase ("make a logical git commit"); outside
  such an explicit request, print git commands instead of running them.
- Leftover temp files from the reproducibility agent the user may remove:
  `rm -rf /var/folders/mz/krpt08hj0679xkt9f2rnss940000gn/T/roomscan-keep.*` (a tracked-file backup
  from its restore test) and the session scratchpad.

---

## 11. OPEN QUESTIONS / BLOCKERS

### Local (no device needed)
| Question | Why | Evidence | Next action |
|---|---|---|---|
| Interval honesty when topology is consistently wrong | stability cannot see it (e26) | flags only | consider refusing rooms on `registration_inconsistent` / `geometry_unreliable` after dev data |
| Photo/video geometric sensitivity | same model outputs → 10.4 vs 0.07 m² under an irrelevant parameter (e26) | now measured and in the intervals | real dev data first; then decide whether to stabilise the room step |
| Doorway width +1 cm bias | synthetic shows 1 cm-bin quantisation | synthetic | sub-bin jamb edges; candidate first fix-loop item (synthetic before/after, laser later) |
| Room 3 bimodal ceiling | ceiling repeat flips 104 mm | e22 | report dominant + secondary level, or the level under most of the floor |
| e18 vs production repeatability gap (73 vs 65 %) | quoting the right number | e22 | quote production; optionally align half construction |
| Clean-machine timed README run, HF offline run | D1/K1 | recipes verified | do on a second user account or fresh clone |

### Requires a device
| Question | Why | Next action |
|---|---|---|
| Walls when every wall's top edge is seen (one sweep) — do doubled faces/re-registration disappear? | LiDAR topology + repeatability | dev capture |
| Does a slow-turn video give continuous tracking and coherent windows? | video tier | dev capture |
| Do overlapping (ultra-wide, landscape) photo sets give correct rooms, and does the overlapping doorway chain stitch? | photo tier | dev capture |
| Real file formats (Stray v1.4, HEIC EXIF, .MOV rotation metadata, magicplan export) | loaders | dev capture |
| Absolute accuracy, calibration, fix loop | all gates | laser session |

### User decisions
Optional scale cue in the photo protocol; repeatability-tolerance reading; Route 1 only if Route 2
fails a gate (unchanged).

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

**First development capture — READY (phase 5).** The literal operator checklist, the
pre-registered reading of each result, export steps, file names and day-one commands are in
`docs/development_capture_protocol.md`; the site template is `benchmark/_template_dev/site.yaml`.
It answers exactly three questions: **A** LiDAR protocol (×2) vs natural capture; **B** slow-turn
vs natural video; **C** 1× spread vs 0.5× overlapping photos of the same room, plus the doorway
chain (e27). Tape: 2–3 walls of room A + the doorway. It is `dev_` data: never benchmark evidence,
never silent calibration input (`bench/leakage.py`). Ultra-wide stays an A/B test.

---

## 13. WHAT SHOULD HAPPEN NEXT (prioritised)

1. **Run the development capture** (user, borrowed LiDAR iPhone, ~30 min + ~10 min tape) exactly as
   in `docs/development_capture_protocol.md`; bring back the files listed there.
2. **Day one:** `roomscan-inspect` every file; resolve every `CHECK` (Stray format, HEIC
   orientation/focal, MOV rotation/bit depth) before reading results; fill
   `benchmark/dev_<date>/site.yaml`; `roomscan-bench ... --execute`; run e27 on C3 with A1 as the
   reference. Record each answer against its pre-registered reading.
3. **Decide from the answers** (not from the sample recordings): photo protocol (overlap/0.5×),
   video protocol (slow turn) and LiDAR protocol wording; only then consider new photo/video
   algorithms, and only for the failure the dev data shows.
4. Interval honesty for consistently wrong rooms (refuse or flag harder) — after dev data.
5. Clean-machine timed README run + offline (`HF_HUB_OFFLINE=1`) end-to-end run (D1/K1).
6. Laser benchmark session (`bm_`/`rep_`/`inc_`/`walk_`) → calibration from explicitly named
   sources → the fix loop (candidates: doorway 1 cm-bin bias, ceiling bimodality, registration).
7. Later: windows (RGB), damage + concealed rules + scope, reports, device matrix.

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
| Apply visit re-registration or side inference to every room | e22: repeatability 65 → 50 %, correct rooms grew; escalate only on failure |
| Judge a room by its area alone | e24: r3 area matched within 1.2 % with a wrong shape |
| Trust size-only (Hungarian) room matches as identity | e24: video rooms were paired with the wrong rooms |
| Fuse MoGe depth over e20 PnP or e08 SIFT pieces to measure rooms | e25: walls smeared / scale −12…−30 % |
| Use view-diverse, non-overlapping photo sets as the photo contract | e23: joint poses 44–66° wrong |
| Edit `geometry.py` defaults | v0 depends on them; add parameters instead (as `segment_rooms` padding) |
| Load MoGe per window / MapAnything per room | memory dips to 4k free pages; one load per run |
| Compare two runs without matching `provenance` | e26: a code change mid-run was misread as nondeterminism |
| Use dev_ results as benchmark evidence or as unnamed calibration input | leakage; enforced by `bench/leakage.py` |
| Make 0.5× ultra-wide mandatory before the dev A/B and a second device | user decision; evidence first |
| Tune photo/video models further on the supplied recordings | they are not evidence for the proposed protocol; the dev capture decides |

---

## 15. HANDOVER QUALITY CHECK

1. What are we building? → §1. 2. What is implemented? → §3 (three tier CLIs, shared back end,
evaluator with `--execute`, env recipes). 3. How do I run it? → §9. 4. Experiments? → §4 (+ phase-3
table via git). 5. What did they prove? → §4–5. 6. What failed? → §4, §5, §14. 7. Decisions? → §6.
8. Uncertain? → §11. 9. Hardware-blocked? → §11, §12. 10. Next? → §13. 11. Key files? → §2, §8.
12. Must not be overwritten? → §8, v0 + `geometry.py` defaults (§14). 13. Pseudo-GT vs benchmark?
→ header, §5 labels, §3 Evaluation, e24 README.

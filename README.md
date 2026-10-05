# roomscan — iPhone captures to a dimensioned floor plan with intervals

A pipeline from iPhone captures at three input tiers (**LiDAR**: Stray Scanner export; **video**:
a plain handheld clip; **photos**: 2–8 stills per room, one folder per room) to one output
contract: rooms, walls, ceiling height, floor area, openings, adjacency, a 90 % interval and a
status on every measurement, JSON to `schema/plan.schema.json`, and a rendered plan.

**Read this first — what is and is not demonstrated.** No iPhone was available, so there is no
laser benchmark, no real-device capture, no head-to-head and no walk-in test. Every accuracy number
in this repository is PSEUDO-GT (against our own LiDAR reconstruction of supplied sample data),
GT-FREE (internal consistency) or SYNTHETIC (rendered geometry). The LiDAR tier works on the sample
captures; **the photo and video tiers run end to end but are not accurate on the supplied
recordings** (0 % of walls within their gates, pseudo-GT). Details: `reports/benchmark_report.md`,
status per requirement: `docs/compliance_matrix.md`, navigation: `SUBMISSION_MANIFEST.md`.

## 1. Requirements

- macOS on Apple Silicon (developed and tested on an M4, 16 GB). LiDAR tier: CPU only. Photo/video
  tiers: MPS GPU (MoGe-2 and MapAnything); CPU fallback exists but is untested and slow; CUDA untested.
- [uv](https://docs.astral.sh/uv/) ≥ 0.12, git, ~12 GB free disk (envs ~3 GB, weights ~7 GB).
- Network access for the one-time install and weight download; afterwards everything runs offline.

## 2. Install (measured: ~2–3 min with a warm uv cache; add download time otherwise)

```
git clone <repo> && cd <repo>
uv sync                                   # project env (Python 3.13, torch, MoGe-2, open3d, ...)
scripts/setup_envs.sh mapanything         # isolated env for MapAnything (photo + video tiers)
uv run python scripts/fetch_weights.py    # pinned weights: MoGe-2 (1.3 GB), MapAnything (4.9 GB), ...
uv run python scripts/fetch_weights.py --check   # offline presence check
uv run pytest -q                          # 35 pass on a clean clone; 4 more need the sample captures; +1 opt-in GPU test (ROOMSCAN_GPU_TESTS=1)
```
MapAnything needs its own env (it pins OpenCV 4.10; the project uses 5.x) and runs as a subprocess;
pycolmap and torch also cannot share a process (both bundle libomp). `scripts/setup_envs.sh video`
builds the LightGlue env used only by experiment e20. Details: `docs/reproduction.md`.

## 3. Run one capture (one command per tier)

```
uv run roomscan-inspect <capture>                      # format checks for any input (recommended first)
uv run roomscan-lidar  <stray_export_dir>   out/lidar  # LiDAR: Stray Scanner export (depth + poses + intrinsics)
uv run roomscan-video  <clip.MOV>           out/video  # video: plain clip, rotation from metadata (--rotate 90 for Stray rgb.mp4)
uv run roomscan-photo  <folder_of_room_folders> out/photo   # photos: <folder>/<room name>/*.HEIC|*.jpg
uv run roomscan-damage out/lidar/plan.json <observations.json> out/lidar/plan_damage.json   # damage regions/rules/scope (see §6)
```
Each run writes `plan.json` (schema-valid; every measurement has `value`, `unit`, `interval`
(90 %, `calibrated: false`), `status` measured / inferred / not_observed, `method`, named
`error_budget` terms), `plan.png` (dimensioned render; dashed = inferred wall), and `debug.json`.
`quality_flags` in the plan say what limited the result (e.g. `upper_walls_not_observed`,
`rooms_unplaced`, `video_fragmented`, `rX_topology_unstable`, `rX_geometry_unreliable`) and how to re-capture.
How to capture: `docs/capture_protocol.md` (one-page Route-2 protocol, not yet field-tested).

## 4. Evaluate

```
uv run roomscan-bench benchmark/<site> --run runs/<id> --execute
```
`--execute` runs every capture listed in `benchmark/<site>/site.yaml` through its tier's pipeline
(raw input → plan), then scores it: walls, ceilings, areas, openings (misses and phantoms count),
adjacency, overlaps, repeatability pairs, head-to-head, interval coverage, per-room scale vs shape
error. Output: `runs/<id>/eval/<site>.{json,md}` and `<site>_walls.png`. Site format and the
purpose rules that keep development data out of the benchmark: `benchmark/README.md`.

## 5. Reproduce the reported results

Needs the supplied sample captures (`single_room.zip`, `single_scan_floor_only.zip`,
`single_scan_with_ceiling.zip`, ~870 MB, **not in git** — they were supplied with the task and must
be provided alongside the repository as a volume) at the repository root. Without them, only the
synthetic smoke test, the fix-loop bundle and the unit tests reproduce.
```
scripts/prepare_intermediates.sh                    # unzip samples, frame cache, other CPU intermediates
uv run python reports/build_reports.py --regenerate # ~25 min on M4: every result from raw inputs
reports/fix_loop/run.sh                             # fix-loop before/after only (CPU, ~2 min)
uv run pytest -q                                    # includes the synthetic smoke test (no sample data needed)
```
`--regenerate` rebuilds the LiDAR reference plan, the pseudo-GT site, simulated photo folders, all
five scored captures, the video identity check, the repeatability harness, the drift ON/OFF
comparison and the fix loop, then renders `reports/benchmark_report.md` from `reports/data/*.json`.
Experiment-level results (e00–e27) regenerate per their READMEs; inputs and producers of every
experiment are listed in `experiments/DEPENDENCIES.md`.

## 6. What does not work / is not done

- **Photo tier:** rooms are not stitched into one plan (`placed: false`); room geometry is wrong on
  the simulated photo sets; the overlapping-photo protocol is a hypothesis.
- **Video tier:** no continuous accurate tracking on the sample clip; rooms come from short coherent
  windows, unplaced, and are inaccurate on the sample clip.
- **LiDAR tier:** needs every wall's top edge in view; no window detection; single dominant ceiling level.
- **Damage:** schema, rule engine (concealed-damage rules CD1–CD4) and scope line items exist and are
  tested; there is **no damage detector** — observations must come from an external detector or a
  manual annotation (`src/roomscan/damage.py`). Example after `reports/fix_loop/run.sh`:
  `uv run roomscan-damage runs/fix_loop/after_fix/plan.json docs/examples/damage_observations_synthetic.json out.json`.
- **Intervals are uncalibrated** (calibration needs laser GT); coverage on pseudo-GT is 0–67 %.

## 7. Physically unverified (blocked: no iPhone)

Laser/tape benchmark and every gate on it (openings ≤ 2 cm, ceiling ≤ 1.5 cm, repeatability,
photo stitch, photo ±8 %, video ±3 %); calibration; head-to-head vs a consumer app; real
HEIC/MOV/current-Stray files (loaders tested with synthetic files only); the capture protocol;
the fix loop on a real benchmark gate; the walk-in test. Prepared next step:
`docs/development_capture_protocol.md`.

## Repository map

| Path | Contents |
|---|---|
| `src/roomscan/lidar/` | LiDAR v1.1: fusion → rooms/register → roomgeo (escalation ladder) → walls → levels → openings → layout → build |
| `src/roomscan/photo/`, `src/roomscan/video/` | photo and video frontends (MoGe-2 in process, MapAnything subprocess) |
| `src/roomscan/cloudroom.py`, `assemble.py`, `quality.py`, `render.py` | shared: metric cloud → room; room → Plan with propagated intervals; geometry-quality terms; render |
| `src/roomscan/damage.py` | damage regions, concealed-damage rules, scope items (no detector) |
| `src/roomscan/bench/` | site format, matching, gates, evaluator, leakage guard, figures |
| `src/roomscan/inspect.py`, `stray.py`, `synthetic.py`, `envs.py`, `provenance.py` | input checks, Stray loader, synthetic capture, isolated envs, run provenance |
| `src/roomscan/lidar_pipeline.py` | LiDAR v0, frozen (fix-loop history) |
| `schema/plan.schema.json` | published output schema (generated from `src/roomscan/model.py`) |
| `reports/` | benchmark report, fix-loop bundle, technical report |
| `docs/` | capture protocol, device matrix, compliance matrix, reproduction, development capture |
| `experiments/e00…e27/` | every experiment with its README, including negative results |
| `ENGINEERING_LOG.md`, `HANDOVER.md` | reasoning record; project state |

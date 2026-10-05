# Room scan pipeline (work in progress)

Pipeline from iPhone captures (photos, video, LiDAR) to a dimensioned, stitched floor plan with
confidence intervals. Status and reasoning: `ENGINEERING_LOG.md`; requirement coverage:
`docs/compliance_matrix.md`.

## Setup

```
uv sync                                   # Python 3.13 env with core + dev + ml groups (torch, MoGe-2, ...)
scripts/setup_envs.sh all                 # isolated envs: MapAnything (photo/video tiers), LightGlue (experiments)
uv run python scripts/fetch_weights.py    # pinned pretrained weights; --check verifies offline
uv run pytest -q                          # includes an end-to-end smoke test on a synthetic capture
```
Details, disk/time numbers and offline use: `docs/reproduction.md`.

Raw sample captures (Stray Scanner exports) are expected unzipped at the repo root:
`single_room/`, `single_scan_floor_only/`, `single_scan_with_ceiling/` (not in git).

## Run (one command per capture; every tier writes plan.json + plan.png + debug.json)

```
# LiDAR tier v1.1 (Stray Scanner export); --no-ladder reproduces v1.0
uv run roomscan-lidar single_scan_with_ceiling runs/v1/with_ceiling_lidar

# Photo tier: a folder with one sub-folder of stills per room (EXIF focal only); rooms unplaced
uv run roomscan-photo <photos_dir> runs/photo

# Video tier: a plain clip (no intrinsics/IMU used); rooms per coherent window, unplaced
uv run roomscan-video <clip.mov> runs/video          # Stray rgb.mp4: rotation inferred (--rotate 90)

# LiDAR prototype v0, kept unchanged as the regenerable baseline
uv run python -m roomscan.lidar_pipeline single_scan_with_ceiling runs/v0/with_ceiling_lidar

# Score a benchmark site; --execute first runs every capture through its tier's pipeline
uv run roomscan-bench benchmark/<site> --run runs/<run_id> --execute
```

Output schema: `schema/plan.schema.json` (generated from `src/roomscan/model.py`).
Benchmark format: `benchmark/README.md`.

## Layout

| Path | Contents |
|---|---|
| `src/roomscan/stray.py` | Stray Scanner loader (pose/frame conventions verified, see tests) |
| `src/roomscan/lidar/` | LiDAR tier v1.1: fusion → rooms/register → roomgeo (ladder) → walls → levels → openings → layout → build |
| `src/roomscan/photo/`, `src/roomscan/video/` | photo and video frontends (MoGe-2 + MapAnything subprocess) |
| `src/roomscan/cloudroom.py`, `assemble.py`, `render.py` | shared: metric cloud → room; room → Plan with propagated intervals; plan render |
| `src/roomscan/envs.py`, `envs/`, `scripts/` | isolated interpreters, env recipes, weight fetch, intermediates |
| `src/roomscan/synthetic.py` | exact synthetic Stray capture for smoke tests |
| `src/roomscan/geometry.py`, `lidar_pipeline.py` | shared 2-D segmentation helpers; v0 baseline pipeline |
| `src/roomscan/model.py` | output representation (every measurement carries an interval) |
| `src/roomscan/bench/` | benchmark format, matching, gates, report |
| `experiments/eNN_*` | one directory per experiment; scripts regenerate `out/` |
| `docs/` | compliance matrix, capture-protocol hypotheses, physical benchmark procedure |

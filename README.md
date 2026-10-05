# Room scan pipeline (work in progress)

Pipeline from iPhone captures (photos, video, LiDAR) to a dimensioned, stitched floor plan with
confidence intervals. Status and reasoning: `ENGINEERING_LOG.md`; requirement coverage:
`docs/compliance_matrix.md`.

## Setup

```
uv sync            # Python 3.13 env with core + dev + ml groups (torch, MoGe-2, transformers)
uv run pytest -q
```

Raw sample captures (Stray Scanner exports) are expected unzipped at the repo root:
`single_room/`, `single_scan_floor_only/`, `single_scan_with_ceiling/` (not in git).

## Run

```
# LiDAR tier v1 (Stray Scanner export -> plan.json + plan.png + debug.json)
uv run roomscan-lidar single_scan_with_ceiling runs/v1/with_ceiling_lidar

# LiDAR prototype v0, kept unchanged as the regenerable baseline
uv run python -m roomscan.lidar_pipeline single_scan_with_ceiling runs/v0/with_ceiling_lidar

# Score plans against a benchmark site (one command per site)
uv run roomscan-bench benchmark/<site> --run runs/<run_id>
```

Output schema: `schema/plan.schema.json` (generated from `src/roomscan/model.py`).
Benchmark format: `benchmark/README.md`.

## Layout

| Path | Contents |
|---|---|
| `src/roomscan/stray.py` | Stray Scanner loader (pose/frame conventions verified, see tests) |
| `src/roomscan/lidar/` | LiDAR tier v1: fusion → rooms → walls → levels → openings → layout → build → render |
| `src/roomscan/geometry.py`, `lidar_pipeline.py` | shared 2-D segmentation helpers; v0 baseline pipeline |
| `src/roomscan/model.py` | output representation (every measurement carries an interval) |
| `src/roomscan/bench/` | benchmark format, matching, gates, report |
| `experiments/eNN_*` | one directory per experiment; scripts regenerate `out/` |
| `docs/` | compliance matrix, capture-protocol hypotheses, physical benchmark procedure |

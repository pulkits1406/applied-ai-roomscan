# e18 — LiDAR repeatability: room-local aggregation + robust planes

GT-free, within-capture consistency on `single_scan_with_ceiling`. **Not the repeatability gate**
(that needs two separate captures made per the protocol); halves here hold ~50 % of the data.

## Commands

```
uv run python experiments/e18_lidar_repeat/run_split.py single_scan_with_ceiling V0,V1,V2,V3 visit   # topology, visit halves
uv run python experiments/e18_lidar_repeat/run_split.py single_scan_with_ceiling V0,V1,V2,V3 chunk   # topology, chunk halves
uv run python experiments/e18_lidar_repeat/run_fixed_topology.py single_scan_with_ceiling V0,V1,V2,V3 2.0
uv run python experiments/e18_lidar_repeat/run_fixed_topology.py single_scan_with_ceiling V2,V4 2.0
uv run python experiments/e18_lidar_repeat/run_fixed_topology.py single_scan_with_ceiling V5 2.0
uv run python experiments/e18_lidar_repeat/run_fixed_topology.py single_scan_with_ceiling V6 2.0
```

## Variants (`roomgeo.py`)
V0 e14 baseline · V1 multi-visit, per-room ICP · V2 V1 + camera-oriented normals + robust offsets
in a clutter-free band, edge extent only · V3 V2 + line merge/notch absorption · V4 V2 + outermost
density peak · V5 V2 with close-range (≤ 2.5 m) near-normal (incidence ≤ 60°) observations only ·
V6 V2 topology/ceiling + V5 wall positions (per-wall fallback to V2)

## Topology stability (independent halves)
Visit halves: 28/28 failed (no upper-wall observations per half). Chunk halves: most rooms still
fail to find all walls → wall existence needs full upper-wall coverage.

## Position repeatability given topology (2 s chunk halves, ICP-aligned onto the full room)

| Variant | rooms (full topology failed) | wall Δ median / p90 mm | walls within max(1 cm, 0.5 %) | ceiling Δ median / max mm | ceilings ≤ 1 cm |
|---|---|---|---|---|---|
| V0 | 4 (3) | 10.2 / 36.7 | 57 % | 11.4 / 110 | 50 % |
| V1 | 5 (2) | 11.2 / 49.8 | 44 % | 84.9 / 123 | 25 % |
| V2 | 5 (2) | 11.0 / 27.6 | 65 % | 8.7 / 21 | 75 % |
| V3 | 5 (2) | 11.5 / 44.2 | 62 % | 8.6 / 21 | 75 % |
| V4 | 5 (2) | 19.5 / 71.0 | 31 % | 8.4 / 21 | 75 % |
| V5 | 3 (4) | 7.5 / 25.7 | 61 % | 108.8 / 125 | 0 % |
| **V6** | 5 (2) | **9.8 / 33.5** | **73 %** | 8.7 / 21 | 75 % |

Per room (V6): rectangular rooms 3, 4, 8 → 0–10 mm per wall; complex rooms 1 (corridor merged
with a room, 6 walls) and 7 (wardrobes, 8 walls) carry the 33–45 mm residuals; rooms 2 and 6 fail
topology even with all data.

## Caveats
- Halves share the trajectory, so drift is common to both (removed anyway by the per-room ICP).
- Halves have half the data of a real capture; random noise would shrink ~√2 with full captures,
  view-dependent bias would not.
- No GT: these are consistency numbers, not accuracy.

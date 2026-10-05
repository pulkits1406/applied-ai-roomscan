# e20 — video tier: continuous RGB-only tracking [PSEUDO-GT reference = ARKit poses]

Run by a subagent that **stopped before writing its own report** (stream watchdog: no progress for
10 min during a MapAnything model load under memory pressure). This README was written by the main
session from the saved outputs (`out/*_eval.json`, `out/table.md`) and the scripts' docstrings.
Not run: MapAnything windows on `single_scan_with_ceiling` (see "Not run").

## Scripts (see each docstring)
`features.py` ALIKED + LightGlue features/matches · `colmap_lg.py` pycolmap mapper on those matches ·
`moge_depth.py` MoGe-2 metric depth per keyframe (resumable, memory-guarded) · `rgbd_odom.py`
pseudo-RGB-D odometry (`o3d` Open3D RGB-D odometry; `pnp` LightGlue matches lifted with MoGe depth
+ PnP-RANSAC with a fallback ladder; `pnp_pgo` + loop edges and pose-graph; `*_cut` = trajectory
cut at failed links instead of bridged) · `mapanything_windows.py` 16-frame windows chained by
Sim(3) · `blind.py` where frame-to-frame links fail · `report.py` → `out/table.md`, `summary.json`.

## Results (`out/table.md`)
`single_scan_with_ceiling` (215 s, 99 m path; the decisive capture):

| method | pieces | % in largest | longest s | ATE m | 20 s-window ATE m | window scale spread % | drift end m | long-range dist err med/p90 % |
|---|---|---|---|---|---|---|---|---|
| COLMAP SIFT (e08) | 14 | 40 | 21.2 | 1.24 | 0.044 | 17.2 | 0.39 | 10.7/30.4 |
| COLMAP + ALIKED/LightGlue | 1 | 65 | 39.8 | 2.97 | 0.752 | 135.7 | 5.54 | 56.0/81.0 |
| Open3D RGB-D odometry (MoGe depth) | 1 | 100 | 214.6 | 2.95 | 0.861 | 276.8 | 14.51 | 58.8/75.5 |
| PnP on MoGe depth | 1 | 100 | 214.6 | 2.65 | 0.261 | 56.9 | 9.79 | 46.1/73.9 |
| PnP + pose graph (218 loop edges kept) | 1 | 100 | 214.6 | 2.59 | 0.904 | 190.3 | 9.10 | 44.4/72.6 |
| PnP, cut at failed links | 10 | 29 | 61.8 | 0.53 | 0.248 | 14.3 | 1.51 | 7.2/20.9 |

`single_room` (37 s): MapAnything windows — 1 piece, 67 % of frames, ATE 0.92 m, window scale
spread 2.0 %, long-range error 41 % median; PnP — 1 piece, 100 %, ATE 0.42 m, 13 % median.

Failure anatomy (`out/*_blind.json`): 1.8 % of frame-to-frame links fail (36 of 1947 on
with_ceiling, 10 intervals ≤ 1.2 s), at rotation 53 °/s vs 25 °/s overall and surface distance
0.86 m vs 1.59 m. In the pose-graph run, ordinary PnP links have 0.3° median rotation error and
−5.9 % median translation-length error (MoGe depth bias), but fallback links bridging the failures
have 4–13° rotation error each; rotation error accumulates over 905 links.

## Not run
MapAnything windows on `with_ceiling`: its model load (≈ 6.4 GB peak) starved the 16 GB machine and
the subagent stalled; on `single_room` it was the least accurate continuous method, so the main
session decided not to retry it.

## Caveats
Pseudo-GT = ARKit poses (LiDAR/trajectory self-consistency ~3 %). One apartment, two captures.
Walkthrough was not recorded per our protocol (fast turns close to walls are what break tracking).

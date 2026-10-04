# e08 — RGB-only COLMAP SfM on walkthroughs + MoGe-2 metric scale (ARKit/LiDAR = pseudo-GT)

pycolmap 4.2.1 (COLMAP 4.2.1, CPU, SIFT only; ALIKED/LightGlue are not in the wheel). Input: e07 keyframe cache (12 Hz, 960x720).
SIFT peak_threshold 0.001 (the default 0.0067 gives a median of only 429 keypoints on these walls). Sequential matching (overlap 10, or 8 at stride 1)
plus vocab-tree loop detection ("seqloop"). Mapper relaxations: init_min_tri_angle 8, abs_pose_min_num_inliers 20.
`fixed` = PINHOLE true K, no refinement. `refined` = SIMPLE_RADIAL, focal and k1 refined.
```
cd experiments/e08_video_sfm
uv run python sfm.py single_room --stride 1 --matching seqloop                       # ~3.3 min
uv run python sfm.py single_scan_with_ceiling --stride 1 --overlap 8 --matching seqloop --mappers incremental global --intrinsics fixed  # ~29 min (too slow)
uv run python scale.py single_room single_room_s1_p0.001_seqloop incremental_fixed --max-frames 300   # MoGe 1.7 s/frame (MPS)
uv run python jitter.py <capture> <capture>_s1_p0.001_seqloop incremental_fixed
uv run python report.py single_room=single_room_s1_p0.001_seqloop:incremental_fixed single_scan_with_ceiling=single_scan_with_ceiling_s1_p0.001_seqloop:incremental_fixed
```
MoGe (torch) runs in a subprocess (`moge_depth.py`) because torch and pycolmap load conflicting OpenMP runtimes, which abort the process.
| capture / run | mapper | models | registered (any) | largest | ATE largest (Sim3) | mapping s |
|---|---|---|---|---|---|---|
| single_room s1 seqloop | incremental fixed | 7 | 302/342 | 78 | 17 mm | 14 |
| single_room s1 seqloop | global fixed | 5 | 327/342 | 148 | 537 mm (broken) | 10 |
| single_room s2 exhaustive | incremental fixed | 6 | 144/171 | 38 | 15 mm | 15 |
| ceiling s1 seqloop | incremental fixed | 14 | 1762/1948 | 798 | 1.47 m (window scale 0.80–1.14) | 259 |
| ceiling s1 seqloop | global fixed | 7 | 1875/1948 | 1779 | 3.17 m (broken) | 85 |
| ceiling s3 seqloop | incremental fixed | 15 | 446/650 | 52 | 83 mm | 54 |

Fragments break where the camera turns at more than 80 °/s close to blank walls (0.2–0.7 m). Neither exhaustive matching nor loop detection bridges these breaks.
The global mapper (GLOMAP) estimates rotations well (RPE 0.4°) but places positions wrongly on these sequences. Refining the intrinsics gives no gain (fragment focal 769–996 px vs 800 true).
Ceiling s1 models 1–3 (306/252/147 frames): ATE 94/171/20 mm, window-scale range up to 12% → **scale drift within a model**.
**Scale (single_room, incremental_fixed, 6 fragments ≥30 frames, all frames):** MoGe per-frame-median estimator vs each fragment's Sim3 scale:
−2.5, +1.9, −2.2, −1.8, +3.5 % (+45.6 % on fragment 4, where SfM failed: RPE rotation 19°). Single-frame |err| median 2.5–5.5 %, p90 7–12 %.
Convergence (all fragments pooled after normalising each by its own scale; 20 random subsets): median/p90 |err| = 1.6/5.2 % (5 frames), 0.8/3.0 (10), 0.9/1.6 (25), 0.6/1.3 (50), 0.5/1.0 (100), 0.1/0.5 (200).
Read this convergence as optimistic. Signed per-fragment errors of ±2–4 % cancel when pooled; a single reconstruction keeps its own error.
LiDAR oracle on the same pixels gives −2.2 to +6.9 % (not ≈0). LiDAR / ARKit-pose-triangulated depth of the same tracks = 1.009–1.04 (median ≈1.03).
So the pseudo-GT is internally inconsistent at about 3 %, and SfM structure scale differs from trajectory scale by 2–5 %. The ±3 % target is at the noise floor of this evaluation.
**Poses for IMU (`out/*_pose_noise.json`):** cache = 12 Hz. Within continuous runs, poses arrive at about 9 Hz. Continuous runs last at most 3.5–8 s (single_room) and 12–13.5 s (with_ceiling).
White-noise jitter vs ARKit: median 1.4–5 mm, p90 up to 10 mm (fragment 4: 9 mm). Local 2 s rigid-fit RMSE: median 4–29 mm.

Outputs: `out/<capture>_sfm_poses.json` = incremental_fixed. It lists every model (`model` index; each model has its own scale and frame).
`out/<capture>_s*_sfm_summary.json` holds the per-run metrics, plus `summary.json`, `single_room_scale.json` and the PNGs. Large binaries are in `cache/e08/`.
Caveats: all reference values are pseudo-GT (ARKit/LiDAR). The `*_s*_sfm_poses.json` files written before the fix pick global-mapper models (wrong positions), so use the canonical files instead.

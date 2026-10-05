# Experiment dependency map (e00–e21)

What each experiment reads, which script produces each **gitignored** input, which interpreter it
runs under, and whether it needs the GPU. Paths are relative to the repo root. Tracked outputs
(`out/*.json|png|md|jsonl|txt`) are committed. Ignored inputs (`*.ply`, `*.npy`, `*.npz`,
`cache/`, `runs/`, raw captures) must be regenerated. `scripts/prepare_intermediates.sh` does the
cheap ones, and adds the GPU ones with `--gpu`.

**Envs.** These are interpreters, and they cannot be merged:

| env | interpreter (first hit wins) | needed because |
|---|---|---|
| **project** | `uv run python` (`.venv`, Py 3.13) | moge, pycolmap, open3d, transformers |
| **mapanything** | `$ROOMSCAN_PY_MAPANYTHING` → `.envs/mapanything/bin/python` → legacy `cache/e13/.venv/bin/python` | mapanything pins opencv-headless 4.10 |
| **video** | `$ROOMSCAN_PY_VIDEO` → `.envs/video/bin/python` → legacy `cache/e20/.venv/bin/python` | lightglue + kornia |

Docstrings and READMEs still name the legacy `cache/e13/.venv` and `cache/e20/.venv` paths. Any
interpreter from the same row works.

**RAW** = `single_room/`, `single_scan_floor_only/`, `single_scan_with_ceiling/`, read through
`roomscan.stray.StrayCapture`. **WC** = `single_scan_with_ceiling`.

## Gitignored intermediates and their producers

| ignored input | producer | env / GPU | approx. runtime | consumers |
|---|---|---|---|---|
| `single_*/` raw dirs | `unzip single_*.zip` + rename hash dir (see below) | – | seconds | almost every experiment |
| `experiments/e02_fusion/out/<cap>_vox2cm.ply` (91 MB) | `e02_fusion/fuse.py [cap…]` | project / CPU | not recorded (minutes) | e03 cross_capture, e04, e10 segment(_v2), e12 run_moge (WC), e17 doorways (WC) |
| `cache/frames/<cap>/{frames.json,rgb/*.jpg}` (223 MB) | `e07_frame_cache/extract.py [cap…]` | project / CPU | not recorded | e08, e12, e13 (WC, at import), e16, e17, e20 |
| `cache/e09/<cap>_T_wc_corrected{,_v2}.npy` | `e09_drift_posegraph/posegraph{,_v2}.py [cap…]` | project / CPU | 2–74 s per capture | e09 ablation.py (`""` / `_v2`) |
| `experiments/e10_room_seg/out/WC_v2b_h{1.9,2.05,1.6}_labels.npy` | `e10_room_seg/segment_v2.py single_scan_with_ceiling` | project / CPU | not recorded | **h1.9**: e09 per_room_rigid, e12 photo_sets, e17 doorways (equality check). Via `photo_sets.json["label_file"]`: e13 stitch_probe, e17 select_doorway_frames |
| `experiments/e10_room_seg/out/<cap>_{header,morph}_labels.npy` | `e10_room_seg/segment.py` | project / CPU | – | none |
| `cache/e08/<cap>_s<N>_<tag>/` COLMAP DBs and models | `e08_video_sfm/sfm.py` | project / CPU | 3.3 min (single_room) to 29 min (WC) | e08 scale, jitter, report |
| `cache/e08/vocab_tree_faiss_flickr100K_words32K.bin` | `sfm.py` downloads it (or `scripts/fetch_weights.py`) | – | – | e08 sfm `--matching seqloop` |
| `cache/e08/moge/<cap>/<row>.npy` | `e08_video_sfm/moge_depth.py` (subprocess of scale.py) | project / **GPU** | 1.7 s/frame | e08 scale; e20 rgbd_odom/moge_depth (reused through `MOGE_DIRS`) |
| `cache/e12/<row>.npy` (359 files, 35 MB) | `e12_photo_scale/run_moge.py` (not resumable) | project / **GPU** | ≈ 10 min | e12 camh_cue, door_cue; e16 e16lib (run_sweep, check_bf16_weights, summarize) |
| `cache/e16/runs/*.npz`, `cache/e16/crop/*.npz` | `e16…/run_sweep.py`, `crop_probe.py` | mapanything / **GPU** | 24 min GPU (37 min wall); 54 s | e16 summarize, check_bf16_weights; resume |
| `cache/e17/*.npz` | `e17_photo_stitch/run_stitch.py` | mapanything / **GPU** | 354 s GPU | run_stitch `--eval-only` |
| `cache/e20/feat/<cap>/*.npz`, `cache/e20/match/<cap>/{chunk_*.npz,pairs.json}` | `e20…/features.py <cap>` (`TORCH_HOME=cache/e20/torch`) | **video** / **GPU** | single_room 31 s + 106 s; WC 184 s + 1029 s | e20 colmap_lg, rgbd_odom (pnp*), blind, fullrate |
| `cache/e20/moge/<cap>/<row>.npy` | `e20…/moge_depth.py <cap> --stride 2` | project / **GPU** | WC 818 frames, 49 min (3.6 s/frame) | e20 rgbd_odom |
| `cache/e20/colmap/…`, `cache/e20/poses/<cap>_<method>.json` | e20 colmap_lg.py, rgbd_odom.py, `mapanything_windows.py chain` | project (colmap_lg, rgbd_odom) / mapanything (chain) / CPU | 146 s to 1447 s (colmap) | e20 report |
| `cache/e20/mapanything/<cap>_s*_w*_o*/w*.npz` | `e20…/mapanything_windows.py infer` | mapanything / **GPU** | 20–30 s/window; single_room 1120 s | `mapanything_windows.py chain` |
| `cache/e20/fullrate/<cap>/…` | `fullrate.py dump` (project) → `match` (video, GPU) | mixed | – | `fullrate.py eval`. Only `dump` has been run |
| `runs/pseudo_v0/…` | `e15…/make_pseudo_site.py` | project / CPU | ≈ 2 × 50 s | the same script (evaluator) |
| model weights | `scripts/fetch_weights.py` (manifest `envs/weights.yaml`) | – | – | see the manifest |

**Raw zips.** `single_room.zip` → `c00a170fe1/` → `single_room/`; `single_scan_floor_only.zip` →
`1a8384c3f6/` → `single_scan_floor_only/`; `single_scan_with_ceiling.zip` → `c7d28f72c6/` →
`single_scan_with_ceiling/`. Each zip contains exactly one top-level folder, checked with
`unzip -Z1` on 2026-10-05.

## Per experiment

| exp | scripts | reads (besides RAW) | env | GPU / runtime |
|---|---|---|---|---|
| e00 | survey.py, contact_sheet.py `<cap>` | RAW only | project | no |
| e01 | reprojection.py | RAW only | project | no |
| e02 | fuse.py | RAW only → **.ply** | project | no |
| e03 | revisit_residual.py; cross_capture.py | e02 .ply (cross_capture) | project | no |
| e04 | height_maps.py | e02 .ply | project | no |
| e05 | sync.py | RAW only | project | no |
| e06 | mono_scale.py | MoGe-2 weights | project | MPS, 1.6 s/frame × 75 × 2 |
| e07 | extract.py | RAW only → **cache/frames** | project | no |
| e08 | sfm.py, moge_depth.py, scale.py, jitter.py, report.py | cache/frames, vocab tree, cache/e08 models, MoGe-2 | project | sfm CPU 3–29 min; MoGe 1.7 s/frame |
| e09 | posegraph(_v2).py, ablation.py `[""\|_v2]`, per_room_rigid.py | cache/e09 npy (ablation); e10 WC_v2_grid.json + **WC_v2b_h1.9_labels.npy** (per_room_rigid) | project | no |
| e10 | segment.py, segment_v2.py | e02 .ply | project | no |
| e11 | imu_scale, attenuation, lever_arm, noise_robustness, imu_on_sfm(_v2) | e08 `out/*_sfm_poses.json` (tracked) for imu_on_sfm* | project | no |
| e12 | photo_sets → run_moge → analyze / camh_cue / door_cue → door_cue_filter; ensemble_dav2 | cache/frames (WC), e10 v2b_h1.9 labels + v2_grid.json, e02 WC .ply (run_moge), cache/e12, Grounding DINO (door_cue), DAv2 (ensemble_dav2) | project | run_moge ≈ 10 min; door_cue and ensemble_dav2 MPS (not recorded) |
| e13 | run_subsets.py / run_all.sh, summarize.py, stitch_probe.py (never run) | e12 photo_sets.json, cache/frames WC, MapAnything weights (`HF_HOME` defaults to cache/e13/hf) | mapanything | MPS, 1.2–10 s/run + ~40 s load; ≈ 15–20 min/room |
| e14 | debug_rooms.py, visit_repeat.py | RAW only (v0 pipeline) | project | no; ≈ 50 s/plan |
| e15 | make_pseudo_site.py | RAW WC → runs/pseudo_v0, benchmark site.yaml | project | no |
| e16 | e16lib.py, run_sweep.py, crop_probe.py, check_bf16_weights.py, summarize.py | e12 photo_sets.json, **cache/e12/*.npy** (MoGe comparison), cache/frames, MapAnything weights | mapanything (summarize: no GPU) | run_sweep 24 min GPU |
| e17 | doorways.py, select_doorway_frames.py, sift_link.py (project); run_stitch.py (mapanything) | e02 WC .ply + e10 h1.9 labels (doorways), photo_sets.json, cache/frames, doorway_frames.json | project / mapanything | run_stitch 354 s GPU; sift_link ≈ 2 s |
| e18 | roomgeo.py (lib), run_split.py, run_fixed_topology.py | RAW only | project | no; 36–208 s |
| e19 | stitch.py, stitch_v2.py `<cap> <variant>` | RAW only (imports e18 roomgeo) | project | no |
| e20 | features.py, fullrate.py match (video); mapanything_windows.py (mapanything); moge_depth.py, colmap_lg.py, rgbd_odom.py, blind.py, fullrate.py dump/eval, report.py (project) | cache/frames, cache/e20/*, cache/e08/moge, e08 `out/<cap>_sfm_poses.json` (report) | all three | see table above |
| e21 | analyze.py | e16 `out/runs.jsonl` (tracked) | project | no |

Env assignments come from each script's docstring, README or imports. `e20/common.py` imports
only numpy, so it loads in all three envs. `e13/common.py` and `e20/mapanything_windows.py` call
`setdefault` on `HF_HOME=cache/e13/hf` and `PYTORCH_ENABLE_MPS_FALLBACK=1`. `TORCH_HOME` is
never set in code; it is given on the command lines in the e20 docstrings
(`TORCH_HOME=cache/e20/torch`).

## DAG of ignored intermediates

```
 single_*.zip ──unzip+rename──► RAW single_*/
   │
   ├─► e02 fuse.py ──► e02/out/<cap>_vox2cm.ply ─┬─► e03 cross_capture, e04
   │                                             ├─► e10 segment_v2.py ──► e10/out/WC_v2b_h1.9_labels.npy (+ tracked WC_v2_grid.json)
   │                                             │        ├─► e09 per_room_rigid
   │                                             │        ├─► e17 doorways ─► doorways.json ─► select_doorway_frames ─► run_stitch*, sift_link
   │                                             │        └─► e12 photo_sets.py ──► e12/out/photo_sets.json (tracked)
   │                                             │                 ├─► e12 run_moge.py [GPU] ──► cache/e12/*.npy ─┬─► e12 camh_cue, door_cue [GPU]
   │                                             │                 │        (also reads e02 WC .ply)             └─► e16 run_sweep* [GPU]
   │                                             │                 ├─► e13 run_subsets* [GPU] ─► e13 runs.jsonl
   │                                             │                 └─► e16 run_sweep* ─► e16 runs.jsonl (tracked) ─► e21 analyze
   │                                             └─► e12 run_moge (WC floor height)
   ├─► e07 extract.py ──► cache/frames/<cap>/ ───┬─► e08 sfm ─► cache/e08 models ─► scale (─► e08 moge_depth [GPU] ─► cache/e08/moge), jitter, report
   │                                             │                                   └─► e08/out/<cap>_sfm_poses.json (tracked) ─► e11 imu_on_sfm*, e20 report
   │                                             ├─► e12 photo_sets / run_moge / door_cue / ensemble_dav2
   │                                             ├─► e13 common (WC, at import) ─► e13, e16, e17
   │                                             └─► e20 features.py† [GPU] ─► cache/e20/{feat,match}/<cap> ─┬─► colmap_lg ─┐
   │                                                 e20 moge_depth.py [GPU] ─► cache/e20/moge/<cap> ────────┴─► rgbd_odom ─┼─► cache/e20/poses ─► e20 report
   │                                                 e20 mapanything_windows* infer [GPU] ─► chain ─────────────────────────┘
   ├─► e09 posegraph{,_v2}.py ──► cache/e09/*.npy ─► e09 ablation
   └─► e00 e01 e05 e06 e11 e14 e15 e18 e19 (RAW only)

 * = mapanything env   † = video env   everything else = project env
```

## Gotchas

- `segment_v2.py` with no arguments regenerates all three captures. That overwrites the tracked
  `segmentation_v2b.json`, which currently describes WC only. Pass `single_scan_with_ceiling`.
  `prepare_intermediates.sh` does, and also restores the tracked summaries afterwards.
- `segment_v2.py` does not create `out/`. The directory is tracked, so it exists in a clone.
- `fuse.py` rewrites `fusion_summary.json` with only the captures passed in that run.
- `run_moge.py` recomputes every row on each run. The e20 features/moge scripts and e13/e16 sweeps
  are resumable.
- These files sit in `out/` without a producing script (ad hoc, kept as evidence):
  `e10/out/*_v2_h*_labels.npy`, `*_rooms_v2.png`, `segmentation_v2.json` (an earlier segment_v2
  revision), `e06/out/upright_check_857.jpg`, `e13/out/fov_crop_probe.txt`,
  `e17/out/doorway_frames.png`.
- `cache/e20/torch/hub/checkpoints/{depth-save,disk_lightglue_v0-1_arxiv}.pth` are not used by any
  script.
- `cache/e20/mapanything/single_scan_with_ceiling_s6_w12_o4/` is an aborted partial run.

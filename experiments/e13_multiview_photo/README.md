# e13 — MapAnything multi-view metric scale (photo tier, pseudo-GT)

MapAnything v1.1.4 (repo commit `3d10cf7`), checkpoint `facebook/map-anything-apache` (HF sha `00f9c24`, Apache-2.0), PyTorch 2.14.1 on MPS, bf16 (fp32 gives the same numbers to 0.1 pp). Inputs are upright 720x960 frames, fed at 378x504 (resolution set 504, so nothing is cropped). All errors are against Stray LiDAR depth (conf==2, <6 m) and ARKit centres, which are **pseudo-GT**.

```
# isolated venv: mapanything pins opencv-python-headless==4.10.0.84, which conflicts with the project's >=5.0
uv venv cache/e13/.venv --python 3.12
VIRTUAL_ENV=cache/e13/.venv uv pip install torch==2.14.1 torchvision==0.29.1 -e cache/e13/map-anything pandas scipy matplotlib
VIRTUAL_ENV=cache/e13/.venv uv pip install --no-deps -e .
cd experiments/e13_multiview_photo && ./run_all.sh      # run_subsets.py per room, resumable -> out/runs.jsonl
../../cache/e13/.venv/bin/python summarize.py           # -> out/summary.json, out/scale_err_vs_K.png
../../cache/e13/.venv/bin/python stitch_probe.py        # written, NOT run
```

**Coverage: room 1 only** (135 candidates). Each K has 15 seeded, time-spread subsets × 2 variants, except K=8 (2 / 1 runs). The sweep was stopped because of machine memory pressure. Rooms 3/4/6/7/8 and the stitching probe were not run. Two extra room-3 K=3 subsets (smoke test, not in runs.jsonl) gave -34 to -38 %.

| K | \|err\| med / p90, images | same, images+K | mean signed (+K) | focal err | fcorr \|err\| med / p90 (+K) | AbsRel (aligned) | s/run |
|---|---|---|---|---|---|---|---|
| 1 | 31.7 / 36.7 | 29.2 / 35.0 | -24.5 | -34 % | 3.9 / 15.7 | 0.053 | 1.2 |
| 2 | 30.8 / 36.3 | 27.8 / 34.3 | -28.0 | -33 % | 5.7 / 11.7 | 0.065 | 2.8 |
| 3 | 34.8 / 40.9 | 33.4 / 38.6 | -32.3 | -35 % | 4.1 / 8.2 | 0.069 | 5.5 |
| 4 | 34.0 / 38.5 | 31.6 / 36.9 | -33.0 | -34 % | 3.0 / 7.2 | 0.071 | 5.8 |
| 6 | 33.6 / 35.2 | 31.7 / 33.5 | -31.4 | -35 % | 1.5 / 4.6 | 0.069 | 9.8 |
| 8 (n≤2) | 31.9 | 30.8 | -30.8 | -34 % | 2.3 | 0.10 | 22–27 |

Error is `100*(median(pred/LiDAR)-1)`, pooled over all images in a run, so negative means the room comes out too small. The LiDAR/pred ratio is about 1.45–1.53, compared with 1.045 for MoGe-2.

## Findings (measured)
- **Large, consistent under-scaling, about -31 %.** 0 of 76 multi-view runs are within ±8 %. MoGe-2 single-image (median 6.4 %) does far better.
- **The predicted focal is about 35 % too short** (FOV ~70° against a true 48.5°). The model predicts ~65–71° even on 50 % centre crops whose true FOV is 25° (`out/fov_crop_probe.txt`), so its focal output does not respond to crops.
- **Given intrinsics are mostly ignored.** Passing K improves |err| by a median of 1.9 pp (better in 75 of 76 paired runs). Output focal moves by only about 6 % when input fx is halved or doubled. I checked that the ray-direction conditioning path does run (hooked); CPU and MPS agree, and torch 2.8 and 2.14 agree. Upstream issue #93 reports the same symptom: given intrinsics make no difference.
- **fcorr** = each view's depth × (f_true / f_pred), applied after inference. This uses the pinhole relation Z = f·S/s: a wrong focal scales depth by the same factor. With it, errors drop to a median of 1.5–6 %, and p90 is ≤ 8 % from K≥3. It **needs known intrinsics** (EXIF focal plus sensor size, or a per-phone-model table); it cannot be used without them.
- Sim(3) centre scale was skipped on near-collinear sets (n = 8/11/14 at K = 3/4/6). Where computed it is noisy: median 20–32 % under, RMSE 0.6–1.3 m. Pairwise baseline ratio pred/ARKit is about 0.70, so poses are under-scaled like depth.
- Within one run, per-image scale ratios still spread by 7–14 % at K=2–6, so the views do not share one consistent scale.

## Caveats
Single room and single capture. The frames are video stills, not real photos. Bias consistency across rooms is **unknown**. K=8 has n≤2. The model downloaded the DINOv2 hub code into `~/.cache/torch/hub` (outside `cache/`). `cache/e13/hf` holds an aborted partial download of `map-anything-apache-v1`.

# e16 — does MapAnything + true intrinsics generalise across rooms? (photo tier, pseudo-GT)

MapAnything v1.1.4 `facebook/map-anything-apache`, images + true K. Setup is e13's (upright 720x960 at 378x504, MPS, bf16 autocast). Change from e13: the DINOv2 encoder and the info-sharing transformer are cast to bf16 weights, so the process holds about 4.2–4.6 GB instead of 7.7 GB. On 5 subsets this moves metrics by ≤ 0.12 pp (`out/check_bf16_weights.json`). Those 5 runs keep their fp32-weight cache. Subsets are e13's time-spread sampler with e13's seeds, so room 1 subsets 0–9 match e13 (r1 K2 s0 reproduces −27.0 %). There are 10 subsets per (room, K) for rooms {1,3,4,6,7,8} and K ∈ {2,3,4,6,8}, 300 runs in total. No K was skipped (the smallest room, 7, has 16 candidates). MoGe-2 (FOV given) is scored on the same rows from `cache/e12/*.npy`, with the same pooled median of pred/LiDAR (conf==2, <6 m).

**Pseudo-GT floor:** LiDAR depth and the ARKit trajectory disagree in scale by ~3 %, so differences below ~3 % cannot be resolved.

```
PY=cache/e13/.venv/bin/python; D=experiments/e16_photo_scale_generalization
$PY $D/run_sweep.py --rooms 3 4 6 7 8 1 --ks 2 3 4 6 8   # 24.1 min GPU (37 min wall); caches cache/e16/runs/*.npz
$PY $D/crop_probe.py      # Part B(d), 54 s GPU
$PY $D/summarize.py       # no GPU -> out/summary.json, out/per_room_table.md (full table), out/per_room_err_vs_K.png
```

## Part A — per room. Values are for K = 2/3/4/6/8, in % against LiDAR. fcorr = depth × f_true/f_pred per view (fx)
| room | MA-raw bias | MA-fcorr med\|e\| | fcorr p90 | fcorr bias | fcorr ≤8 % | MoGe-2 med\|e\| | MoGe p90 | MoGe bias | MoGe ≤8 % |
|---|---|---|---|---|---|---|---|---|---|
| 1 | −28/−33/−33/−32/−32 | 6.6/2.3/3.1/1.3/2.0 | 14/6/7/5/3 | +7/+1/0/+1/+1 | 60/100/100/100/100 | 1.7/5.9/3.6/5.8/4.2 | 10/10/11/11/10 | −2/−6/−4/−6/−4 | 80/60/80/60/80 |
| 3 | −34/−34/−33/−33/−34 | 2.1/3.3/4.4/3.9/5.0 | 7/6/7/5/10 | +2/+3/+4/+4/+4 | 90/100/100/100/80 | 1.4/2.0/2.8/2.1/2.5 | 7/8/4/3/4 | −1/−2/−3/−2/−3 | 90/90/100/100/100 |
| 4 | −31/−32/−33/−32/−32 | 6.5/6.1/2.6/2.5/2.6 | 11/9/8/6/5 | −3/−3/0/0/0 | 70/70/90/100/100 | 7.5/8.1/8.0/7.5/7.2 | 11/11/10/8/8 | −8/−8/−8/−8/−7 | 60/50/50/80/100 |
| 6 | −32/−36/−36/−34/−34 | 6.6/4.9/2.8/3.4/5.8 | 9/8/7/6/8 | +6/+1/0/+2/+3 | 70/100/90/90/80 | 14.5/13.3/17.5/17.5/16.0 | 20/17/19/19/18 | −15/−13/−18/−18/−16 | 30/20/10/0/0 |
| 7 | −7/−14/−16/−17/−16 | **43.7/34.1/32.2/31.3/32.6** | 57/50/48/35/38 | +44/+34/+32/+31/+33 | 0/0/0/0/0 | 2.7/2.5/1.4/2.3/1.2 | 6/5/3/4/3 | −2/−2/0/−2/−1 | 100 all |
| 8 | −28/−30/−30/−29/−29 | 12.4/10.5/9.9/11.8/10.9 | 33/18/17/13/14 | +12/+11/+10/+12/+11 | 30/30/30/0/0 | 12.3/12.5/11.1/11.3/11.2 | 16/17/13/15/13 | −12/−13/−11/−11/−11 | 0/10/10/0/0 |

Catastrophic runs (|err| > 25 %): MA-raw 243/300, MA-fcorr 48/300 (all in room 7, plus 3 in room 8 at K=2), MoGe-2 1/300. Pooled over all runs: fcorr median 6.0 % / p90 31.5 %, MoGe-2 7.0 % / 16.1 %.

## Part B — what produces the improvement (`out/summary.json`)
- **(a) The focal ratio is a constant, not a measurement.** f_pred/f_true over 1380 views: mean 0.654, sd 0.037, p5–p95 0.62–0.73. Room means are 0.64–0.68 and K means 0.652–0.662. In room 7 it is 0.641 ± 0.008 while room 7's depth ratio is 0.84. Across runs, the correlation of log depth ratio with log focal ratio is 0.04. The predicted aspect is wrong too: (fy_pred/fy_true)/(fx_pred/fx_true) = 1.116. Correcting with fy or with √(fx·fy) instead of fx changes per-room |err| by 2–15 pp (`B_a_fcorr_variants_*`); no variant is best in every room.
- **(b) Depth and pose disagree in scale.** Median depth ratio is 0.685 and median focal ratio 0.644, so depth/focal is 1.05 (p10–p90 0.98–1.31). The pairwise camera-baseline ratio (pred/ARKit, K ≥ 4) varies by room: 0.72 (1), 0.87 (3), 1.12 (4), **3.15 (6)**, 1.52 (7), 0.75 (8). Across runs, baseline/depth has median 1.27 and p10–p90 0.86–4.7, with correlation 0.03. So the error is not one global metric-scale factor. Within a run, the per-view log depth deviation follows the per-view log focal deviation with slope 0.71 (r = 0.30): it partly tracks the focal error, as the pinhole relation predicts.
- **(c) Held-out constant (learned on rooms 1, 3, 4; tested on 6, 7, 8; no EXIF).** c = LiDAR/pred = 1.489 (per-K range 1.449–1.509). Results on rooms 6–8, all K pooled:

  | correction | median \|e\| | p90 | ≤ 8 % | > 25 % |
  |---|---|---|---|---|
  | constant | 8.0 % | 30.5 % | 50 % | 30/150 |
  | per-view EXIF fcorr | 11.4 % | 36.9 % | 35 % | 48/150 |
  | MoGe-2 | 10.9 % | 18.1 % | 39 % | 0 |

  The constant's per-room bias is −2.8 % (room 6), **+25.8 % (room 7)** and +5.5 % (room 8).
- **(d) Crop probe (rooms 3/6/8, K=3, 2 subsets).** At c = 1 / 0.75 / 0.5 the true FOV is 48.5° / 37.3° / 25.4°. The predicted FOV stays at 70.5° / 67.9° / 67.6° even though the crop-correct K is supplied. Raw depth error tracks the crop: −34 / −46 / −59 %. fcorr medians are +1.4 / +1.2 / +13.6 %, and max |fcorr| is 18 / 28 / 50 %, all from room 8.

## Interpretation (not measured directly)
MapAnything appears to assume a fixed ~70° FOV and to infer depth from apparent size under that FOV, so its depth scales roughly with f_pred/f_true. Per-view "focal correction" therefore behaves mostly like a constant ×1.5. It works where the scene matches the model's size prior (rooms 1, 3, 4, 6) and fails where it does not. Room 7 is a small bathroom whose views are dominated by a large mirror and ceiling; MoGe-2 is within 3 % there. Pose scale is unreliable (room 6 baselines are 3× too long), so this setup cannot supply a metric camera layout.

## Caveats
Single capture and a single phone. The photos are video stills from one walkthrough. The room-7 mirror may also affect LiDAR, but MoGe-2 agrees with LiDAR there. 10 subsets per cell, so p90 is roughly the second-worst run.

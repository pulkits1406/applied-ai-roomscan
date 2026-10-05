# e24 — one evaluator path for all three tiers [PSEUDO-GT]

```
uv run roomscan-lidar single_scan_with_ceiling runs/e24/lidar_v11          # reference plan (LiDAR v1.1)
uv run python experiments/e24_three_tier_eval/make_site.py runs/e24/lidar_v11/plan.json
uv run roomscan-bench benchmark/pseudo_gt_lidar_v11 --run runs/e24/three_tier --execute
uv run python experiments/e24_three_tier_eval/video_identity.py          # video room identity (eval only)
```
`--execute` runs every capture from its raw input through its tier's production CLI in a
subprocess (raw → pipeline → plan.json → scores). Wall time on M4: LiDAR 14 s / 6.5 s, photo
127 s per folder set, video 479 s. Outputs copied to `out/`.

**Reference = our LiDAR v1.1 plan of `single_scan_with_ceiling`** (6 rooms; r6 excluded: its walls
are inferred). It is not a laser benchmark (~3 % LiDAR/trajectory self-inconsistency, e08) and the
same capture is never scored against itself. The photo folders and the video come from the same
walk, so they are not independent of the reference's trajectory.

## Results (`out/pseudo_gt_lidar_v11.md`)

| capture | rooms matched | how matched | walls within tier gate | median \|wall err\| | median \|area err\| | interval coverage |
|---|---|---|---|---|---|---|
| LiDAR `floor_only` (independent, no upper-wall sweep) | 1/6 | size only | — (no relative gate) | 17.9 % | 35 % | 0 % (5) |
| LiDAR `single_room` (partial) | 1/6 | size only | — | topology mismatch | 23 % | 0 % (1) |
| photo, view-diverse K6 | 2/6 | folder label | 0 % (±8 %) | 680 % | — | 29 % (7) |
| photo, overlapping sweep K≤8 | 3/6 | folder label | 0 % (±8 %) | 272 % | 44 % | 17 % (12) |
| video (plain clip) | 2/6 | **size only** | 0 % (±3 %) | 17.6 %* | 12.9 %* | 67 %* |

\* identity unverified. By true identity (`out/video_identity.json`, ARKit room labels used for
evaluation only) the three video rooms are r1, r8, r1 with area errors −99.8 %, −60 %, −63 %:
the evaluator's size matching had paired them with other rooms of similar perimeter.

## What this shows
- **The integration works** [FACT]: one command per capture for all three tiers, one evaluator
  path from raw input, schema-valid plans with explicit flags (`rooms_unplaced`,
  `video_fragmented`, `walls_inferred`, `no_floor`, `upper_walls_not_observed`, ...).
- **Accuracy of the photo and video frontends on this data is far from the gates**
  [PSEUDO-GT]. The area of photo-sweep r3 matched within 1.2 % but its walls are 5.73 × 1.58 m
  vs 3.0 × 3.0 m: a coincidence, not a measurement. Wall-level scoring is the honest view.
- **Intervals are overconfident wherever topology is wrong** (coverage 0–29 % vs 90 % nominal):
  the face/scale sigmas describe measurement noise, not topology failure. Calibration (G8) needs
  a term for topology/registration risk or a refusal to emit a room below a coherence threshold.
- **Evaluation-integrity fixes made here:** labelled rooms match only by label (a photo folder of
  an unscored room had been size-matched to another room); size-matched rooms are reported as
  `rooms_matched_by_size_only (identity unverified)`.
- **Instability:** two runs of the same clip gave window-1 areas 10.4 and 0.07 m² (same focal,
  same accepted windows; MPS fp16 numeric differences flip discrete topology decisions on
  marginal clouds). Cached reruns are deterministic.
- Independent LiDAR captures without an upper-wall sweep fail as e10/e18 predicted (1 merged room
  each) and say so in their flags.

# e17 — can per-room photo folders be stitched from images alone? (pseudo-GT)

The algorithm receives only images plus true intrinsics (the EXIF case) for each folder. ARKit poses, LiDAR and doorway positions are used for frame selection and evaluation only. `doorways.py`/`doorways.json` are pre-existing (doorway centres from the e10 v2b segmentation).

```
uv run python experiments/e17_photo_stitch/select_doorway_frames.py   # -> out/doorway_frames.json/.png
cache/e13/.venv/bin/python experiments/e17_photo_stitch/run_stitch.py # 36 MapAnything runs, 354 s GPU; cache/e17/*.npz
uv run python experiments/e17_photo_stitch/sift_link.py               # Method 2, CPU, ~2 s
```

**Doorway photo** for pair (A,B): a walkthrough frame within 0.8 m of the doorway centre with |pitch| < 25°, forward·(doorway→centroid of B's label region) > 0.7, sharp, and < 40°/s. Cameras on A's side or in the unlabelled doorway band are preferred, then the nearest. Up to 2 frames per pair, ≥ 30 rows apart, put in A's folder: 1-4: 8915, 8885 · 3-4: 5535, 5565 (camera already inside room 4; no A-side frame looks into B) · 6-8: 6595, 6565 · 6-7: 7390, 7360.

**Method 1.** One joint MapAnything run (images + K) on 4 A photos, then 0/1/2 doorway photos, then 4 B photos. A and B photos are drawn with e13's time-spread sampler, with doorway rows excluded. 3 seeded subsets × 3 variants per pair. Evaluation fits:
- *fitA* (the protocol): Sim(3) on A's 4 cameras. Rotation is the chordal mean of the camera orientations; scale and translation come from the centres.
- *fitA_s1*: the same, but the scale is fixed to 1, i.e. MapAnything's own metric scale is trusted.
- *fitAll*: a diagnostic only. It also uses B's ARKit poses, so it is a best case that separates a bad reconstruction from a bad A-only alignment.

**Success rule** (set before looking): B camera-centre error median < 0.5 m and B yaw error median < 15°.

## Results (median over 3 subsets; B error in m / yaw error in °)
| pair | variant | true A–B dist | fitA | fitA_s1 | fitAll (diag.) | A×B rel-rot err ° | A fit resid m | success fitA/s1/all |
|---|---|---|---|---|---|---|---|---|
| 1-4 | none / door1 / door2 | 2.47 | 2.63/76 · 2.20/72 · 2.43/61 | 3.16 · 2.16 · 2.57 | 1.17/45 · 1.10/40 · 0.99/36 | 87 · 86 · 74 | 1.2–1.5 | 0/0/0 all |
| 3-4 | none / door1 / door2 | 2.82 | 0.94/5 · 0.88/6 · 0.82/3 | 0.95 · 0.93 · 0.90 | 0.75/12 · 0.60/6 · 0.61/3 | 6 · 7 · 4 | 0.22 | 0/0/1, 0/0/0, 0/0/0 |
| 6-8 | none / door1 / door2 | 2.01 | 2.61/99 · 2.60/100 · 2.44/102 | 2.88 · 2.78 · 2.71 | 1.14/98 · 1.11/88 · 1.16/75 | 96 · 96 · 101 | 0.31–0.35 | 0/0/0 all |
| 6-7 | none / door1 / door2 | 1.18 | 1.22/26 · 1.05/31 · 0.98/34 | 0.80 · 0.61 · 0.58 | 0.50/4 · 0.47/2 · 0.43/15 | 20 · 42 · 43 | 0.33–0.37 | 0/0/2 each |

Measured: **0 of 36 runs pass under the protocol fits.** Doorway photos change the B error by ≤ 0.45 m and do not fix orientation. The fitA scale is degenerate (0.01–0.45, or negative) in pairs 1-4, 6-8 and 6-7. Either A's own predicted layout does not match ARKit (room-1 residual 1.2–1.5 m) or A spans too little to fix a scale (room 6: 0.3–0.5 m RMS). Even with fitAll, B is off by 0.4–1.2 m, and 1-4 and 6-8 are rotated 35–100°. So the joint reconstruction itself is wrong, not just the alignment. 3-4 gets orientation right (3–6°) but places B 0.8–0.9 m off.

**Method 2, SIFT** (480x640, ratio 0.75, MAGSAC F at 1 px; `out/sift_links.json`). The table gives the max verified inliers. The control is a doorway photo against photos of the other 4 rooms.

| pair | door → B | door → A | best A×B, no doorway photo | control max (p90) |
|---|---|---|---|---|
| 1-4 | 0 / 7 | 8 / 16 | 27 | 10 (0) |
| 3-4 | 31 / 55 | 38 / 44 | 20 | **35** (11) |
| 6-8 | 71 / 44 | 101 / 72 | 81 | 9 (8) |
| 6-7 | 19 / 22 | 0 / 25 | 45 | 10 (7) |

The feature link is clearly above control only for 6-8, and there it is not needed: A×B already reaches 81, because rooms 6 and 8 share near-duplicate frames (6600/6605) through the 0.3 m-dilated room membership. 1-4 and 6-7 fall below or near the noise floor. 3-4 sits at the control maximum, which comes from repetitive white-wall/rail texture.

## Interpretation and what is likely needed
A doorway photo taken from A's side looking into B shares almost nothing with A's own photos (door→A ≤ 16 inliers for 1-4), so it cannot act as a bridge. MapAnything also fails to keep 8–10 sparse views of white-walled rooms consistent, even inside one room (A residual up to 1.5 m in room 1). Likely requirements:
1. A short overlapping chain through each doorway, about 3 photos with ≥ 50 % overlap: from inside A showing the door frame and A context, in the doorway, and from inside B looking back at the same door frame. The shared door frame then appears from both sides.
2. A metric pose scale from somewhere other than MapAnything. Its baseline scale varies 0.7–3× by room (e16).

## Caveats
The "photos" are walkthrough stills. A and B sets can contain time-adjacent frames near the doorway (dilated room membership), which makes the problem easier than real photo folders. 3 subsets per variant. ARKit drift over the walkthrough is ignored. The LiDAR/ARKit pseudo-GT scale floor is ~3 %.

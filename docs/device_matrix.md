# Device matrix

Which tier runs on which hardware, and what accuracy each tier honestly delivers **today**.
No iPhone was available: no tier has been measured against laser/tape ground truth, so the
"honest accuracy" column states what is known and of what kind. Numbers are from
`reports/benchmark_report.md` (generated; regenerate with `reports/build_reports.py --regenerate`)
and the experiment READMEs named next to them (e21 photo scale, e26 interval widths).

## Capture side

| Tier | Phones | App | Input used by the pipeline | Verified on a real device? |
|---|---|---|---|---|
| LiDAR | iPhone Pro with LiDAR (12 Pro or newer; 15 Pro+ for the assessment) | Stray Scanner (free) | depth 256×192 (size read from the export), confidence, per-frame poses + intrinsics; RGB not needed | sample exports of an unknown iPhone Pro model (supplied data); current Stray version **not checked** |
| Video | any iPhone 15 or newer | built-in Camera, Video | frames + timestamps only; rotation from file metadata; field of view estimated from the frames; no IMU | no (plain MOV handling tested with synthetic files) |
| Photo | any iPhone 15 or newer | built-in Camera, Photo | pixels + EXIF orientation + 35 mm-equivalent focal; no depth, no poses | no (HEIC handling tested with synthetic files) |

All iPhone 15 models have a main and an ultra-wide (0.5×) camera; the photo protocol does not
require the ultra-wide (it is an open A/B question, `docs/development_capture_protocol.md`).

## Processing side

| Tier | Machine (tested) | Runtime on the sample data (M4, 16 GB) | Models |
|---|---|---|---|
| LiDAR | Apple Silicon Mac, CPU only | 6.6–33 s per capture | none |
| Photo | Apple Silicon Mac with MPS (16 GB); CPU fallback untested | ~137 s for 6–7 room folders | MoGe-2 ViT-L (in process), MapAnything Apache (isolated env) |
| Video | Apple Silicon Mac with MPS (16 GB) | ~480 s for a 215 s clip | MoGe-2, MapAnything |

CUDA/Linux is untested; MapAnything's load peaks near 6.4 GB, so one GPU model runs at a time.

## Honest accuracy per tier

| Tier | What is known | Kind | What is not known |
|---|---|---|---|
| LiDAR | exact synthetic rooms: walls −1.9 … −2.2 mm, ceiling −1.2 mm, doorway +10 mm; on the sample apartment 7/7 rooms built; wall repeatability between two halves of one capture: 60–65 % of walls within max(1 cm, 0.5 %), median Δ 10 mm; ceilings 50–60 % within 1 cm | SYNTHETIC; GT-FREE | absolute accuracy vs laser; repeatability across two captures; the ~3 % disagreement between LiDAR depth and ARKit trajectory scale (e08) means percent-level scale bias cannot be excluded |
| Video | runs end to end; on the sample clip 0 % of walls within ±3 %; the rooms it finds are −60 to −100 % in area by identity; tracking fragments at fast turns close to walls | PSEUDO-GT | whether a clip recorded per protocol (slow turns) is accurate |
| Photo | scale from two model families: 5.3 % median / 14 % p90 per room on walkthrough frames (e21); room geometry: 0 % of walls within ±8 % on simulated photo folders; rooms not stitched | PSEUDO-GT | real HEIC photos; overlapping/ultra-wide sets; any stitched plan |

**Intervals** are reported on every measurement but are **uncalibrated** at every tier
(`calibrated: false`); on pseudo-GT they cover 0–67 % of reference values against 90 % nominal.
They widen with measured geometric instability (photo walls ±0.4–1.3 m where the
reconstruction is unstable, e26) but do not cover rooms that are consistently wrong — those are flagged.

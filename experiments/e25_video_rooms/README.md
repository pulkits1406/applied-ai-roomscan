# e25 — video tier as a product frontend: rooms from a plain clip [PSEUDO-GT = LiDAR v1.1]

Production code: `src/roomscan/video/` (`roomscan-video <clip> <out>`), reusing the photo-tier
inference/scale (`roomscan.photo`) and the shared room step (`roomscan.cloudroom`).

## 1. Rooms inside tracked pieces — negative (`rooms_from_pieces.py`, `sift_pieces.py`)
Upper bound: e20/e08 caches use the TRUE intrinsics and true-FOV MoGe depth.
- **PnP on MoGe depth, cut at blind links** (e20's best continuous-and-accurate candidate):
  fusing MoGe depth over a 10–60 s piece gives areas 6.7–28 m² for the 32.5 m² room r1, 25 m²
  for the 10.8 m² room r8, 81 m² for the 8.9 m² room r3, and a 70-wall polygon: ~25 cm drift per
  20 s (e20) smears every wall. `out/rooms_single_scan_with_ceiling_pnp_s2_cut.json`
- **COLMAP-SIFT pieces** (locally accurate, 4.4 cm 20 s-window ATE) with a GT-free scale (the s
  that minimises occupied voxels of fused MoGe depth): scale −12…−30 %, polygons fragmented (40–90
  walls in r1/r3), because COLMAP models mix frames from across the walk. `out/sift_rooms_*.json`
**Conclusion [MEASURED]:** on the supplied walkthrough, no available tracker yields geometry
coherent enough to measure a room over a tracked piece.

## 2. Overlapping windows measured like photo sets (production v0)
`uv run roomscan-video single_scan_with_ceiling/rgb.mp4 runs/e25/video_with_ceiling` (rotation
inferred for Stray exports; no intrinsics/poses/IMU read). Clip → 427 keyframes at 2 fps → FOV
from MoGe on 16 frames (777.5 px; true ≈ 795 px, −2.5 %) → 18 windows of 12 s × 8 frames →
MoGe-2 + MapAnything (one batched worker) → GT-free coherence check → shared room step.

[MEASURED] 3 of 18 windows coherent (9 incoherent: camera heights spread > 15 cm; 3 no floor in
view; 3 no region/polygon) → 3 unplaced rooms; 543 s first run, 38 s from cache.
[PSEUDO-GT, by identity] the three rooms come from r1, r8, r1 and are −99.8 %, −60 %, −63 % in
area; the evaluator's size-only matching had reported 13 % (fixed: such matches are now marked
identity-unverified). Two uncached runs differed (window 1: 10.4 vs 0.07 m²): marginal clouds
flip discrete topology decisions.

## Conclusion
The video frontend is a working product path (plain clip → plan JSON with explicit
fragmentation), not an accurate one on this footage. The binding constraint is unchanged from
e20: the walkthrough was not recorded per protocol (fast turns close to walls). What a
protocol-following clip does is the device question.

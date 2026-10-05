# Technical report — iPhone capture to dimensioned floor plan, three tiers

*≤ 6 pages. Numbers come from `reports/benchmark_report.md` (machine-generated) or the experiment
named next to them. Evidence kinds: **[PSEUDO-GT]** scored against our own LiDAR reconstruction of
supplied sample data; **[GT-FREE]** internal consistency; **[SYNTHETIC]** rendered exact geometry;
**[BLOCKED]** needs a physical capture or laser GT. No iPhone was available, so there is no
benchmark, no head-to-head and no walk-in test; nothing below is laser-verified accuracy.*

## 1. Problem and requirements

Three capture tiers (2–8 photos per room; a handheld video; LiDAR depth + poses) must produce one
contract: rooms with walls, ceiling height, floor area and openings, a stitched plan with correct
adjacency, damage regions, concealed-damage flags, scope items, a confidence interval on every
measurement, JSON to a published schema and a rendered plan — with intervals that widen honestly as
the data thins. The scoring rewards robustness on unseen captures (walk-in 30 %) and a measured
diagnose-and-repair loop (25 %) over feature count, and penalises confident garbage.
That shaped three engineering rules we kept throughout: measure before deciding (27 experiments,
negative results kept), one shared back end so tiers differ only in evidence and uncertainty, and
honest status/flags on every output.

The only data were three Stray Scanner exports of one apartment, one evening, no ground truth.
They fix the evidence ceiling: our LiDAR reconstruction is the best available reference, and it is
self-inconsistent at ~3 % (LiDAR depth vs ARKit-trajectory scale, e08) — no claim below 3 %
absolute scale is possible from it.

## 2. Architecture and tier design

```
LiDAR  Stray export ── fusion (camera-oriented normals) → rooms (segmentation, visits, registration) ─┐
Photo  room folders ── MoGe-2 per photo + MapAnything joint poses → ensemble scale → gravity ──────────┼→ room step → assemble → Plan JSON → render
Video  plain clip ──── keyframes → FOV from frames → 12 s overlapping windows (photo machinery) ───────┘        (shared)     └→ evaluator
```
- **Room step (shared):** segmentation of free space bounded by vertical structure above 1.9 m (door
  lintels separate rooms, e10); wall faces = density peaks of interior-facing points spanning
  ≥ 1.2 m and reaching ≥ 2 m; rectilinear outline from the line arrangement; floor/ceiling from
  horizontal surfaces (e18). LiDAR calls it directly; photo/video via `cloudroom.py`.
- **Assembly (shared, `assemble.py`):** tiers hand over an outline whose edges are wall faces, a 1σ
  term set per face and a relative scale σ; lengths, areas and perimeter get intervals by a
  finite-difference Jacobian. Statuses `measured | inferred | not_observed`; `quality_flags` say
  what limited the result and how to re-capture.
- **Isolation and reproducibility:** MapAnything runs in its own pinned env as a subprocess
  (OpenCV conflict), pycolmap never shares a process with torch (OMP abort, verified); every plan
  records its code version; model outputs are bitwise repeatable across processes and LiDAR runs
  agree to ≤ 2.3e-11 m (e26).
- **Capture route:** Route 2 — Stray Scanner for LiDAR, the built-in Camera for photo and video
  (`docs/capture_protocol.md`; not field-tested). Device matrix: `docs/device_matrix.md`.

## 3. Geometry, room reconstruction and drift

**LiDAR (v1.1).** The decisive data facts were established first: Stray poses are camera-to-world
with OpenCV axes (6–8 mm reprojection vs 200–430 mm for the ARKit reading, e01); video frame j is
depth row j+1 (e05). Drift is close to rigid within a room and accumulates between rooms: a per-room
rigid fit brings 7/8 rooms to 11–15 mm (e09), while a generic pose graph over ARKit made revisits
worse (negative, e09). So every room is measured **room-locally** from its own visits, each
registered to the longest one. Two failure modes found on the sample apartment (e22):
inter-visit misregistration of 30–160 mm that point-to-plane ICP cannot bridge (two visits 22 cm
apart, seeing disjoint height bands, doubled every wall face), and walls never observed at
structural height. Applying stronger registration or wall inference to every room traded
repeatability for coverage (within-gate walls 65 % → 50 %, a correct room grew 1.9 m²), so v1.1
**escalates only for rooms whose topology fails**: re-registration by face-profile hypotheses
verified by surface overlap, then inferred sides with wide intervals. Result: 7 of 7 non-empty
regions become rooms (v1.0: 5) and the rooms that already worked are unchanged [GT-FREE].

**Drift accountability [GT-FREE].** What we do: room-local measurement (dimensions are immune to
inter-room drift) plus a common Manhattan yaw for placement. The ablation (report §3) is honest about
its effect: the stitched footprint barely changes (room overlap 0.098 vs 0.093 m², yaw corrections
≤ 0.9°) — **room translations are ARKit's**. Translation stitches we built made placement worse
(plane-anchored with shared walls: corner disagreement 54.8 → 69–90 mm median; doorway constraints
measurable in 0 of 4 doorways, e19). For the gate's "poses used as-is" rule, placement is therefore
at risk; dimensions are not.

**Photo.** Scale has no single reliable cue: MoGe-2 alone is 7.5 % median / 19 % p90 per room;
MapAnything with EXIF "focal correction" is really a ×1.5 constant (its predicted FOV is ~70°
regardless of the true 48°, e16) and fails in a mirror bathroom (+31…+44 %); more photos, camera
height and zero-shot door cues did not remove per-room bias (e12). The two families' errors are
partly independent, so the ensemble √(MoGe × MapAnything·c) gives 5.3 % median / 14 % p90 with no
run beyond 25 % [PSEUDO-GT, leave-one-room-out, e21]. Geometry is the binding problem: MapAnything's
joint poses are 44–66° wrong for view-diverse low-overlap photos and 2–7° for overlapping sets
(e23, simulated from walkthrough frames); single photos cannot measure a room; single doorway photos
link rooms in 0/36 cases (e17). Result on simulated folders: rooms unplaced, **0 % of walls within
±8 %** [PSEUDO-GT].

**Video.** No RGB-only tracker gave a continuous and accurate walkthrough on the sample clip: SIFT
COLMAP fragments into 14 pieces, depth odometry is continuous but drifts 9–15 m, and only 1.8 % of
frame links fail — all during fast turns close to walls (e20). Fusing depth over tracked pieces
smears walls (e25). The shipped v0 measures rooms from 12 s overlapping windows, accepted only if
GT-free coherent (camera heights agree within 15 cm): 3 of 18 windows on the sample clip, rooms
−60…−100 % in area by identity, **0 % of walls within ±3 %** [PSEUDO-GT]. Fragmentation is explicit.

## 4. Measurements, uncertainty and calibration

**Error budget (1σ, uncalibrated, per face unless stated).**

| Term | LiDAR | Photo | Video | Source |
|---|---|---|---|---|
| face measurement | 7 mm (topology-only face 30 mm) | 30 mm | 30 mm | e18 half-to-half / provisional |
| inferred (never-observed) face | 100 mm | 150 mm | 150 mm | provisional |
| stability (re-run under ±1° yaw and point halves) | measured per face | measured | measured | `quality.py`, e26 |
| registration (camera-height spread) | — | measured | measured | e23/e26 |
| scale, shared per room | 0.4 % | 8.7 % | 8.7 % ⊕ 3 % FOV | e08 / e21 / e25 |
| levels (floor, ceiling) | 5 mm each | 30 mm + scale | 30 mm + scale | e18 |

Terms are independent and propagated separately, so every measurement's `error_budget` names its
contributions; scale is reported as `shared:scale` because it is one factor per room. The
stability term replaced a weakness we measured ourselves: a parameter that should not matter (grid
padding) moved one video room from 10.4 to 0.07 m² while its interval stayed narrow (e26). Now
unstable geometry widens (photo walls ±0.1 m → ±0.4–1.3 m) and stable LiDAR rooms do not; rooms with
a floor-area interval beyond ±50 % are flagged `geometry_unreliable`.

**Calibration analysis.** Coverage of the 90 % intervals on pseudo-GT is 0–67 % (report §1).
Widening does not and should not cover rooms that are consistently wrong (e.g. a photo set that
reconstructs the larger space its camera looks into); those are flagged instead. True calibration
needs laser GT [BLOCKED]; the evaluator already emits residual rows tagged by GT kind and site
purpose, and `bench/leakage.py` refuses benchmark sites as calibration or tuning input.

## 5. Benchmark and fix-loop evidence

**Benchmark [BLOCKED].** The evaluator implements every gate (openings with misses and phantoms,
ceiling and repeat spread, repeatability pairs, stitch/adjacency/overlap, tier wall gates,
head-to-head, coverage) and runs captures from raw input (`--execute`). Without captures, the
report contains PSEUDO-GT tier results, GT-FREE repeatability between two halves of one capture
(60–65 % of LiDAR walls within max(1 cm, 0.5 %), median Δ 10 mm; ceilings 50–60 % within 1 cm),
the drift ablation, SYNTHETIC validation and timing (LiDAR 7–33 s, photo ~137 s, video ~480 s on an M4).
An evaluation-integrity bug was caught and fixed on the way: size-only room matching had paired
video rooms with the wrong rooms and reported 13 % area error instead of −60…−100 %; matches now
carry their method and size-only matches are labelled unverified.

**Fix loop (`reports/fix_loop/`).** The real requirement — the worst gate on our own benchmark — is
blocked; our worst pseudo-GT results (photo/video walls) are not fixed and not claimed. The shipped
cycle we can regenerate: on an exact synthetic two-room capture, two room dimensions came out
−499 mm and −1004 mm (areas −16.7 %, −33.6 %). Root cause: the segmentation's 0.6 m morphological
closing eroded rooms near the grid border (margin 0.3 m / 0 m), and the 0.5 m room crop then cut a
wall away; evidence: region inset 0.27/0.53 m with observed floor up to the wall. Fix: pad the grid
beyond the closing radius. Predicted: noise-floor errors; result: all eight walls within −1.9…−2.2 mm,
areas −0.12 %/−0.16 %, no change on the real capture (≤ 0.36 mm). The declaration is retrospective
and the data synthetic; we say so.

## 6. Failure modes, limitations and walk-in readiness

- **Coverage-hungry topology (LiDAR):** walls exist only where their top edge was observed;
  captures without an upper-wall sweep produce no rooms (`upper_walls_not_observed`). Protocol rule:
  every wall's top edge in view.
- **Photo and video geometry:** not reliable on the supplied recordings; photo stitching unsolved.
  The overlapping-photo and slow-turn rules are hypotheses until the development capture
  (`docs/development_capture_protocol.md`, pre-registered readings) — the next decision depends on it.
- **Mirrors, glass, wet-look surfaces, low light:** a mirror-dominated bathroom breaks MapAnything's
  scale (+31…+44 %, e16); glass produces LiDAR depth artefacts (e00); no targeted captures yet.
- **Other gaps:** no window detection; single dominant ceiling level (one room's ceiling flips
  104 mm between halves); doorway widths carry a +1 cm bin bias (synthetic); damage regions, the
  concealed-damage rules (CD1–CD4) and scope items are implemented as a tested mechanism, but there
  is **no damage detector** and no damage benchmark.
- **Walk-in readiness:** all three tiers run cold, one command each, on inputs they have not seen
  (sample, synthetic), with `roomscan-inspect` checking real file formats first. On a real iPhone
  capture we expect LiDAR to produce a usable plan if the protocol is followed, and photo/video to
  run but measure poorly — the intervals and flags are designed to say so rather than hide it.

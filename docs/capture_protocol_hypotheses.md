# Capture protocol — hypotheses under test (NOT the final one-page protocol)

Each rule states the evidence for it, or says it is untested. The final one-page protocol will
be written only after the physical pre-check (docs/physical_benchmark_procedure.md §7).

## Route: stock apps (Route 2) — provisional

| Tier | App (provisional) | Evidence / risk |
|---|---|---|
| LiDAR | Stray Scanner | Format verified on 3 sample captures (e00–e05). Refuses to record on non-LiDAR phones (source code). |
| Video | stock Camera **or** Sensor Logger (video + IMU) | IMU makes metric scale ~1–2 % on ARKit trajectories (e11); without IMU, scale relies on monocular depth (~5–10 % bias). Whether video+IMU still counts as the "video tier" is an **open interpretation question**. |
| Photo | stock Camera (HEIC, EXIF kept) | EXIF focal length removes intrinsics ambiguity (MoGe error 9.8 % → 6.4 % with known FOV, e06). |

## Rules for the LiDAR walk (hypotheses)

| Rule | Status | Evidence |
|---|---|---|
| Point the phone up at the **top of every wall and the ceiling** in every room | **supported** | Room segmentation needs vertical structure above ~1.9 m (door lintels separate rooms). `floor_only` and `single_room` never observed above ~1.7 m → segmentation found 0 rooms; `with_ceiling` → 8 regions (e10). Ceiling height impossible otherwise (e04: `floor_only` "ceiling" = furniture tops). |
| Leave doors **open** | supported (indirect) | Geometric doorway detection = lintel present + no wall at mid-height + floor visible (lidar_pipeline v0 found 4 doorways); a closed door reads as wall. |
| Finish one room before moving to the next (one continuous visit per room) | **hypothesis** | Drift is ~rigid within a room and accumulates between rooms (e09 per-room ICP: 7/8 rooms drop to 11–15 mm median). Each room's geometry is taken from its longest visit. Revisits help only once visits are aligned and averaged (not yet implemented). |
| Walk back to the start ("loop") | **unsupported so far** | Pose-graph loop closure v1/v2 did not reliably improve consistency (e09). May still help the stitched footprint; keep as optional until tested on our own captures. |
| Stay 1–3 m from walls, move slowly | supported (literature + data) | Confidence-2 depth is mostly < 3 m (e00); Apple RoomPlan guidance. |
| Avoid pointing at mirrors / glass for long; turn on lights | hypothesis | Glass shower/windows produce depth artefacts in samples (e00 contact sheets). |

## Rules for photos (hypotheses)

| Rule | Status | Evidence |
|---|---|---|
| 2–8 photos per room, one folder per room | required by assessment | — |
| More photos per room improve scale | **mostly refuted** | MoGe-2: |error| median ~6 % at K = 1…8; per-room bias dominates (e12). |
| Stand in a corner, phone level, wide view of opposite walls with floor and ceiling lines visible | hypothesis | Single-image shape error after scaling is 1.3 % (e06): geometry is good when walls are in view. |
| One photo from each doorway looking into the adjacent room | **hypothesis, being tested** | Needed for stitching: per-room folders share no poses. MapAnything mixed-room probe pending (e13). |
| A reference object for scale (A4 sheet, card) | **not assumed allowed** | Would likely fix scale, but may violate "any picture in, results out"; question for the user. |
| Door height as a scale cue | **not deployable yet** | Oracle (true doors only): ~5 % per photo; zero-shot detector boxes: fails (e12 door cue). |
| Camera-height cue ("hold at chest height") | **refuted** | Single-image floor-plane height error 21 % median (e12 camh_v2); operator's own camera height varies ±9 cm. |

## Rules for video (hypotheses)

| Rule | Status | Evidence |
|---|---|---|
| Walk at normal pace with natural hand motion | supported (for IMU scale) | IMU scale needs acceleration excitation; samples had 0.4–0.55 m/s² rms and gave ~1–2 % (e11). |
| Same room order and ceiling sweep as LiDAR | hypothesis | Lets one geometry back end serve all tiers. |
| Keep under ~3–4 minutes | hypothesis | Processing time + drift; sample captures were 37–215 s. |

## Potentially non-compliant ideas (flagged, not adopted)

- Reference object in every room (A4 / card): may contradict "any picture in, results out".
- Using ARKit poses on non-LiDAR phones for the video tier (e.g. Record3D/NeRFCapture): the
  photo tier explicitly says "no poses"; the video tier does not, but a pose-logging app makes
  the video tier look like the LiDAR tier without depth.
- Video + IMU (Sensor Logger): benchmark literature calls this visual-inertial, not video-only.
- Requiring a specific app for photos/video: the assessment says photos/video "from any iPhone 15
  or newer"; stock Camera must therefore always work, with any extra app as an optional boost.

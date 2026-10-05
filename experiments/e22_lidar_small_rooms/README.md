# e22 — why LiDAR v1 drops regions 2 and 6, and a general fix [MEASURED, GT-free]

Capture: `single_scan_with_ceiling`. No GT; every number below is internal consistency.

## Commands
```
uv run python experiments/e22_lidar_small_rooms/dump_rooms.py single_scan_with_ceiling [coarse]   # cache/e22 room clouds
uv run python experiments/e22_lidar_small_rooms/diagnose.py single_scan_with_ceiling 2,6,3,4 [_coarse]
uv run python experiments/e22_lidar_small_rooms/per_visit.py single_scan_with_ceiling 2,6
uv run python experiments/e22_lidar_small_rooms/compare_registration.py single_scan_with_ceiling
uv run python experiments/e22_lidar_small_rooms/repeat.py single_scan_with_ceiling v1_0,reg,rules,reg_rules
uv run python experiments/e22_lidar_small_rooms/repeat.py single_scan_with_ceiling ladder
uv run roomscan-lidar single_scan_with_ceiling runs/e22/v1_0_repro --no-ladder     # v1.0 behaviour
uv run roomscan-lidar single_scan_with_ceiling runs/e22/ladder                     # v1.1 (default)
```
All scripts call the production modules (`roomscan.lidar.*`); there is no experiment copy of the
pipeline.

## Diagnosis (`out/diagnose.json`, `out/per_visit.json`, `out/diag_room*.png`)
Wall existence needs a density peak of wall-facing points whose supporting points span ≥ 1.2 m
vertically and reach ≥ 2.0 m (V6 rule).

- **Region 2 (≈ 3 m², low ceiling 2.28 m): inter-visit misregistration.** Visit 1 (95–103 s) never
  looked above 1.52 m; visit 2 (198–205 s) saw only 1.3–2.29 m. Their u-faces agree within 1 cm
  but **both v-faces are offset by +22/+23 cm** in the same direction (2.675→2.897, 5.005→5.240):
  drift, not furniture. ICP (5 cm correspondence radius) cannot bridge 22 cm, and the visits barely
  overlap in height, so visit 2 was merged unaligned. Each wall then appears as two half-height
  faces, neither of which passes the extent test → no v lines → no polygon.
- **Region 6 (≈ 1.3 m² WC): missing evidence + misregistration.** Visits disagree by 8 cm (u) and
  15–20 cm (v); additionally the wall on one side is **never observed above 1.31 m in any visit**,
  so no line can exist there regardless of registration.
- **Registration is a general problem, not a small-room one.** Face agreement between each visit
  and the reference visit (median |offset| of same-side faces, `compare_registration.py`): raw poses
  30–160 mm across rooms; ICP fixes some (room 7 49→5.5 mm) but not visits that are > 5 cm off.
  e03's 28 mm revisit median understates per-room visit disagreement on this capture.
- **Topology test is knife-edge.** With better registration room 3's main wall (18 k points)
  failed by 1 cm (p98 height 1.99 m vs 2.00 m threshold) because residual 2–3 cm misalignment moved
  its upper part out of the ±2 cm test window.

## Fix candidates (`out/repeat_*.json`; e18 protocol on production code: topology from all data,
wall lengths re-measured from two disjoint 2 s-chunk halves; gate max(1 cm, 0.5 %))

| config | rooms with polygon | wall Δ med / p90 mm | within gate | ceilings ≤ 1 cm |
|---|---|---|---|---|
| v1.0 (ICP, V6 rules) | 1,3,4,7,8 | 10.1 / 41.9 | 65 % | 50 % |
| verified re-registration everywhere | 1,2,4,7,8 (loses 3) | 10.9 / 36.9 | 50 % | 100 % |
| 3 cm window + inferred sides everywhere | all 7 | 9.2 / 43.9 | 66 % | 60 % |
| both everywhere | all 7 | 10.0 / 40.3 | 52 % | 80 % |
| **ladder (shipped)** | **all 7** | 10.1 / 41.0 | 60 %* | 60 % |

\* the ladder leaves rooms 1, 3, 4, 7, 8 byte-identical to v1.0; the drop from 65 % is the two
recovered rooms (room 2: 2/6 walls within gate). Room 6's inferred/unmeasured faces are not
re-measured by the halves (Δ = 0 by construction); counting only walls whose own face was measured
gives 35 walls, 66 % within gate.

Re-registration ("verified": per-axis face-profile cross-correlation hypotheses, each verified by
vertical-surface overlap fitness, must beat the unmoved visit by 0.03, then ICP refine) repairs the
doubled faces (room 2: 160 → 16 mm; room 6: 42–120 → 0–7 mm; room 3: 82 → 25 mm) **but** lowers
half-to-half repeatability when applied everywhere: its discrete hypothesis choice can differ
between data subsets. Inferring sides everywhere grows rooms that were correct (room 7 +1.9 m²,
room 4 4 → 6 walls). Trade-off recorded; neither is applied globally.

## Decision [DECISION-PROV] — LiDAR v1.1
`roomscan.lidar.roomgeo.measure` escalates per room **only when topology fails**:
v1.0 → re-registered visits → inferred sides at the observed-floor boundary. Each step is a
quality flag (`rX_visits_reregistered`, `rX_walls_inferred`). Inferred faces carry σ = 10 cm and
`status: inferred`; faces placed from topology only (`unmeasured`) σ = 3 cm (v1.0 gave them the
measured-face σ of 7 mm, which was overconfident). Result: 7 of 7 non-empty regions produce rooms
(was 5), doorways 2 → 4. `--no-ladder` reproduces v1.0 geometry exactly (checked wall by wall).

## Open
- Production v1.0 on the e18 protocol gives 65 % walls / 50 % ceilings within gate vs e18's
  experiment-code V6 73 % / 75 %. The difference comes from how halves are built (production:
  per-visit registration inside each half; e18: chunk clouds ICP'd straight onto the full cloud)
  and from room 3's bimodal ceiling (2.96 vs 3.07 m levels; the dominant pick flips between
  halves, 104 mm). Production numbers are the ones to quote.
- Registration remains the main LiDAR repeatability risk; a real repeat capture made per protocol
  (every wall's top edge in one sweep) is needed to know whether the doubled faces occur at all
  when visits are not split by height.

# HANDOVER — submission state (2026-10-06)

Authoritative project state at submission. Navigation for an assessor: `SUBMISSION_MANIFEST.md`.
Requirement status: `docs/compliance_matrix.md`. Reasoning record: `ENGINEERING_LOG.md` (all
phases). Earlier, more detailed handovers are in git history (`git show 4435f2a:HANDOVER.md` —
phase 5, `2b53124` — phase 4, `8328541` — phase 3).

> **No iPhone was available.** There is no laser/tape benchmark, no real-device capture, no
> head-to-head and no walk-in test. Every accuracy number is PSEUDO-GT (our own LiDAR reconstruction
> of supplied sample data as reference; ~3 % LiDAR/trajectory self-inconsistency), GT-FREE
> (internal consistency) or SYNTHETIC (rendered geometry). None is benchmark accuracy.

## 1. What exists

| Area | State | Where |
|---|---|---|
| LiDAR tier v1.1 | IMPLEMENTED; 7/7 sample rooms; synthetic walls ~2 mm; repeatability 60–65 % of walls within gate between halves of one capture (GT-FREE) | `src/roomscan/lidar/`, `roomscan-lidar` |
| Photo tier v0 | IMPLEMENTED end to end; **inaccurate** on simulated folders (0 % walls within ±8 %); rooms unplaced | `src/roomscan/photo/`, `roomscan-photo` |
| Video tier v0 | IMPLEMENTED end to end; **inaccurate** on the sample clip (0 % within ±3 %; rooms −60…−100 % by identity); fragmented, unplaced | `src/roomscan/video/`, `roomscan-video` |
| Shared back end | room step, Jacobian-propagated intervals with named error terms, geometry-quality widening, render, provenance | `cloudroom.py`, `assemble.py`, `quality.py`, `render.py`, `provenance.py` |
| Damage | mechanism only: regions, rules CD1–CD4, scope items; **no detector** | `damage.py`, `roomscan-damage` |
| Evaluator | all gates, `--execute` from raw input, partial tape GT, identity-aware matching, leakage guard, figures | `src/roomscan/bench/`, `roomscan-bench` |
| Input checks | real-file format/convention report | `inspect.py`, `roomscan-inspect` |
| Reports | benchmark report, fix-loop bundle, technical report, reproduction audit — all numbers generated | `reports/` |
| Capture docs | one-page Route-2 protocol (not field-tested), device matrix, development-capture protocol | `docs/` |
| Reproducibility | env recipes, pinned weights + fetch, intermediates DAG, clean-clone audit | `envs/`, `scripts/`, `docs/reproduction.md` |
| Tests | 39 pass with sample data (35 on a clean clone) + 1 opt-in GPU test | `tests/` |

## 2. How to run and reproduce

```
uv sync && scripts/setup_envs.sh mapanything && uv run python scripts/fetch_weights.py
uv run pytest -q
uv run roomscan-lidar <stray_dir> out/   |  uv run roomscan-photo <room_folders> out/  |  uv run roomscan-video <clip> out/
uv run roomscan-bench benchmark/<site> --run runs/<id> --execute
uv run python reports/build_reports.py --regenerate    # every reported number, from raw sample inputs (~25 min, M4)
reports/fix_loop/run.sh                                # fix-loop before/after (CPU, ~2 min)
scripts/clean_clone_audit.sh <empty_dir> [--weights-from <hf cache>]
```
Sample captures (~870 MB zips) are not in git and must be provided alongside the repository.

## 3. Key decisions (final unless marked)

- Stray conventions: T_wc with OpenCV camera axes, world +y up, video frame j ↔ depth row j+1 (e01, e05).
- One shared representation and back end; tiers differ in evidence and σ terms only.
- LiDAR: room-local measurement; per-room escalation ladder only when topology fails (e22);
  segmentation padded 1 m (fix loop); placement `arkit_yaw` (PROVISIONAL; ablation shows it is
  nearly "poses as-is" for translation).
- Photo: MoGe-2 geometry on MapAnything joint poses, scale = two-family ensemble (e21) (PROVISIONAL).
- Video: 12 s overlapping windows with a GT-free coherence gate (PROVISIONAL).
- Uncertainty: named per-face terms incl. measured stability; never a blanket factor; uncalibrated.
- Development data (`dev_`) never counts as benchmark evidence or unnamed calibration input.
- Ultra-wide photos: A/B test only. Sensor Logger/IMU: diagnostic only, not the video contract.

## 4. Blocked by a physical device

Benchmark set B1–B5, gates G1–G3 and G5–G8 on real data, head-to-head, fix loop on a real gate,
raw benchmark data, real-file validation (current Stray, HEIC, MOV), field test of the protocol,
walk-in. Prepared: `docs/development_capture_protocol.md` (30-minute session, pre-registered
readings for LiDAR / video / photo questions), `benchmark/_template_dev/`, e27 chain evaluation,
`docs/physical_benchmark_procedure.md` (the later laser session).

## 5. Biggest risks

1. Photo and video geometry are not reliable; photo stitching has no working method.
2. Drift gate: placement translations are ARKit's (ablation: negligible footprint change).
3. Intervals uncalibrated; consistently wrong rooms are flagged, not covered.
4. Real-device behaviour of every tier is untested.

## 6. Next steps (in order)

1. Development capture with a borrowed iPhone Pro, exactly per `docs/development_capture_protocol.md`;
   `roomscan-inspect` every file; resolve every `CHECK`; `roomscan-bench --execute` on the dev site.
2. Decide photo/video/LiDAR protocol from those answers (not from the sample recordings).
3. Laser benchmark session → calibration from named sources → the real fix loop.

## 7. Do not

Treat Stray poses as ARKit camera axes · pair video frame j with depth row j · present pseudo-GT or
synthetic numbers as benchmark accuracy · compare runs without matching `provenance` · use `dev_`
data as benchmark evidence · apply visit re-registration or wall inference to every room (e22) ·
judge rooms by area alone or trust size-only room matches as identity (e24) · retry generic pose
graphs or plane-anchored translation stitches without new evidence (e09, e19) · edit
`src/roomscan/lidar_pipeline.py` (frozen v0) or `geometry.py` defaults · run two GPU models at once ·
make 0.5× photos mandatory before the A/B · tune photo/video on the supplied recordings.

## 8. Git and housekeeping

- Branch `master`; tag `engineering-baseline-v1` = verified baseline before submission preparation;
  submission commits follow it. No remote is configured.
- Ignored on purpose: raw data and zips, the case PDF, `cache/`, `runs/`, `.envs/`, `.weights/`,
  `*.ply`, `*.npy`, `*.npz`.
- Temporary files outside the repo (safe to delete by the user): the session scratchpad (clean-clone
  audit) and `prepare_intermediates.sh` backups named `roomscan-keep.*` in the system temp directory.

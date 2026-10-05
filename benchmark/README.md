# Benchmark format

One directory per physical site. Drop raw captures and measurements in, then:

```
uv run roomscan-bench benchmark/<site> --run runs/<run_id>
```

scores every `runs/<run_id>/<capture_id>/plan.json` against `site.yaml` and writes
`runs/<run_id>/eval/<site_id>.{json,md}` with all gates, repeatability, interval coverage and the
head-to-head table. (Running the pipeline over the captures to produce the plans is a separate
step; `--execute` will be added when the pipeline exists.)

```
benchmark/<site_id>/
  site.yaml                 ground truth + capture manifest (schema: src/roomscan/bench/gt.py)
  captures/<capture_id>/    raw files exactly as exported from the phone
      lidar:  Stray Scanner folder (rgb.mp4, depth/, confidence/, odometry.csv, imu.csv, camera_matrix.csv)
      video:  clip (.MOV) [+ sensor-log zip if the protocol uses one]
      photo:  <room_id>/IMG_xxxx.HEIC ...   (one folder per room; folder name = GT room id)
  incumbent/<app>/          the competitor app's native export (PDF/CSV), untouched
  measurements/             photos of the tape/laser sheet, so GT values are auditable
```

## Ground-truth conventions (what an evaluator would measure)

| Quantity | Definition | Procedure |
|---|---|---|
| Wall length | finished-surface corner to corner, horizontal, at ~1.0 m height | laser, 3 readings, median used |
| Wall order | clockwise seen from above, starting at the wall holding the room's entry opening | — |
| Ceiling height | finished floor to finished ceiling | ≥ 3 points per room away from beams/fixtures, median used |
| Opening width | clear opening between finished jambs (door frame inside faces), at mid-height | laser or tape, 2 readings |
| Floor area | only if the room is not a rectangle (rectangles use opposite-wall means) | — |
| Topology | which rooms connect through which opening | `connects:` on each opening |
| Damage | surface (wall id / floor / ceiling), class, extent w × h in metres | tape |

`gt_kind` must be one of `laser`, `tape`, `pseudo_lidar`. Anything `pseudo_lidar` is reported with
a PSEUDO-GT banner: those references come from our own LiDAR reconstruction and measure tier
consistency, never accuracy.

## Required benchmark composition (from the assessment)

- one multi-room capture: ≥ 3 rooms + a connector, at **all three tiers** (photo tier = per-room folders)
- one furnished room with staged damage spanning ≥ 2 classes
- ≥ 1 room captured twice at the same tier → give both captures the same `repeat_group`
- head-to-head: 2 rooms scanned with the incumbent app; fill `incumbent:` with its numbers and keep the export

See `_template/site.yaml`.


## Site purposes and tuning leakage (enforced by `src/roomscan/bench/gt.py` and `bench/leakage.py`)

Every `site.yaml` declares `purpose`, and capture ids carry the matching prefix:

| purpose / prefix | Data | May be used to tune or calibrate? | Counts as benchmark evidence? |
|---|---|---|---|
| `dev_` | development captures (`docs/development_capture_protocol.md`): answer protocol/model questions | tune: yes; calibrate: only when named explicitly | **no** |
| `bm_` | the benchmark (laser GT) | **no** | yes |
| `rep_` | repeat captures for repeatability | **no** | yes |
| `inc_` | incumbent-app comparison | **no** | yes |
| `walk_` | walk-in proxy (second person, one-page protocol only) | **no** | yes |
| `fail_` | deliberate failure-mode captures | tune: yes | no (checks that failures are flagged) |
| `pseudo`, `synthetic` | our own reconstructions / rendered geometry | consistency and code tests only | no |

Rules: a benchmark-purpose site cannot contain a `dev_` capture (validation error); `dev`/`bm`
sites need laser/tape GT; calibration takes an explicit list of sources and refuses benchmark
sites; any change made after looking at `bm_` results is logged as a fix-loop step with its
before/after. Templates: `_template/site.yaml` (benchmark), `_template_dev/site.yaml` (first
development session). Partial tape GT is allowed: list every wall in order and leave out
`length_m` for walls that were not measured.

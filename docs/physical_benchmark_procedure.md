# Physical benchmark procedure (draft, for a borrowed/rented device)

Status: draft. Everything here can be done in **one session of ~4–5 hours** with one borrowed
iPhone Pro. Items marked **[verify]** must be checked on the device before the main session.

## 1. What to borrow / have

| Item | Requirement | Why |
|---|---|---|
| iPhone | **Pro model with LiDAR** (15 Pro / 16 Pro / 17 Pro; 12–14 Pro acceptable for development, record the model) | One Pro phone covers all three tiers (photo and video tiers do not use LiDAR). The evaluator uses an iPhone 15+, so a 15 Pro or newer is preferred for the final benchmark. |
| Laser distance meter | ±1.5 mm class (e.g. Bosch GLM 40 / GLM 50-27, Leica DISTO D2) | Ground truth at mm level; gates are 1–2 cm |
| Tape measure (5 m) | any | Opening widths, damage extents |
| Masking tape + marker | — | Label rooms/walls in photos so GT is auditable |
| Laptop with AirDrop or cable | — | Getting files off the phone |

No Apple Developer account is needed: every app below is a stock App Store app.

## 2. Apps to install (free tiers)

| App | Tier | Notes |
|---|---|---|
| Stray Scanner (v1.4+, free) | LiDAR | Our loader is verified against its format |
| Camera (stock) | photo, video | Photos: 1×, Lens Correction on, HEIC kept; video: 1×, 30 fps, 1080p or 4K |
| Sensor Logger (free, v1.67+) | video (+IMU) | **[verify]** per-frame video timestamps, IMU rate while recording, export zip contents |
| magicplan (Starter, free: 2 projects) | head-to-head | **[verify]** which exports the free plan allows (sketch PDF with dimensions, statistics CSV) |

Transfer settings: AirDrop with *Options → All Photos Data* ON (keeps HEIC + EXIF), or USB with
*Settings → Photos → Transfer to Mac or PC → Keep Originals*. Never send via chat apps (strips EXIF).

## 3. Site requirements (fixed by the assessment)

- One property with **≥ 3 rooms + a connector** (hallway/corridor), doors between them.
- One **furnished** room with **staged damage of ≥ 2 classes** (removable: e.g. a printed/painted
  water-stain patch on paper taped flat to a wall, a drawn "crack" on tape, a peeling-paint
  patch). Photograph and measure each damage extent.
- Include, if available: a mirror, a window/glass door, a glossy floor, and one dim room — the
  assessment asks us to cover mirrors, glass, wet-look surfaces and low light.
- Name rooms with short ids (`kitchen`, `hall`, `bed1`, …); put a masking-tape label on each
  room's entry-wall so photos are self-documenting.

## 4. Ground-truth measurement (do this first, ~90 min)

Follow `benchmark/README.md` conventions. Record into `benchmark/<site>/site.yaml`.

1. **Walls**: per room, clockwise from above, starting at the wall with the entry door. Laser
   from corner to corner at ~1.0 m height, rear of the device flat on the wall; **3 readings**.
   Where furniture blocks the line, measure at a clear height and note it.
2. **Ceiling height**: ≥ 3 points per room (centre + two away from fixtures), laser on the floor
   pointing up. Note bulkheads/soffits separately.
3. **Openings** (doors, doorways, windows): clear opening between finished jambs at mid-height
   (2 readings), plus height; which wall it is on; which room it connects to.
4. **Room diagonals** for any non-rectangular room (lets us check shape, not just lengths).
5. **Damage**: surface, class, width × height (tape).
6. Photograph each measurement sheet → `benchmark/<site>/measurements/`.

## 5. Captures (~2–2.5 h), grouped by purpose

Every capture id starts with its purpose prefix so evidence never mixes. LiDAR walks follow the
current hypotheses: every wall's **top edge** in view at least once, pass within ~2 m of each wall
facing it, sweep the ceiling, doors fully open (e18, e19).

| Purpose | id prefix | Tier | What | Count | Feeds |
|---|---|---|---|---|---|
| **Development** (may be inspected and tuned on) | `dev_` | LiDAR, video, photo | 1 room + 1 doorway, each tier; one Sensor Logger clip (diagnostic only) | 1 each | loaders, stitching chain test, IMU diagnostic |
| **Benchmark** (scored, never tuned on) | `bm_` | LiDAR | whole property, one recording | 1 | all LiDAR gates, drift ablation |
| | `bm_` | video | whole property, stock Camera, plain clip | 1 | video gates |
| | `bm_` | photo | per-room folders, 2–8 stills each, stock Camera | 1 set | photo gates, stitch |
| | `bm_` | photo | same rooms, **overlapping doorway chain** variant (≈ 3 overlapping shots of each door frame: inside A, in the doorway, inside B) | 1 set | tests the e17 stitching hypothesis |
| | `bm_` | LiDAR | furnished damage room alone | 1 | damage output |
| **Repeatability** | `rep_` | LiDAR | whole property again + damage room again (same protocol, different start point) | 2 | repeatability + ceiling-spread gates |
| | `rep_` | photo/video | one room again at each tier | 1 each | interval calibration at thin tiers |
| **Incumbent comparison** | `inc_` | magicplan | 2 rooms incl. the damage room; keep native export | 1 | head-to-head table |
| **Walk-in proxy** | `walk_` | all three | a second person follows only the one-page protocol, cold | 1 each | protocol clarity, cold-run timing |
| **Failure mode** (expected to fail; documents behaviour) | `fail_` | LiDAR | fast walk, no upper-wall sweep | 1 | quality_flags behaviour |
| | `fail_` | photo | mirror-dominated bathroom, 2 photos | 1 | known photo-scale failure (e16: room 7) |

Ground truth (section 4) covers every room in `bm_`, `rep_`, `inc_` and `walk_` captures.

## 6. Files to hand back

```
benchmark/<site>/
  site.yaml                         (filled in from the measurement sheet)
  measurements/*.jpg
  captures/C1a/  ... Stray export folders (unzipped)
  captures/C3a/clip.MOV ...
  captures/C4/sensorlogger.zip
  captures/C5a/<room_id>/*.HEIC ...
  incumbent/magicplan/*.pdf|csv
```

Device facts to note per capture: model, iOS version, app version, lens, fps, lighting.

## 7. Cheap pre-check (before booking the main session)

If a phone is available for even 20 minutes: one 30 s Stray capture, one 20 s Camera clip, one
20 s Sensor Logger recording, three HEIC photos, a magicplan free export. These answer the
**[verify]** items above and are enough for the loaders to be finished before the main session.

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

## 5. Captures (~2–2.5 h)

Each capture gets an id; note start time, operator, lighting. Order matters only for battery.

| # | Tier | What | Repeats |
|---|---|---|---|
| C1 | LiDAR | whole property in one Stray recording, per the protocol | **×2** (repeatability) |
| C2 | LiDAR | the furnished/damage room alone | ×2 |
| C3 | Video | whole property, stock Camera, per protocol | ×2 |
| C4 | Video+IMU | whole property, Sensor Logger video | ×1 (scale experiment) |
| C5 | Photo | per room 2–8 stills (stock Camera), one folder per room, per protocol | ×2 (two different photo sets) |
| C6 | Photo | same, but with the *minimum* allowed (2 photos/room) | ×1 (stress test) |
| C7 | magicplan | 2 rooms (incl. the damage room), export | ×1 |
| C8 | LiDAR | a deliberately bad capture (fast walk, no ceiling sweep) | ×1 (failure-mode evidence) |

Also capture once by a **second person** who has only read the one-page protocol (closest proxy
for the walk-in test). Note anything they found ambiguous.

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

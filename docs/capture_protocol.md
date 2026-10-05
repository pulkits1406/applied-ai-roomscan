# Capture protocol (Route 2: stock apps) — one page

**Not yet field-tested on a device.** Rules marked † are hypotheses awaiting the development capture
(`docs/development_capture_protocol.md`); see the evidence table below the page.

**Install (≈ 3 min):** LiDAR tier only — install **Stray Scanner** (free, App Store) on an iPhone Pro
with LiDAR. Photo and video tiers use the **built-in Camera app** on any iPhone 15 or newer.
In Settings → Camera → Formats choose **High Efficiency**. Nothing else to install.

**Before any capture (every tier):** open every door fully (door against the wall). Lights on.
Wipe the lens. Hold the phone upright (portrait) unless told otherwise.

**LiDAR (Stray Scanner)**
1. Start recording at the entrance. Walk slowly (about half normal walking speed).
2. In each room: stand 1–2 m from the walls and **slowly turn so the top edge of every wall
   (where wall meets ceiling) is on screen at least once**, then tilt up and sweep the ceiling.
3. Walk once around the room 1–2 m from the walls, phone at chest height, pointing at the walls.
4. Finish a room before entering the next. Go through doorways slowly. Visit every room.
5. Stop recording. Expect about 1–2 minutes per room.

**Video (Camera → Video, 1×)**
1. Start at the entrance. Walk through every room at a calm pace.
2. In each room turn on the spot **slowly (a full turn in about 15–20 s)**, standing **at least 1 m
   from the walls**, with floor and the wall–ceiling edge in view.† Never turn quickly close to a wall.
3. Keep the clip under 4 minutes; one clip for the whole property.

**Photos (Camera → Photo)**
1. One room at a time; take **2–8 photos per room**, phone level, no zoom, flash off.
2. Stand near the middle of the room and turn on the spot, **each photo overlapping the previous
   one by about a third**, so that together they go all the way round.† Every photo shows some floor
   and the wall–ceiling edge.
3. If the room is small, stand in the doorway and photograph it from there, then from the far corner
   looking back.
4. For each doorway, take one photo from each side looking through it at the same door frame.†

**Avoid:** fast turns close to walls; pointing at mirrors or large windows for long; dark rooms;
your own feet or fingers in frame. Do not crop, filter or edit photos or videos.

**Hand over the files**
- LiDAR: in Stray Scanner, share each scan (zip) by AirDrop/USB; one scan per property.
- Photos/video: AirDrop from Photos with **Options → "All Photos Data" ON** (keeps original HEIC/MOV
  and the photo's camera data), or USB with "Keep Originals". Never via WhatsApp/Messages/email.
- Folder layout: `property/lidar/<stray scan>/`, `property/video.MOV`,
  `property/photos/<room name>/*.HEIC` (one folder per room, folder name = room name).

**Run:** `uv run roomscan-inspect <file or folder>` (checks the files), then one command per capture:
`uv run roomscan-lidar <scan> out/`, `uv run roomscan-video <clip> out/`, `uv run roomscan-photo <photos> out/`.

---

## Evidence status of each rule (for the assessor; not part of the operator page)

| Rule | Status | Evidence |
|---|---|---|
| Every wall's top edge in view, ceiling swept (LiDAR) | supported on supplied data | captures that never looked above ~1.7 m give no rooms (e10, e24 `lidar_floor_only`/`single_room`); wall existence needs structural evidence above 2 m (e18) |
| 1–2 m from walls, near-perpendicular views (LiDAR) | supported (GT-free) | close-range views give more repeatable wall positions (e18 V5/V6) |
| One continuous visit per room (LiDAR) | supported (GT-free) | inter-visit disagreement 30–160 mm before re-registration (e22) |
| Doors fully open | supported (indirect) | doorways are found from the lintel + empty mid-height band (e10, e14); a half-open leaf fills the jamb gap (e19) |
| Slow turns ≥ 1 m from walls (video) † | hypothesis | all tracking failures on the sample clip happen in fast turns close to surfaces (e20); not yet tested on a protocol-following clip |
| Overlapping photos turning on the spot † | hypothesis (simulation only) | joint photo registration errors fall from 44–66° to 2–7° with overlap (e23, simulated from video frames); not tested on real photos |
| 0.5× ultra-wide photos | **not required**; A/B test in the development capture | wider view could make overlap and full coverage compatible within 8 photos (e23); unverified |
| Doorway photo chain † | hypothesis | single doorway photos failed to link rooms (e17, 0/36); the overlapping chain is untested |
| Floor and ceiling edge in every photo | supported (simulation) | photo sets showing no floor cannot be segmented (e23 `no_floor`) |
| Reference object (A4, card) | **not used** | not required by the contract; photo scale comes from two model families (e21) |
| Sensor Logger / IMU | **not part of the protocol** | video tier = plain clip; IMU studied as a diagnostic only (e11) |
| Mirrors, glass, wet-look surfaces, low light | flagged risk | a mirror-dominated bathroom breaks MapAnything's photo scale (+31…+44 %, e16); glass produces LiDAR artefacts (e00); no real-device test |
| File hand-off (HEIC/MOV originals, Stray zip) | partly verified | HEIC/MOV/Stray loaders tested with synthetic files (`tests/test_inputs.py`); real iPhone files **BLOCKED** until a device is available |

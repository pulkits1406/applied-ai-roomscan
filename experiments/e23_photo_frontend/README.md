# e23 — photo tier as a product frontend: room geometry from photo folders [PSEUDO-GT]

Production code: `src/roomscan/photo/` (`roomscan-photo <folder_of_room_folders> <out>`), shared
`src/roomscan/cloudroom.py` (segmentation / walls / levels reused from the LiDAR tier) and
`src/roomscan/assemble.py`. The experiment scripts here call the production modules; there is no
experiment copy of the pipeline. Pseudo-GT = LiDAR v1.1 rooms of the same capture.

## Inputs (simulated; `make_photo_folders.py`)
One folder per room (`r<label>` = production LiDAR segmentation label of the camera), JPEG stills
from the e07 walkthrough cache with **only** the EXIF an iPhone writes (integer 35 mm-equivalent
focal; round trip 804 px vs true 800 px). Two selections:
- `diverse` K=6: view-direction-diverse, ~60° apart, little overlap (a user photographing each
  wall); the photo-like filter needs median LiDAR depth > 1.6 m (hidden info, selection only).
- `sweep` K≤8: consecutive frames of one visit ≥ 22° apart (~50 % overlap at 48° portrait FOV).

## Results
**Model inference through the production path works**: MoGe-2 in process, MapAnything in the
rebuilt isolated env via `roomscan.envs` (load 19 s, 6 views 6.6 s on M4).

**MapAnything's joint poses decide whether a room can be built** (`pose_check.py`, vs ARKit):

| set | rel. rotation error median (rooms 1/3/4/6/7/8) | camera-centre Sim3 RMSE vs spread |
|---|---|---|
| diverse K6 | 66.5 / 43.6 / 63.5 / 4.6 / 2.4 / 2.6° (max 110–180°) | ≈ spread (uninformative) |
| sweep K≤8 | 5.1 / 3.8 / 2.1 / 7.1 / 89.3 / 2.3° (r2 92°) | 0.06–0.37 m vs 0.35–0.87 m |

With low-overlap photos the joint reconstruction is wrong (camera heights in one room 1.4–2.6 m;
the floor smears; segmentation fails or merges neighbouring rooms: diverse-set areas r4 54 m² vs
0.9 m², r6 17 m² vs 1.5 m²). With overlapping sweeps the clouds are coherent (camera heights
1.3–1.7 m) in r1/r3/r4/r6. r3's area matched (8.86 vs 8.93 m²) **but by coincidence**: its
walls are 5.73 × 1.58 m vs 3.0 × 3.0 m (e24 wall-level scoring). r2 and the mirror bathroom r7
still fail registration. Coherent cloud ≠ correct room: wall placement inside the sweep's partial
view is still wrong.

**Single photos cannot measure a room** (`per_photo.py`): 26/36 photos give no region; a portrait
main-camera photo (48° HFOV) rarely shows two opposite walls or both ends of a wall.

**Floor visibility:** r8's sweep photos show almost only ceiling (3.08 m room, camera tilted up):
no floor → no room (`no_floor`), reported as such.

**Scale** is unchanged from e21 (ensemble √(MoGe × MA·C_MA), C_MA = 1.455, fixed ±14.3 % at
90 %): photo-tier scale is not the binding problem; registration within a room is.

## Interpretation
1. [MEASURED] Joint feed-forward reconstruction needs overlap: rotation errors fall from 44–66° to
   2–7° when consecutive photos overlap (5 of 7 rooms).
2. [HYPOTHESIS, needs device] 2–8 portrait main-camera photos cannot both cover 360° and overlap
   (needs ≥ ~10 at 48°). Candidate protocol: 0.5× ultra-wide, landscape, from the room centre,
   5–6 photos turning ~60–70° with visible overlap, floor and ceiling edge in every photo; plus
   the overlapping doorway chain for stitching (e17). Not testable on the supplied video frames
   (main camera only).
3. Small rooms (lobby r4, WC r6) cannot be simulated faithfully from this walkthrough: frames
   whose camera is inside them look out of them.

## Decision [DECISION-PROV]
Photo frontend = MoGe-2 per-photo geometry placed by MapAnything joint poses (`--mode moge`;
`--mode ma` uses MapAnything's own depth: incoherent more often) + ensemble scale + shared
room step, rooms unplaced, provisional uncalibrated σ (scale 8.7 % shared, faces 3 cm / 15 cm
inferred). It produces schema-valid plans with honest flags; its accuracy depends on overlap in
the capture, which the protocol must enforce.

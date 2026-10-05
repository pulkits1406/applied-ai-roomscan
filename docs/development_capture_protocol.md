# Development capture protocol (first borrowed-iPhone session, ~30 minutes)

**This is development data (`dev_`), not the benchmark.** It answers three technical questions about
the capture protocol and the pipelines. It is never scored as benchmark evidence, never used for
calibration unless named explicitly, and never merged into a `bm_` / `rep_` / `inc_` / `walk_` site
(`src/roomscan/bench/leakage.py` enforces this). The 4–5 h laser benchmark is a separate, later
session (`docs/physical_benchmark_procedure.md`).

| Question | What is compared | Pre-registered reading of the result (written before the data exists) |
|---|---|---|
| **A — LiDAR** | protocol capture ×2 vs natural capture | *Supported* if both protocol captures find room A with the sketched number of walls and no `visits_reregistered` / `topology_unstable` / `walls_inferred` flag, the taped walls are within 2 cm (median), and the two protocol captures agree within max(1 cm, 0.5 %) per wall; *and* the natural capture is worse on at least one of these. |
| **B — Video** | slow-turn clip vs natural clip, same route | *Supported* if the slow clip has ≥ 50 % coherent windows and at least twice the natural clip's fraction, and the room-A windows give taped walls within 10 % with a consistent scale ratio. A failure of the slow clip is an equally useful answer. |
| **C — Photo** | 1× spread set vs 0.5× overlapping set (same room), plus the doorway chain | The evaluator splits each room's error into a **scale ratio** (one common factor) and a **shape error** (after removing it). *Overlap is the main issue* if the 0.5× set's shape error is < 5 % while the 1× set's is not. *Scale is the main issue* if shape is good but the scale ratio is off by > 8 %. *Model/content is the issue* if both sets have poor shape although registration is consistent (camera-height spread < 0.1 m). Chain: *supported* if room B lands within 0.5 m / 15° of the LiDAR reference. |

Ultra-wide (0.5×) is an **A/B test only**. The official protocol does not change until the real files
work, the geometry improves materially, and the result is not specific to this device.

---

## Before you go (at home, 10 minutes)

- [ ] Borrowed phone: an iPhone **Pro** with LiDAR (12 Pro or newer). Write down: model, iOS version.
- [ ] Install **Stray Scanner** (free, App Store). Open it once; allow camera access.
- [ ] Settings → Camera → Formats → **High Efficiency** (HEIC photos, HEVC video).
- [ ] Settings → Camera → Record Video → note the setting (e.g. 1080p 30 fps). Leave it unchanged.
- [ ] Settings → Camera → **Lens Correction ON** (default). **Grid ON** (helps keep the phone level).
- [ ] Battery > 60 %. Free storage > 5 GB.
- [ ] Bring: 5 m tape (or a laser meter), masking tape, a pen, the measurement sheet below, a laptop.
- [ ] Do **not** install Sensor Logger for this session (diagnostic only; not needed).

## Choose the space (2 minutes)

- [ ] **Room A**: an ordinary room, ideally 3–5 m on a side, with **one doorway** to a second space **B**
      (a corridor or another room).
- [ ] Room A has at least one **awkward case**: a wardrobe, bookcase or sofa against a wall, or an alcove.
- [ ] All doors **fully open** (door leaf against the wall). All lights on. Curtains as they are.

## Tape measurements (8 minutes) — do these first, before any capture

- [ ] Count the walls of room A seen from above (a plain rectangle = 4). Draw a sketch; number the
      walls **clockwise from above, W1 = the wall with the doorway to B**.
- [ ] Measure W1, W2 and W3 **corner to corner at about 1 m height**, twice each.
      (If there is time, also W4 and the rest — every extra wall makes the comparison stronger.)
- [ ] Measure the doorway D1 **clear width between the jambs at 1.0 m height**, twice.
- [ ] Optional: ceiling height at the centre of A, once.
- [ ] Photograph the sketch and the filled sheet.

```
MEASUREMENT SHEET                       date ______   phone model ______   iOS ______
Room A walls (count) ____      tape / laser: ______
W1 (doorway wall)  ______  ______        W2 ______  ______        W3 ______  ______
W4 ______  ______   W5 ______  ______   W6 ______  ______
Doorway D1 clear width at 1.0 m  ______  ______        Ceiling (centre)  ______
Notes (awkward case, mirrors, glass, lighting): ______________________________________
```

## Captures, in this order (≈ 17 minutes)

Write the clock time and any problem next to each step on the sheet.

### A — LiDAR (Stray Scanner), ~7 min

**A1 protocol capture**
- [ ] Start in room B, ~1.5 m from the doorway, phone in portrait. Press record.
- [ ] Walk slowly into room A. Stay roughly **1–2 m from the walls**.
- [ ] Turn slowly on the spot so that **the top edge of every wall** (where wall meets ceiling) is in
      view at least once, then tilt up and **sweep the whole ceiling** once.
- [ ] Walk once around the room 1–2 m from the walls, phone at chest height, pointing at the walls,
      including the awkward case.
- [ ] Do **not** leave room A until it is finished (one continuous visit). Then walk back through the
      doorway into B, show B's walls and top edges briefly, stop. (~2 min)
- [ ] **A2**: repeat A1 exactly (repeatability pair).
- [ ] **A3 natural**: walk the same route at a normal pace, phone at chest height, **no** deliberate
      top-edge or ceiling sweep. (~1 min)

### B — Video (stock Camera, Video mode, 1×), ~4 min

- [ ] **B1 slow-turn protocol**: portrait, start in B. Walk into A. In A, **turn on the spot slowly**
      (a full turn in ~15–20 s), standing **≥ 1 m from the walls**, keep floor and ceiling edge in view.
      Walk once around A at ≥ 1 m from the walls, then back into B. Never turn quickly close to a wall.
      (~1.5 min)
- [ ] **B2 natural**: the same route at a natural pace, turning the way you normally would. (~1 min)

### C — Photos (stock Camera, Photo mode), ~6 min

Keep the phone **level** (grid lines), no zoom other than the lens button stated, flash off.
- [ ] **C1 1× main-camera spread (room A)**: portrait, 1×, 6 photos — stand back in different places
      and photograph each wall/corner once (the "natural" way). Do not try to overlap.
- [ ] **C2 0.5× overlapping (room A)**: tap **0.5**, **landscape**. Stand near the centre of A. Take
      6 photos turning ~60° between photos so that **each photo overlaps the previous one by about a
      third**, each showing floor and the wall–ceiling edge.
- [ ] **C3 doorway chain (rooms A and B)**, 0.5×, landscape:
  - room A: 5 overlapping photos as in C2, then **chain-1**: standing in A ~1 m from the doorway,
    door frame centred; **chain-2**: standing in the doorway, looking into B.
  - room B: **chain-3**: standing in B ~1 m from the doorway, looking back at the same door frame;
    then 5 overlapping photos of B as in C2.
  (≤ 8 photos per room.)
- [ ] Note which photos belong to which set (the Photos app timestamps are enough if you write the
      clock time when each set starts).

## Before leaving (2 minutes)

- [ ] In Stray Scanner, check that A1, A2, A3 are listed and play back.
- [ ] In Photos, check that the videos play and the photo sets are complete.

---

## Export (at the laptop)

- **Stray**: Stray Scanner → each scan → share/export (zip) → AirDrop to the Mac (or Files → USB).
  Keep the folders unmodified.
- **Photos and videos**: AirDrop from Photos with **Options → "All Photos Data" ON** (keeps original
  HEIC + EXIF and original MOV), *or* Image Capture over USB with "Keep Originals". **Never** send via
  WhatsApp/Messages/email (they strip EXIF and re-encode).
- Check on the laptop: photos end in `.HEIC`, videos in `.MOV`.

## Files to bring back

Put them under `benchmark/dev_<YYYYMMDD>/captures/` with these names (matches
`benchmark/_template_dev/site.yaml`):

| What | Path |
|---|---|
| Stray A1, A2, A3 (unzipped folders) | `A_lidar_protocol_1/`, `A_lidar_protocol_2/`, `A_lidar_natural/` |
| Video B1, B2 (original MOV) | `B_video_slow.MOV`, `B_video_natural.MOV` |
| Photos C1 | `C_photo_main_spread/A/*.HEIC` |
| Photos C2 | `C_photo_uw_overlap/A/*.HEIC` |
| Photos C3 | `C_photo_doorway_chain/A/*.HEIC` (5 + chain-1 + chain-2), `C_photo_doorway_chain/B/*.HEIC` (chain-3 + 5) |
| Sheet + sketch photos | `measurements/` |

## Day-one processing (engineer)

```
for f in benchmark/dev_<date>/captures/*; do uv run roomscan-inspect "$f" --json "runs/dev_<date>/inspect_$(basename $f).json"; done
cp benchmark/_template_dev/site.yaml benchmark/dev_<date>/site.yaml      # fill tape readings, wall count, date
uv run roomscan-bench benchmark/dev_<date> --run runs/dev_<date> --execute
```
`roomscan-inspect` re-checks every format assumption on the real files: Stray pose convention and
video offset, HEIC orientation and 35 mm focal (and which lens), MOV rotation, codec and bit depth.
Anything reported as `CHECK` is resolved before reading any result.

## What only the device can tell us ([BLOCKED] until the files exist)

- Stray Scanner's current export format (columns, depth size, video offset) — checked by `roomscan-inspect`.
- Whether iPhone HEIC orientation is applied exactly once (container `irot` + EXIF) — the inspect report
  shows the upright size and the orientation tag.
- What 35 mm-equivalent focal the 0.5× lens writes, and whether Lens Correction leaves residual distortion.
- Whether stock-Camera MOV files carry a focal-length tag (the pipeline estimates the field of view anyway).
- 10-bit HDR video decoding (frames are converted to 8-bit; geometry should be unaffected).

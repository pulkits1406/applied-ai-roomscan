# e27 — doorway-chain placement test, prepared for the development capture (question C)

`chain_eval.py <photo_dir with A/ B/> <LiDAR plan.json> <A id> <B id> [out.json]`: all photos of
both room folders (incl. chain photos) reconstructed jointly with the production photo modules;
each room taken from its own cameras; the photo frame fitted to the LiDAR reference using room A
only; room B's centroid offset and relative-yaw error reported as reconstructed and with the photo
scale error removed. Pre-registered success: B within 0.5 m and 15° (scale removed). Outcomes
`room_missing` and `B_not_separated` (both folders' cameras fell into one segmented region) are
reported as such.

**Mechanics check:** the reference rooms rotated 0.7 rad, scaled 1.3× and shifted are placed back
with 0.000 m / 0.0° (scale removed) — the evaluation itself is exact.

**Simulated walkthrough frames (r3 + r4 sweep folders, no real chain photos):** `B_not_separated`
(`out/sim_r3_r4.json`): the joint reconstruction merges the two spaces. Expected — these frames are
not a doorway chain, and r4's frames look out of r4 (e23). The real test needs the dev capture's
C3 set.

Not production: photo-folder stitching stays unsolved until this test passes on real photos.

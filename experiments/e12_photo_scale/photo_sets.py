"""Per-room candidate 'photos' from the with_ceiling walkthrough (pseudo-GT photo tier).

A frame is photo-like if the camera is near level (|pitch| < 25 deg), sees a wide view
(median LiDAR depth > 1.6 m), is sharp (top 60 % Laplacian variance within the capture)
and rotates slowly (< 40 deg/s). Room membership = camera position inside a segmented room
(e10 v2b labels, barrier height 1.9 m), dilated 0.3 m. Rooms with < 6 candidates are dropped.
Output: out/photo_sets.json {room_id: [rows...]} consumed by e12 and e13.
"""
import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e12_photo_scale/out"; OUT.mkdir(parents=True, exist_ok=True)
CAP = "single_scan_with_ceiling"
meta = json.loads((ROOT / f"cache/frames/{CAP}/frames.json").read_text())["frames"]
grid = json.loads((ROOT / f"experiments/e10_room_seg/out/{CAP}_v2_grid.json").read_text())
lab = np.load(ROOT / f"experiments/e10_room_seg/out/{CAP}_v2b_h1.9_labels.npy")
lab_d = ndi.grey_dilation(lab, size=(7, 7))
sharp_thr = np.percentile([m["sharpness"] for m in meta], 40)
sets = {}
for m in meta:
    if abs(m["pitch_deg"]) > 25 or m["median_depth_m"] < 1.6 or m["sharpness"] < sharp_thr or m["ang_speed_dps"] > 40: continue
    T = np.array(m["T_wc"]); i, j = np.floor((T[[0, 2], 3] - np.array(grid["origin_xz"])) / grid["res"]).astype(int)
    if not (0 <= i < lab.shape[0] and 0 <= j < lab.shape[1]): continue
    r = int(lab[i, j] or lab_d[i, j])
    if r: sets.setdefault(r, []).append(m["row"])
sets = {k: v for k, v in sorted(sets.items()) if len(v) >= 6}
(OUT / "photo_sets.json").write_text(json.dumps({"capture": CAP, "label_file": f"experiments/e10_room_seg/out/{CAP}_v2b_h1.9_labels.npy", "rooms": sets}, indent=1))
print({k: len(v) for k, v in sets.items()}, "total", sum(map(len, sets.values())))

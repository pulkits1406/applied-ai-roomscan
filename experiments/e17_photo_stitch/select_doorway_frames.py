"""Pick walkthrough frames that stand in for "doorway photos" (evaluation-side simulation only).

For doorway pair (A, B) from doorways.json: frames whose camera centre (world x,z) is within 0.8 m of
the doorway centre, |pitch| < 25 deg, and whose forward vector (xz) has positive dot product with the
direction doorway -> centroid of B's region (e10 v2b label grid). Picks: see the ranking comment below
(sharpness gate = top 60 % within capture, as photo_sets.py). Selected frames go into A's folder. If none face B,
the reverse direction (in B's folder, looking into A) is reported instead.
Writes out/doorway_frames.json.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
CAP = "single_scan_with_ceiling"
meta = json.loads((ROOT / f"cache/frames/{CAP}/frames.json").read_text())["frames"]
sets = json.loads((ROOT / "experiments/e12_photo_scale/out/photo_sets.json").read_text())
grid = json.loads((ROOT / f"experiments/e10_room_seg/out/{CAP}_v2_grid.json").read_text())
lab = np.load(ROOT / sets["label_file"])
doors = json.loads((HERE / "doorways.json").read_text())["doorways"]
sharp_thr = np.percentile([m["sharpness"] for m in meta], 40)
org, res = np.array(grid["origin_xz"]), grid["res"]


def region_centroid(room: int) -> np.ndarray:
    ij = np.argwhere(lab == room)
    return org + (ij.mean(0) + 0.5) * res


def room_of(xz) -> int:
    i, j = np.floor((np.asarray(xz) - org) / res).astype(int)
    return int(lab[i, j]) if 0 <= i < lab.shape[0] and 0 <= j < lab.shape[1] else 0


def candidates(door_xz, target_room, radius=0.8):
    tgt = region_centroid(target_room); d = tgt - door_xz; d /= np.linalg.norm(d)
    out = []
    for m in meta:
        T = np.array(m["T_wc"]); c = T[[0, 2], 3]
        if np.linalg.norm(c - door_xz) > radius or abs(m["pitch_deg"]) >= 25:
            continue
        fwd = T[:3, 2][[0, 2]]; fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
        cos = float(fwd @ d)
        if cos > 0:
            out.append({"row": m["row"], "cos_to_target": round(cos, 3), "dist_to_door_m": round(float(np.linalg.norm(c - door_xz)), 2),
                        "pitch_deg": m["pitch_deg"], "sharp_ok": bool(m["sharpness"] >= sharp_thr),
                        "ang_speed_dps": m["ang_speed_dps"], "median_depth_m": m["median_depth_m"],
                        "camera_room_label": room_of(c)})
    out.sort(key=lambda x: -x["cos_to_target"])
    return out, tgt


res_all = []
for dw in doors:
    A, B = dw["rooms"]; door = np.array(dw["center_xz"])
    fwdc, tgtB = candidates(door, B)
    revc, tgtA = candidates(door, A)
    # Up to two picks at least 30 frame rows apart, so they are not near-duplicates.
    # Usable = sharp, < 40 deg/s, looking clearly into B (cos > 0.7). Prefer cameras on A's side or in the
    # unlabelled doorway band, then the closest to the doorway centre.
    usable = [c for c in fwdc if c["sharp_ok"] and c["ang_speed_dps"] < 40 and c["cos_to_target"] > 0.7]
    usable.sort(key=lambda c: (c["camera_room_label"] not in (A, 0), c["dist_to_door_m"]))
    picks = []
    for c in usable:
        if all(abs(c["row"] - p["row"]) >= 30 for p in picks):
            picks.append(c)
        if len(picks) == 2:
            break
    res_all.append({"pair": [A, B], "door_xz": door.tolist(), "centroid_A": tgtA.round(2).tolist(), "centroid_B": tgtB.round(2).tolist(),
                    "n_candidates_looking_into_B": len(fwdc), "n_candidates_looking_into_A": len(revc),
                    "picked_into_B (folder A)": picks, "top_into_A (folder B)": revc[:3], "all_into_B": fwdc[:15]})
    print(A, B, "into B:", len(fwdc), "into A:", len(revc), "picks", [(p["row"], p["cos_to_target"], p["dist_to_door_m"], p["camera_room_label"]) for p in picks])
(HERE / "out").mkdir(exist_ok=True)
(HERE / "out/doorway_frames.json").write_text(json.dumps(res_all, indent=1))

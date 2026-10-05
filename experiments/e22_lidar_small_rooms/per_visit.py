"""Per-visit wall-face peaks for a room: is a doubled face the same wall seen from two visits that
disagree in position (drift not removed by ICP), or two physical surfaces?"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from roomscan import geometry as G
from roomscan.lidar import fusion, rooms as RM
from roomscan.stray import StrayCapture

sys.path.insert(0, str(Path(__file__).parent))
from diagnose import peaks_detail  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main(capture="single_scan_with_ceiling", rooms="2,6"):
    cap = StrayCapture.load(ROOT / capture)
    c = RM.prepare(cap)
    from scipy import ndimage as ndi
    out = {}
    for r in map(int, rooms.split(",")):
        near = ndi.binary_dilation(c.seg.labels == r, iterations=10)
        allz = np.load(ROOT / "cache/e22" / capture / f"room{r}.npz")
        yaw = G.manhattan_yaw(allz["N"]); Rm = G._rot(yaw)
        rows = []
        for a, b in RM.visits(c, r):
            pc = RM._crop(fusion.fuse(cap, c.T, range(a, b + 1, RM.ROOM_STEP), fusion.ALL), near, c.seg)
            P, N = np.asarray(pc.points), np.asarray(pc.normals)
            UV, NUV, H = P[:, [0, 2]] @ Rm.T, N[:, [0, 2]] @ Rm.T, P[:, 1] - c.seg.grid.floor_y
            rec = {"rows": [int(a), int(b)], "t_s": [round(float(cap.timestamps[a] - cap.timestamps[0]), 1), round(float(cap.timestamps[b] - cap.timestamps[0]), 1)],
                   "n": int(len(P)), "max_h": round(float(np.percentile(H, 99)), 2) if len(H) else None}
            for axis, nm in ((0, "u"), (1, "v")):
                for sign in (1, -1):
                    rec[f"{nm}{'+' if sign > 0 else '-'}"] = [(p["coord"], p["n"], p["extent"], p["top"]) for p in peaks_detail(UV, NUV, H, axis, sign) if p["n"] > 300]
            rows.append(rec)
            print(r, rec)
        out[r] = rows
    (ROOT / "experiments/e22_lidar_small_rooms/out/per_visit.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])

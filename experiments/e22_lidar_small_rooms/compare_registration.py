"""Per room and per visit: how each registration method moves the visit, and a GT-free check of the
result. Check = face agreement: for every strong wall-face peak of the reference visit (top >= 2 m),
the offset of the nearest same-side peak of the moved visit (|d| <= 0.3 m). A correct alignment
puts shared faces within ~1-2 cm; misregistration shows as consistent offsets.

Output: out/registration_<capture>.json"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy import ndimage as ndi

from roomscan import geometry as G
from roomscan.lidar import fusion, register, rooms as RM
from roomscan.stray import StrayCapture

sys.path.insert(0, str(Path(__file__).parent))
from diagnose import peaks_detail  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def faces(pc, Rm, floor_y, min_n=300):
    P, N = np.asarray(pc.points), np.asarray(pc.normals)
    UV, NUV, H = P[:, [0, 2]] @ Rm.T, N[:, [0, 2]] @ Rm.T, P[:, 1] - floor_y
    return {(axis, sign): [p for p in peaks_detail(UV, NUV, H, axis, sign) if p["n"] >= min_n] for axis in (0, 1) for sign in (1, -1)}


def agreement(ref_f, mov_f):
    """Offsets (moving - ref) between each ref face and the nearest same-side moving face."""
    out = []
    for k, rf in ref_f.items():
        for p in rf:
            cand = [q["coord"] - p["coord"] for q in mov_f.get(k, []) if abs(q["coord"] - p["coord"]) <= 0.3]
            if cand:
                out.append({"axis": k[0], "sign": k[1], "ref": p["coord"], "d": round(min(cand, key=abs), 3), "ref_top": p["top"]})
    return out


def main(capture="single_scan_with_ceiling"):
    cap = StrayCapture.load(ROOT / capture)
    c = RM.prepare(cap)
    t = cap.timestamps
    res = {}
    for label in range(1, c.seg.labels.max() + 1):
        spans = RM.visits(c, label)
        if len(spans) < 2:
            continue
        near = ndi.binary_dilation(c.seg.labels == label, iterations=10)
        order = sorted(spans, key=lambda ab: t[ab[1]] - t[ab[0]], reverse=True)
        clouds = [RM._crop(fusion.fuse(cap, c.T, range(a, b + 1, RM.ROOM_STEP), fusion.ALL), near, c.seg) for a, b in order]
        ref = clouds[0]
        yaw = G.manhattan_yaw(np.asarray(ref.normals)); Rm = G._rot(yaw)
        rf = faces(ref, Rm, c.seg.grid.floor_y)
        rows = []
        for (a, b), pa in zip(order[1:], clouds[1:]):
            Ti, icp_rec = RM._icp(pa, ref) if len(pa.points) >= 500 else (np.eye(4), {"aligned": False})
            al = register.align(pa, ref, yaw)
            rec = {"rows": [int(a), int(b)], "n": len(pa.points), "icp": icp_rec, "coarse": {"method": al.method, **al.info}}
            for name, T in (("raw", np.eye(4)), ("icp_T", Ti), ("icp_used", Ti if icp_rec.get("aligned") else np.eye(4)), ("coarse_T", al.T)):
                ag = agreement(rf, faces(o3d.geometry.PointCloud(pa).transform(T), Rm, c.seg.grid.floor_y))
                rec[name] = {"n_faces": len(ag), "med_abs_d_mm": round(1000 * float(np.median([abs(x["d"]) for x in ag])), 1) if ag else None,
                             "within_2cm": int(sum(abs(x["d"]) <= 0.02 for x in ag)), "d": [(x["axis"], x["sign"], x["d"]) for x in ag]}
            rows.append(rec)
            print(label, rec["rows"], {k: (rec[k]["n_faces"], rec[k]["med_abs_d_mm"], rec[k]["within_2cm"]) for k in ("raw", "icp_T", "icp_used", "coarse_T")},
                  rec["coarse"].get("shift_uv"), rec["coarse"]["method"], flush=True)
        res[label] = rows
    (ROOT / f"experiments/e22_lidar_small_rooms/out/registration_{capture}.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:])

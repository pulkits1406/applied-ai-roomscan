"""Doorway-chain test for the development capture (question C): can two room folders be placed
relative to each other from photos alone? Photo stitching is not a production feature; this
experiment uses the production photo modules.

  joint      all photos of room folders A and B (incl. the chain photos) reconstructed together
             (MoGe-2 + MapAnything, e21 ensemble scale), each room taken from its own cameras
  reference  a LiDAR plan of the same space (placed rooms), e.g. dev capture A1
  alignment  the photo frame is fitted to the reference with ROOM A ONLY (Manhattan yaw in 90-degree
             steps + centroid; best outline IoU); room B's placement is then compared:
             offset of B's centroid (m) and relative-yaw error; reported both as reconstructed and
             after removing the photo scale error (A's linear size ratio), which isolates placement.
Success (pre-registered, docs/development_capture_protocol.md): B within 0.5 m and 15 degrees.

    uv run python experiments/e27_dev_chain/chain_eval.py <photo_dir with A/ B/> <lidar plan.json> <lidar id of A> <lidar id of B> [out.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from roomscan.cloudroom import room_from_cloud
from roomscan.photo import inference, reconstruct as RC
from roomscan.photo.images import load_photo, room_folders


def _yaw(poly):
    d = np.diff(np.vstack([poly, poly[:1]]), axis=0)
    a = np.mod(np.arctan2(d[:, 1], d[:, 0]), np.pi / 2)
    w = np.linalg.norm(d, axis=1)
    return float(np.arctan2((w * np.sin(4 * a)).sum(), (w * np.cos(4 * a)).sum()) / 4)


def _rot(P, a, c):
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (P - c) @ R.T + c


def joint_rooms(photo_dir, cache):
    f = room_folders(photo_dir)
    names = sorted(f)
    photos = [load_photo(p) for n in names for p in f[n]]
    owner = np.array([n for n in names for _ in f[n]])
    mo = inference.moge(photos, cache / "joint")
    ma = inference.mapanything(photos, cache / "joint")
    pc = RC.room_cloud(mo, ma, "moge")
    out = {}
    for n in names:
        g = room_from_cloud(pc.P, pc.N, pc.cams[owner == n])
        out[n] = None if isinstance(g, str) else np.stack([(g.walls.poly_uv @ G_rot(g.walls.yaw))[:, 0], -(g.walls.poly_uv @ G_rot(g.walls.yaw))[:, 1]], 1)
    return out, {"scale_ma_to_m": pc.scale_ma_to_m, "n_photos": len(photos)}


def G_rot(yaw):
    from roomscan import geometry as G
    return G._rot(yaw)


def _place(pA, pB, ref_A, a, f):
    """Similarity (rotation a about A's centroid, scale f, translation) fitted on room A only."""
    cA = pA.mean(0)
    A2, B2 = (_rot(pA, a, cA) - cA) * f, (_rot(pB, a, cA) - cA) * f
    t = ref_A.mean(0) - A2.mean(0)
    return A2 + t, B2 + t


def evaluate(photo_rooms, ref_A, ref_B):
    pA, pB = photo_rooms.get("A"), photo_rooms.get("B")
    if pA is None or pB is None:
        return {"result": "room_missing", "A": pA is not None, "B": pB is not None}
    sep = Polygon(pA).buffer(0).intersection(Polygon(pB).buffer(0)).area / min(Polygon(pA).area, Polygon(pB).area)
    if sep > 0.8:     # both folders' cameras fell into one segmented region: B is not reconstructed as its own room
        return {"result": "B_not_separated", "outline_overlap_frac": round(float(sep), 3), "success": False}
    s = float(np.sqrt(abs(Polygon(ref_A).area) / abs(Polygon(pA).area)))      # photo -> reference linear scale (from A)
    out = {"photo_to_ref_scale": round(s, 3), "outline_overlap_frac": round(float(sep), 3)}
    for key, f in (("as_reconstructed", 1.0), ("scale_removed", s)):
        cands = []
        for k in range(4):
            a = _yaw(ref_A) - _yaw(pA) + k * np.pi / 2
            A2, B2 = _place(pA, pB, ref_A, a, f)
            iou = Polygon(A2).buffer(0).intersection(Polygon(ref_A)).area / Polygon(A2).buffer(0).union(Polygon(ref_A)).area
            dyaw = np.degrees(np.mod(_yaw(B2) - _yaw(ref_B) + np.pi / 4, np.pi / 2) - np.pi / 4)
            cands.append({"k90": k, "A_iou": round(float(iou), 3), "B_offset_m": round(float(np.linalg.norm(B2.mean(0) - ref_B.mean(0))), 3),
                          "B_rel_yaw_err_deg": round(float(dyaw), 1)})
        out[key] = max(cands, key=lambda c: c["A_iou"])          # orientation chosen from room A alone
    sr = out["scale_removed"]
    out["success"] = bool(sr["B_offset_m"] <= 0.5 and abs(sr["B_rel_yaw_err_deg"]) <= 15)
    out["note"] = "success judged after removing the photo scale error (placement only); as_reconstructed includes it"
    return out


def main(photo_dir, lidar_plan, id_a, id_b, out=None):
    plan = json.loads(Path(lidar_plan).read_text())
    rooms = {r["id"]: np.array(r["polygon"]) for r in plan["rooms"]}
    cache = Path(out).parent / "cache" if out else Path("runs/e27/cache")
    pr, info = joint_rooms(photo_dir, cache)
    res = {**evaluate(pr, rooms[id_a], rooms[id_b]), **info, "photo_dir": str(photo_dir), "reference": str(lidar_plan)}
    print(json.dumps(res, indent=1))
    if out:
        Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])

"""Can rooms be measured inside tracked video pieces? [PSEUDO-GT = LiDAR v1.1 rooms]

Input: e20 tracked pieces (cache/e20/poses/<capture>_<method>.json; default pnp_s2_cut = PnP on
MoGe depth, cut at blind links) and the e20 MoGe depth cache. NOTE: e20 used the TRUE ARKit
intrinsics and true-FOV MoGe depth, so this is an optimistic upper bound for an RGB-only clip.
Per piece: fuse MoGe depth with the piece's poses -> gravity from the cameras' image-up axes ->
cloudroom.rooms_from_cloud (shared segmentation/walls/levels). Each piece room is matched to the
LiDAR room its cameras were in (ARKit poses used for evaluation only).

    uv run python experiments/e25_video_rooms/rooms_from_pieces.py [capture] [method] [min_piece_s]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from roomscan import geometry as G
from roomscan.cloudroom import gravity_rotation, point_normals, rooms_from_cloud, voxel
from roomscan.lidar import rooms as RM
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e20_video_tracking"))
from common import load_frames, load_poses, moge_path  # noqa: E402

R_LAND_UP = np.array([[0, 1.0, 0], [-1, 0, 0], [0, 0, 1]])   # upright camera axes in landscape camera coords
DS = 0.5


def lidar_rooms(plan="runs/e22/ladder/plan.json"):
    p = json.loads((ROOT / plan).read_text())
    S = {s["id"]: s for s in p["surfaces"]}
    return {r["id"]: {"n_walls": len(r["wall_ids"]), "area": round(r["floor_area"]["value"], 2), "ceiling": r["ceiling_height"]["value"],
                      "lengths": sorted(round(S[w]["length"]["value"], 3) for w in r["wall_ids"])} for r in p["rooms"]}


def main(capture="single_scan_with_ceiling", method="pnp_s2_cut", min_piece_s="10"):
    frames = {f["row"]: f for f in load_frames(capture)}
    poses, piece_of, _ = load_poses(ROOT / f"cache/e20/poses/{capture}_{method}.json")
    K = np.median(np.array([f["K"] for f in frames.values()]), axis=0); Kd = K.copy(); Kd[:2] *= DS
    c = RM.prepare(StrayCapture.load(ROOT / capture))
    lid = lidar_rooms()
    pieces = {}
    for r, pid in piece_of.items():
        pieces.setdefault(pid, []).append(r)
    rep = []
    for pid, rows in sorted(pieces.items()):
        rows = sorted(rows)
        dur = frames[rows[-1]]["t"] - frames[rows[0]]["t"]
        if dur < float(min_piece_s):
            continue
        P, N, cams, camR = [], [], [], []
        for r in rows:
            mp = moge_path(capture, r)
            if mp is None:
                continue
            d = np.load(mp).astype(np.float32)
            H, W = d.shape
            v, u = np.mgrid[0:H, 0:W]
            X = np.stack([(u - Kd[0, 2]) * d / Kd[0, 0], (v - Kd[1, 2]) * d / Kd[1, 1], d], -1)
            p, n = point_normals(X[::3, ::3], (np.isfinite(d) & (d < 4.5))[::3, ::3])
            T = poses[r]
            P.append(p @ T[:3, :3].T + T[:3, 3]); N.append(n @ T[:3, :3].T); cams.append(T[:3, 3]); camR.append(T[:3, :3] @ R_LAND_UP)
        P, N, cams, camR = np.concatenate(P), np.concatenate(N), np.array(cams), np.array(camR)
        Rg = gravity_rotation(camR, N)
        P, N = voxel(P @ Rg.T, N @ Rg.T)
        rooms, labs = rooms_from_cloud(P, N, cams @ Rg.T)
        true_lab = np.array([c.room_of_frame[r] for r in rows if moge_path(capture, r) is not None])
        for lab, g in rooms.items():
            tl = true_lab[labs == lab]
            tl = tl[tl > 0]
            truth = f"r{int(np.bincount(tl).argmax())}" if len(tl) else None
            row = {"piece": pid, "piece_s": round(dur, 1), "region": lab, "truth_room": truth, "lidar": lid.get(truth)}
            if isinstance(g, str):
                row["result"] = g
            else:
                uv = g.walls.poly_uv
                row.update({"result": "ok", "step": g.step, "n_walls": len(uv), "sources": g.walls.source,
                            "area": round(abs(G.polygon_area(uv)), 2), "ceiling": None if g.ceil_h is None else round(g.ceil_h, 3),
                            "lengths": sorted(np.round(np.linalg.norm(np.roll(uv, -1, 0) - uv, axis=1), 3).tolist())})
            rep.append(row)
            print({k: row.get(k) for k in ("piece", "piece_s", "truth_room", "result", "n_walls", "area", "ceiling", "lengths")},
                  "| LiDAR", None if not row["lidar"] else {k: row["lidar"][k] for k in ("n_walls", "area", "ceiling", "lengths")}, flush=True)
    (ROOT / f"experiments/e25_video_rooms/out/rooms_{capture}_{method}.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:])

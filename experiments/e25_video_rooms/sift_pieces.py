"""Rooms inside COLMAP-SIFT pieces (e08 poses, arbitrary scale per piece) [PSEUDO-GT = LiDAR v1.1].

Scale per piece without hidden information: fuse each frame's MoGe depth with the piece's poses,
translations multiplied by s, for s on a log grid; the multi-view-consistent s minimises the number
of occupied 3 cm voxels of vertical-surface points (wrong s smears every wall). MoGe depth itself
is metric, so s also fixes the absolute scale up to MoGe's bias. Same caveat as rooms_from_pieces:
e08/e20 MoGe depth was computed with the true field of view.

    uv run python experiments/e25_video_rooms/sift_pieces.py [capture] [min_piece_s]
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
sys.path.insert(0, str(ROOT / "experiments/e25_video_rooms"))
from common import load_frames, moge_path  # noqa: E402
from rooms_from_pieces import DS, R_LAND_UP, lidar_rooms  # noqa: E402

GRID = np.exp(np.linspace(np.log(0.2), np.log(5.0), 61))


def frame_points(d, Kd, step=4):
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    X = np.stack([(u - Kd[0, 2]) * d / Kd[0, 0], (v - Kd[1, 2]) * d / Kd[1, 1], d], -1)
    return point_normals(X[::step, ::step], (np.isfinite(d) & (d < 4.5))[::step, ::step])


def occupancy(per_frame, poses, s, vox=0.03):
    keys = []
    for (p, n), T in zip(per_frame, poses):
        q = p @ T[:3, :3].T + s * T[:3, 3]
        nv = n @ T[:3, :3].T
        keys.append(np.floor(q[np.abs(nv @ np.array([0, 1.0, 0])) < 0.5] / vox).astype(np.int64))  # loose: gravity not known yet
    k = np.concatenate(keys)
    return len(np.unique(k[:, 0] * 73856093 ^ k[:, 1] * 19349663 ^ k[:, 2] * 83492791))


def main(capture="single_scan_with_ceiling", min_piece_s="6"):
    frames = {f["row"]: f for f in load_frames(capture)}
    items = json.loads((ROOT / f"experiments/e08_video_sfm/out/{capture}_sfm_poses.json").read_text())
    K = np.median(np.array([f["K"] for f in frames.values()]), axis=0); Kd = K.copy(); Kd[:2] *= DS
    c = RM.prepare(StrayCapture.load(ROOT / capture))
    lid = lidar_rooms()
    pieces = {}
    for it in items:
        if moge_path(capture, it["row"]) is not None:
            pieces.setdefault(it["model"], []).append((it["row"], np.array(it["T_wc"])))
    rep = []
    for pid, lst in sorted(pieces.items()):
        lst.sort(key=lambda x: x[0])
        rows = [r for r, _ in lst]
        dur = frames[rows[-1]]["t"] - frames[rows[0]]["t"]
        if dur < float(min_piece_s) or len(rows) < 10:
            continue
        per = [frame_points(np.load(moge_path(capture, r)).astype(np.float32), Kd) for r in rows]
        Ts = [T for _, T in lst]
        occ = np.array([occupancy(per, Ts, s) for s in GRID])
        s = float(GRID[np.argmin(occ)])
        # truth scale of this piece (evaluation only): ARKit baselines / SfM baselines
        cg = np.array([np.array(frames[r]["T_wc"])[:3, 3] for r in rows]); cp = np.array([T[:3, 3] for T in Ts])
        iu = np.triu_indices(len(rows), 1)
        bg, bp = np.linalg.norm(cg[:, None] - cg[None], axis=-1)[iu], np.linalg.norm(cp[:, None] - cp[None], axis=-1)[iu]
        keep = bg > 0.2
        s_true = float(np.median(bg[keep] / bp[keep])) if keep.any() else None
        P = np.concatenate([p @ T[:3, :3].T + s * T[:3, 3] for (p, _), T in zip(per, Ts)])
        N = np.concatenate([n @ T[:3, :3].T for (_, n), T in zip(per, Ts)])
        camR = np.array([T[:3, :3] @ R_LAND_UP for T in Ts]); cams = np.array([s * T[:3, 3] for T in Ts])
        Rg = gravity_rotation(camR, N)
        P, N = voxel(P @ Rg.T, N @ Rg.T)
        rooms, labs = rooms_from_cloud(P, N, cams @ Rg.T, min_cams=5)
        true_lab = np.array([c.room_of_frame[r] for r in rows])
        base = {"piece": int(pid), "piece_s": round(dur, 1), "n_frames": len(rows), "scale": round(s, 4),
                "scale_err_pct": None if s_true is None else round(100 * (s / s_true - 1), 1)}
        if not rooms:
            rep.append({**base, "result": "no_room"}); print(base, "no room"); continue
        for lab, g in rooms.items():
            tl = true_lab[labs == lab]; tl = tl[tl > 0]
            truth = f"r{int(np.bincount(tl).argmax())}" if len(tl) else None
            row = {**base, "truth_room": truth}
            if isinstance(g, str):
                row["result"] = g
            else:
                uv = g.walls.poly_uv
                row.update({"result": "ok", "n_walls": len(uv), "sources": g.walls.source, "area": round(abs(G.polygon_area(uv)), 2),
                            "ceiling": None if g.ceil_h is None else round(g.ceil_h, 3),
                            "lengths": sorted(np.round(np.linalg.norm(np.roll(uv, -1, 0) - uv, axis=1), 3).tolist())})
            rep.append(row)
            L = lid.get(truth)
            print({k: row.get(k) for k in ("piece", "piece_s", "scale_err_pct", "truth_room", "result", "n_walls", "area", "ceiling", "lengths")},
                  "| LiDAR", None if not L else {k: L[k] for k in ("n_walls", "area", "lengths")}, flush=True)
    (ROOT / f"experiments/e25_video_rooms/out/sift_rooms_{capture}.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:])

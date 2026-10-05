"""How good are MapAnything's joint camera poses for a room's photos? [PSEUDO-GT = ARKit]
Relative rotation error per photo pair; camera-centre Sim(3) fit RMSE; per-photo camera height
above the floor from each photo's own MoGe floor plane (no cross-photo registration)."""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from roomscan.photo import inference
from roomscan.photo.images import load_photo, room_folders

ROOT = Path(__file__).resolve().parents[2]
R_LAND_UP = np.array([[0, 1.0, 0], [-1, 0, 0], [0, 0, 1]])     # upright camera axes in landscape camera coords


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, S, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s) / len(src))
    D = np.eye(3); D[2, 2] = -1 if np.linalg.det(U) * np.linalg.det(Vt) < 0 else 1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / ((src - mu_s) ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def floor_height(pts, rng):
    P = pts[np.isfinite(pts).all(-1)]
    P = P[P[:, 1] > np.percentile(P[:, 1], 50)]             # lower half of the scene (camera y points down)
    best = None
    for _ in range(400):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n /= nn
        if n[1] < 0: n = -n
        if n[1] < np.cos(np.radians(40)): continue
        d = -n @ s[0]; inl = (np.abs(P @ n + d) < 0.03).sum()
        if best is None or inl > best[0]: best = (inl, abs(d))
    return None if best is None or best[0] < 0.03 * len(P) else float(best[1])


def main(folder="cache/photo_sim/single_scan_with_ceiling_K6", cache="runs/e23/cache"):
    meta = json.loads((ROOT / folder / "_sim_meta.json").read_text())["rooms"]
    floor_y = json.loads((ROOT / "runs/e22/ladder/debug.json").read_text())
    rng = np.random.default_rng(0)
    rep = {}
    for name, paths in room_folders(ROOT / folder).items():
        photos = [load_photo(p) for p in paths]
        ma = inference.mapanything(photos, ROOT / cache / name)
        mo = inference.moge(photos, ROOT / cache / name)
        gt = [np.array(r["T_wc"]) for r in meta[name]]
        Rg = [T[:3, :3] @ R_LAND_UP for T in gt]
        Rp = [P[:3, :3] for P in ma["pose"]]
        rot = [np.degrees(np.arccos(np.clip((np.trace((Rp[i].T @ Rp[j]) @ (Rg[i].T @ Rg[j]).T) - 1) / 2, -1, 1)))
               for i, j in combinations(range(len(gt)), 2)]
        cg, cp = np.array([T[:3, 3] for T in gt]), ma["pose"][:, :3, 3]
        s, R, t = umeyama(cp, cg)
        rmse = float(np.sqrt(((cg - (s * cp @ R.T + t)) ** 2).sum(1).mean()))
        spread = float(np.sqrt(((cg - cg.mean(0)) ** 2).sum(1).mean()))
        hp = [floor_height(m["points"], rng) for m in mo]
        rep[name] = {"rel_rot_err_deg_median": round(float(np.median(rot)), 1), "rel_rot_err_deg_max": round(float(np.max(rot)), 1),
                     "centre_sim3_rmse_m": round(rmse, 3), "true_centre_spread_m": round(spread, 3),
                     "true_cam_y": np.round(cg[:, 1], 2).tolist(), "moge_cam_height": [None if h is None else round(h, 2) for h in hp]}
        print(name, rep[name])
    (ROOT / f"experiments/e23_photo_frontend/out/pose_check_{Path(folder).name}.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])

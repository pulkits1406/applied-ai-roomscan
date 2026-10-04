"""Which pose convention and which depth/pose frame offset make depth maps mutually consistent?

For frame pairs (i, j), back-project depth_i to world with pose_i, project into frame j with
pose_j and compare to depth_j. Correct convention + alignment -> residual ~ sensor noise.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from roomscan.stray import StrayCapture, ARKIT_TO_CV

ROOT = Path(__file__).resolve().parents[2]
CONVS = {
    "arkit_c2w": lambda R: R @ ARKIT_TO_CV,
    "cv_c2w": lambda R: R,
    "arkit_w2c": lambda R: (R @ ARKIT_TO_CV).T,  # treat quaternion as world-to-camera
}

def pose(cap, i, conv, offset=0):
    k = int(np.clip(i + offset, 0, len(cap) - 1))
    o = cap.odometry.iloc[k]
    Rq = Rotation.from_quat([o.qx, o.qy, o.qz, o.qw]).as_matrix()
    t = np.array([o.x, o.y, o.z])
    R = CONVS[conv](Rq)
    if conv.endswith("w2c"):
        t = -(R @ t)
    return R, t

def residual(cap, i, j, conv, offset=0):
    d = cap.depth_m(i); c = cap.confidence(i); K = cap.K_depth(i)
    v, u = np.nonzero((c == 2) & (d < 4.0)); z = d[v, u]
    pc = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
    Ri, ti = pose(cap, i, conv, offset); Rj, tj = pose(cap, j, conv, offset)
    pw = pc @ Ri.T + ti
    pj = (pw - tj) @ Rj  # world -> cam j
    Kj = cap.K_depth(j)
    m = pj[:, 2] > 0.1
    uj = Kj[0, 0] * pj[m, 0] / pj[m, 2] + Kj[0, 2]; vj = Kj[1, 1] * pj[m, 1] / pj[m, 2] + Kj[1, 2]
    inb = (uj >= 0) & (uj < 255.5) & (vj >= 0) & (vj < 191.5)
    if inb.sum() < 500: return None, 0.0
    dj = cap.depth_m(j); cj = cap.confidence(j)
    ui, vi = np.round(uj[inb]).astype(int), np.round(vj[inb]).astype(int)
    ok = cj[vi, ui] == 2
    r = np.abs(dj[vi, ui][ok] - pj[m][inb][ok, 2])
    return float(np.median(r)), float(inb.mean())

out = {}
for name in ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]:
    cap = StrayCapture.load(ROOT / name)
    rng = np.random.default_rng(0)
    pairs = [(i, i + g) for g in (10, 30, 60, 120) for i in rng.integers(0, len(cap) - 121, 12)]
    res = {}
    for conv in CONVS:
        for off in ([-3, -2, -1, 0, 1, 2, 3] if conv == "cv_c2w" else [0]):
            rs = [residual(cap, i, j, conv, off) for i, j in pairs]
            vals = [r for r, f in rs if r is not None]
            ov = np.mean([f for r, f in rs])
            res[f"{conv}@{off:+d}"] = {"median_resid_mm": round(1000 * float(np.median(vals)), 2) if vals else None,
                                      "n_pairs": len(vals), "mean_overlap": round(float(ov), 2)}
    out[name] = res
    print(name, json.dumps(res, indent=1))
(ROOT / "experiments/e01_pose_convention/out").mkdir(exist_ok=True, parents=True)
(ROOT / "experiments/e01_pose_convention/out/reprojection.json").write_text(json.dumps(out, indent=1))

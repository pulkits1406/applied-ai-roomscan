"""Video+IMU scale on corrected canonical COLMAP poses (e08 out/<capture>_sfm_poses.json) [PSEUDO-GT].

v1 (imu_on_sfm.py) used per-run files that picked the global mapper, whose positions are wrong
(e08 report). Here: incremental mapper, fixed intrinsics; every model (piece) has its own scale,
so each piece >= MIN_S seconds is evaluated separately against its own Sim(3) scale to ARKit.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from imu_scale import estimate_extrinsic_and_offset
from lever_arm import build, fit
from imu_on_sfm import umeyama_scale
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e11_imu_scale/out"; MIN_S = 5.0
res = {}
for name in ["single_room", "single_scan_with_ceiling"]:
    f = ROOT / f"experiments/e08_video_sfm/out/{name}_sfm_poses.json"
    if not f.exists(): continue
    poses = json.loads(f.read_text()); cap = StrayCapture.load(ROOT / name); Ta = cap.T_wc(); ts = cap.timestamps
    models = {}
    for p in poses: models.setdefault(p["model"], []).append(p)
    for mid, ps in sorted(models.items()):
        ps = sorted(ps, key=lambda p: p["row"]); rows = np.array([p["row"] for p in ps]); t = ts[rows]
        # longest gap-free stretch (gaps > 0.3 s break the trajectory spline)
        br = np.nonzero(np.diff(t) > 0.3)[0]; segs = np.split(np.arange(len(ps)), br + 1); seg = max(segs, key=len)
        if t[seg[-1]] - t[seg[0]] < MIN_S: continue
        Ts = np.array([ps[i]["T_wc"] for i in seg]); rr = rows[seg]; tt = t[seg]
        s_true, ate = umeyama_scale(Ts[:, :3, 3], Ta[rr, :3, 3])
        off, R_ci, gres = estimate_extrinsic_and_offset(tt, Ts[:, :3, :3], cap.imu)
        r = {"n": len(seg), "dur_s": round(float(tt[-1] - tt[0]), 1), "ate_mm": round(1000 * ate, 1), "gyro_resid": round(float(gres), 3)}
        for sg in (0.3, 0.4, 0.6):
            try:
                a, b, m, ma = build(tt, Ts[:, :3, 3], Ts[:, :3, :3], cap.imu, R_ci, off, sg)
                fw, _ = fit(a, b, m, ma); rv, _ = fit(a, b, m, ma, reverse=True)
                r[f"s{sg}_geo_err_pct"] = round(100 * (np.sqrt(abs(fw * rv)) / s_true - 1), 1); r[f"s{sg}_fwd_rev"] = [round(100 * (fw / s_true - 1), 1), round(100 * (rv / s_true - 1), 1)]
            except Exception as e:
                r[f"s{sg}"] = f"failed: {e}"
        res[f"{name}/model{mid}"] = r; print(name, mid, json.dumps(r))
(OUT / "imu_on_sfm_v2.json").write_text(json.dumps(res, indent=1))

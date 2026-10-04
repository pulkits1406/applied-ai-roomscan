"""Video+IMU metric scale on REAL RGB-only SfM poses (COLMAP, from e08) [PSEUDO-GT reference].

True scale of an SfM model = Sim(3) (Umeyama) scale mapping SfM camera centres onto ARKit centres.
IMU estimate = lever-arm model (lever_arm.py) using SfM positions/rotations only; camera-IMU
rotation and time offset estimated from SfM rotations vs gyro. Error = s_imu / s_true - 1.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from imu_scale import estimate_extrinsic_and_offset
from lever_arm import build, fit
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e11_imu_scale/out"
def umeyama_scale(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt; s = np.trace(np.diag(S) @ D) / (Xc ** 2).sum(1).mean()
    resid = Y - (s * X @ R.T + (my - s * R @ mx)); return s, float(np.sqrt((resid ** 2).sum(1).mean()))
if __name__ == "__main__":
    res = {}
    for f in sorted((ROOT / "experiments/e08_video_sfm/out").glob("*_sfm_poses.json")):
        name = "single_scan_with_ceiling" if f.name.startswith("single_scan_with_ceiling") else "single_room"
        poses = json.loads(f.read_text())
        models = {}
        for p in poses: models.setdefault(p.get("model", 0), []).append(p)
        ps = sorted(max(models.values(), key=len), key=lambda p: p["row"])
        cap = StrayCapture.load(ROOT / name); Ta = cap.T_wc()
        rows = np.array([p["row"] for p in ps]); t = cap.timestamps[rows]
        Ts = np.array([p["T_wc"] for p in ps])
        s_true, ate = umeyama_scale(Ts[:, :3, 3], Ta[rows, :3, 3])
        off, R_ci, gres = estimate_extrinsic_and_offset(t, Ts[:, :3, :3], cap.imu)
        r = {"frames": len(ps), "fps": round(len(ps) / (t[-1] - t[0]), 2), "duration_s": round(float(t[-1] - t[0]), 1),
             "sim3_ate_mm": round(1000 * ate, 1), "gyro_fit_resid": round(float(gres), 4)}
        for sg in (0.4, 0.6):
            a, b, m, ma = build(t, Ts[:, :3, 3], Ts[:, :3, :3], cap.imu, R_ci, off, sg)
            fw, _ = fit(a, b, m, ma); rv, _ = fit(a, b, m, ma, reverse=True)
            r[f"sigma{sg}"] = {"fwd_err_pct": round(100 * (fw / s_true - 1), 2), "rev_err_pct": round(100 * (rv / s_true - 1), 2),
                               "geo_mean_err_pct": round(100 * (np.sqrt(fw * rv) / s_true - 1), 2)}
        res[f.name] = r; print(f.name, json.dumps(r))
    (OUT / "imu_on_sfm.json").write_text(json.dumps(res, indent=1))


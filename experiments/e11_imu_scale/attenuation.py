"""Diagnose the low-biased IMU scale: errors-in-variables attenuation vs. filter bandwidth.

Forward OLS (trajectory accel as regressor) is attenuated by noise in p''; reverse OLS
(IMU accel as regressor) is attenuated the other way. If attenuation is the cause, the two
bracket the truth and converge as the low-pass removes the noisy band.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp
sys.path.insert(0, str(Path(__file__).parent))
from imu_scale import estimate_extrinsic_and_offset, G, FS
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e11_imu_scale/out"

def signals(t_kf, p_kf, R_kf, imu, R_ci, off, sig_s):
    ti = imu["timestamp"].to_numpy() + off
    tg = np.arange(max(t_kf[0], ti[0]) + 0.5, min(t_kf[-1], ti[-1]) - 0.5, 1 / FS)
    pos = CubicSpline(t_kf, p_kf)(tg); Rw = Slerp(t_kf, Rotation.from_matrix(R_kf))(tg).as_matrix()
    f = -G * imu[["a_x", "a_y", "a_z"]].to_numpy()
    f_i = np.stack([np.interp(tg, ti, f[:, k]) for k in range(3)], 1)
    M = Rw @ R_ci; rhs = np.einsum("nij,nj->ni", M, f_i); sig = sig_s * FS
    acc = gaussian_filter1d(pos, sig, axis=0, order=2) * FS ** 2
    rhs = gaussian_filter1d(rhs, sig, axis=0); Mf = gaussian_filter1d(M, sig, axis=0)
    sl = slice(int(3 * sig), len(tg) - int(3 * sig))
    return acc[sl], rhs[sl], Mf[sl]

def fit(acc, rhs, Mf, reverse=False):
    n = len(acc); C = np.hstack([-np.tile(np.eye(3), (n, 1)), Mf.reshape(-1, 3)])
    if not reverse:
        x, *_ = np.linalg.lstsq(np.hstack([acc.reshape(-1, 1), C]), rhs.reshape(-1), rcond=None); return x[0]
    x, *_ = np.linalg.lstsq(np.hstack([rhs.reshape(-1, 1), C]), acc.reshape(-1), rcond=None); return 1 / x[0]

res = {}
for name in ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]:
    cap = StrayCapture.load(ROOT / name); T = cap.T_wc(); t = cap.timestamps
    off, R_ci, _ = estimate_extrinsic_and_offset(t, T[:, :3, :3], cap.imu)
    idx = np.arange(0, len(cap), 5)
    r = {}
    for sg in [0.05, 0.1, 0.15, 0.25, 0.4, 0.6]:
        a, b, m = signals(t[idx], T[idx, :3, 3], T[idx, :3, :3], cap.imu, R_ci, off, sg)
        fwd, rev = fit(a, b, m), fit(a, b, m, reverse=True)
        r[f"sigma{sg}"] = {"fwd_err_pct": round(100 * (fwd - 1), 2), "rev_err_pct": round(100 * (rev - 1), 2),
                           "geo_mean_err_pct": round(100 * (np.sqrt(fwd * rev) - 1), 2),
                           "corr": round(float(np.corrcoef(a.reshape(-1), (b - b.mean(0)).reshape(-1))[0, 1]), 3)}
    res[name] = r; print(name, json.dumps(r))
(OUT / "attenuation.json").write_text(json.dumps(res, indent=1))

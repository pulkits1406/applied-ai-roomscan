"""IMU scale with a camera-IMU lever arm term.

ARKit/SfM positions are the camera centre c; the accelerometer is at the IMU origin, offset by
r (IMU frame). c'' = a_imu + R_wi (alpha x r + omega x (omega x r)), linear in r:
    s p'' - g + M b - M A(t) r = M f,   A = [alpha]_x + [omega]_x^2,  M = R_wc R_ci
Fit forward and reverse regressions; if the model is now right they should bracket 1.
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

def skew(v):
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], 1), np.stack([v[:, 2], z, -v[:, 0]], 1), np.stack([-v[:, 1], v[:, 0], z], 1)], 1)

def build(t_kf, p_kf, R_kf, imu, R_ci, off, sig_s, t_range=None):
    ti = imu["timestamp"].to_numpy() + off
    lo, hi = max(t_kf[0], ti[0]) + 0.5, min(t_kf[-1], ti[-1]) - 0.5
    if t_range: lo, hi = max(lo, t_range[0]), min(hi, t_range[1])
    tg = np.arange(lo, hi, 1 / FS)
    pos = CubicSpline(t_kf, p_kf)(tg); Rw = Slerp(t_kf, Rotation.from_matrix(R_kf))(tg).as_matrix()
    f = -G * imu[["a_x", "a_y", "a_z"]].to_numpy(); w = imu[["alpha_x", "alpha_y", "alpha_z"]].to_numpy()
    f_i = np.stack([np.interp(tg, ti, f[:, k]) for k in range(3)], 1)
    w_i = np.stack([np.interp(tg, ti, w[:, k]) for k in range(3)], 1)
    w_i = gaussian_filter1d(w_i, 0.03 * FS, axis=0)
    al_i = np.gradient(w_i, 1 / FS, axis=0)
    A = skew(al_i) + np.einsum("nij,njk->nik", skew(w_i), skew(w_i))
    M = Rw @ R_ci; sig = sig_s * FS
    acc = gaussian_filter1d(pos, sig, axis=0, order=2) * FS ** 2
    rhs = gaussian_filter1d(np.einsum("nij,nj->ni", M, f_i), sig, axis=0)
    Mf = gaussian_filter1d(M, sig, axis=0); MA = gaussian_filter1d(np.einsum("nij,njk->nik", M, A), sig, axis=0)
    sl = slice(int(3 * sig), len(tg) - int(3 * sig))
    return acc[sl], rhs[sl], Mf[sl], MA[sl]

def fit(acc, rhs, Mf, MA, lever=True, reverse=False):
    n = len(acc); cols = [-np.tile(np.eye(3), (n, 1)), Mf.reshape(-1, 3)]
    if lever: cols.append(-MA.reshape(-1, 3))
    C = np.hstack(cols)
    if not reverse:
        x, *_ = np.linalg.lstsq(np.hstack([acc.reshape(-1, 1), C]), rhs.reshape(-1), rcond=None); return x[0], x
    x, *_ = np.linalg.lstsq(np.hstack([rhs.reshape(-1, 1), C]), acc.reshape(-1), rcond=None); return 1 / x[0], x

if __name__ == "__main__":
    res = {}
    for name in ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]:
        cap = StrayCapture.load(ROOT / name); T = cap.T_wc(); t = cap.timestamps
        off, R_ci, _ = estimate_extrinsic_and_offset(t, T[:, :3, :3], cap.imu)
        idx = np.arange(0, len(cap), 5); r = {}
        for sg in [0.15, 0.25, 0.4, 0.6]:
            a, b, m, ma = build(t[idx], T[idx, :3, 3], T[idx, :3, :3], cap.imu, R_ci, off, sg)
            out = {}
            for lever in [False, True]:
                fw, x = fit(a, b, m, ma, lever); rv, _ = fit(a, b, m, ma, lever, True)
                out["lever" if lever else "no_lever"] = {"fwd_err_pct": round(100 * (fw - 1), 2), "rev_err_pct": round(100 * (rv - 1), 2),
                                                         **({"lever_arm_cm": (100 * x[7:10]).round(1).tolist()} if lever else {})}
            r[f"sigma{sg}"] = out
        res[name] = r; print(name, json.dumps(r))
    (OUT / "lever_arm.json").write_text(json.dumps(res, indent=1))

"""Metric scale of an up-to-scale camera trajectory from the accelerometer.

Model (world = trajectory frame, unknown scale s, unknown gravity g, accel bias b in IMU frame):
    s * p''(t) - g + R_wc(t) R_ci b = R_wc(t) R_ci f(t),     f = -9.81 * raw_accel   (iOS reports -specific force in g)
Linear in (s, g, b). The same low-pass filter is applied to both sides, so twice-differentiating
the trajectory does not amplify noise beyond the filter band. R_ci (IMU->camera rotation) and
the IMU/camera clock offset are estimated from gyro vs. trajectory angular velocity, not assumed.

Test design: the ARKit trajectory is divided by a random scale and randomly rotated (so the
estimator cannot exploit known gravity/scale); recovered s * applied_scale should be 1.
Subsampling the trajectory to keyframe rate mimics SfM output.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e11_imu_scale/out"; OUT.mkdir(parents=True, exist_ok=True)
G = 9.81; FS = 100.0  # resample rate (Hz)

def body_rates(t, Rs):
    """Angular velocity in the body frame from a rotation sequence (finite differences)."""
    dR = Rotation.from_matrix(np.einsum("nji,njk->nik", Rs[:-1], Rs[1:]))
    return dR.as_rotvec() / np.diff(t)[:, None], 0.5 * (t[1:] + t[:-1])

def kabsch(A, B):
    """R minimising ||R A_i - B_i||."""
    U, _, Vt = np.linalg.svd(B.T @ A); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt

def estimate_extrinsic_and_offset(t_cam, R_wc, imu):
    w_c, tm = body_rates(t_cam, R_wc)
    ti = imu["timestamp"].to_numpy(); gyr = imu[["alpha_x", "alpha_y", "alpha_z"]].to_numpy()
    best = None
    for off in np.arange(-0.1, 0.1001, 0.005):
        g = np.stack([np.interp(tm, ti + off, gyr[:, k]) for k in range(3)], 1)
        R = kabsch(g, w_c); r = np.linalg.norm(w_c - g @ R.T, axis=1).mean()
        if best is None or r < best[2]: best = (off, R, r)
    return best  # (offset to add to imu t, R_ci, mean residual rad/s)

def estimate_scale(t_kf, p_kf, R_kf, imu, R_ci, off, sigma_s=0.15, t_range=None):
    ti = imu["timestamp"].to_numpy() + off
    lo, hi = max(t_kf[0], ti[0]) + 0.5, min(t_kf[-1], ti[-1]) - 0.5
    if t_range: lo, hi = max(lo, t_range[0]), min(hi, t_range[1])
    tg = np.arange(lo, hi, 1 / FS)
    if len(tg) < 3 * FS: return None
    pos = CubicSpline(t_kf, p_kf)(tg)
    Rw = Slerp(t_kf, Rotation.from_matrix(R_kf))(tg).as_matrix()
    f = -G * imu[["a_x", "a_y", "a_z"]].to_numpy()
    f_i = np.stack([np.interp(tg, ti, f[:, k]) for k in range(3)], 1)
    M = Rw @ R_ci                                   # IMU -> world
    rhs = np.einsum("nij,nj->ni", M, f_i)
    sig = sigma_s * FS
    acc = gaussian_filter1d(pos, sig, axis=0, order=2) * FS ** 2
    rhs = gaussian_filter1d(rhs, sig, axis=0)
    Mf = gaussian_filter1d(M, sig, axis=0)
    trim = int(3 * sig); sl = slice(trim, len(tg) - trim)
    n = len(tg[sl])
    A = np.zeros((3 * n, 7)); A[:, 0] = acc[sl].reshape(-1)
    A[:, 1:4] = -np.tile(np.eye(3), (n, 1))
    A[:, 4:7] = Mf[sl].reshape(-1, 3)
    x, *_ = np.linalg.lstsq(A, rhs[sl].reshape(-1), rcond=None)
    s, g = x[0], x[1:4]
    exc = float(np.sqrt(np.mean(np.sum((acc[sl] * s) ** 2, 1))))  # motion excitation (m/s^2 rms)
    return {"s": float(s), "g_norm": float(np.linalg.norm(g)), "bias": x[4:7].round(3).tolist(), "excitation_rms": exc, "dur": float(hi - lo)}

def main():
  rng = np.random.default_rng(0)
  res = {}
  for name in ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]:
      cap = StrayCapture.load(ROOT / name); T = cap.T_wc(); t = cap.timestamps
      off, R_ci, gres = estimate_extrinsic_and_offset(t, T[:, :3, :3], cap.imu)
      r = {"imu_offset_s": round(float(off), 3), "gyro_fit_resid_rad_s": round(float(gres), 4),
           "R_ci": R_ci.round(3).tolist()}
      for step in [1, 5, 15]:                       # 60 Hz VIO rate, ~9 fps keyframes, ~3 fps keyframes
          idx = np.arange(0, len(cap), step)
          trials = []
          for k in range(5):
              s_applied = float(np.exp(rng.uniform(-1, 1)))           # unknown SfM scale
              Rr = Rotation.random(random_state=k).as_matrix()          # unknown SfM world orientation
              p = (T[idx, :3, 3] @ Rr.T) / s_applied
              Rk = np.einsum("ij,njk->nik", Rr, T[idx, :3, :3])
              e = estimate_scale(t[idx], p, Rk, cap.imu, R_ci, off)
              trials.append(e["s"] / s_applied)
          whole = estimate_scale(t[idx], T[idx, :3, 3], T[idx, :3, :3], cap.imu, R_ci, off)
          win = []
          for a in np.arange(t[0], t[-1] - 10, 10.0):
              e = estimate_scale(t[idx], T[idx, :3, 3], T[idx, :3, :3], cap.imu, R_ci, off, t_range=(a, a + 10))
              if e: win.append((e["s"], e["excitation_rms"]))
          win = np.array(win)
          r[f"step{step}"] = {"whole_scale_err_pct": round(100 * (whole["s"] - 1), 3), "g_norm": round(whole["g_norm"], 3),
                              "excitation_rms": round(whole["excitation_rms"], 3),
                              "random_frame_trials_err_pct": [round(100 * (x - 1), 3) for x in trials],
                              "10s_window_err_pct_median_abs": round(100 * float(np.median(np.abs(win[:, 0] - 1))), 2),
                              "10s_window_err_pct_p90_abs": round(100 * float(np.percentile(np.abs(win[:, 0] - 1), 90)), 2),
                              "n_windows": len(win)}
      res[name] = r; print(name, json.dumps(r))
  (OUT / "imu_scale_arkit.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

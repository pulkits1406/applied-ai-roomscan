"""Shared helpers for e08: frame lists, Sim(3) alignment, trajectory metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FRAMES = ROOT / "cache/frames"
CACHE = ROOT / "cache/e08"
OUT = ROOT / "experiments/e08_video_sfm/out"
CAPTURES = ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]


def load_frames(capture: str) -> list[dict]:
    d = json.loads((FRAMES / capture / "frames.json").read_text())
    return d["frames"]


def run_dir(capture: str, stride: int, tag: str) -> Path:
    return CACHE / f"{capture}_s{stride}_{tag}"


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares (s, R, t) with dst ~= s * R @ src + t. src, dst: (N, 3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var_s = (xs**2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def traj_metrics(t: np.ndarray, T_sfm: np.ndarray, T_gt: np.ndarray, window_s: float = 20.0, window_step_s: float = 10.0) -> dict:
    """Compare SfM camera-to-world poses (arbitrary Sim3 frame) with ARKit poses (metres).

    The global Sim3 scale s maps SfM units to metres. Window scales are Sim3 fits restricted to
    a time window; their spread relative to s measures scale drift along the trajectory.
    """
    c_s, c_g = T_sfm[:, :3, 3], T_gt[:, :3, 3]
    s, R, tr = umeyama(c_s, c_g)
    c_al = (s * (R @ c_s.T)).T + tr
    err = np.linalg.norm(c_al - c_g, axis=1)
    out = {
        "n": int(len(t)),
        "scale_m_per_unit": s,
        "ate_rmse_m": float(np.sqrt((err**2).mean())),
        "ate_median_m": float(np.median(err)),
        "ate_max_m": float(err.max()),
        "gt_path_length_m": float(np.linalg.norm(np.diff(c_g, axis=0), axis=1).sum()),
        "gt_extent_m": (c_g.max(0) - c_g.min(0)).round(3).tolist(),
    }
    # Relative pose errors over ~1 s: rotation (deg) and length ratio of camera displacement.
    rot_err, len_ratio = [], []
    for i in range(len(t)):
        j = np.searchsorted(t, t[i] + 1.0)
        if j >= len(t):
            break
        dg = c_g[j] - c_g[i]
        ds = c_s[j] - c_s[i]
        Rg = T_gt[i, :3, :3].T @ T_gt[j, :3, :3]
        Rs = T_sfm[i, :3, :3].T @ T_sfm[j, :3, :3]
        dR = Rg.T @ Rs
        rot_err.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
        if np.linalg.norm(dg) > 0.15:
            len_ratio.append(s * np.linalg.norm(ds) / np.linalg.norm(dg))
    lr = np.array(len_ratio)
    out["rpe_1s_rot_deg_median"] = float(np.median(rot_err)) if rot_err else None
    out["rpe_1s_len_ratio_n"] = int(len(lr))
    out["rpe_1s_len_abs_err_median_pct"] = float(100 * np.median(np.abs(lr - 1))) if len(lr) else None
    out["rpe_1s_len_abs_err_p90_pct"] = float(100 * np.percentile(np.abs(lr - 1), 90)) if len(lr) else None
    # Per-window Sim3 scale.
    wins = []
    t0 = t[0]
    while t0 + window_s <= t[-1] + 1e-6 or not wins:
        m = (t >= t0) & (t < t0 + window_s)
        ext = np.linalg.norm(c_g[m].max(0) - c_g[m].min(0)) if m.sum() else 0
        if m.sum() >= 10 and ext > 0.3:
            sw, Rw, tw = umeyama(c_s[m], c_g[m])
            e = np.linalg.norm((sw * (Rw @ c_s[m].T)).T + tw - c_g[m], axis=1)
            wins.append({"t0": float(t0 - t[0]), "n": int(m.sum()), "gt_extent_m": float(ext), "scale_rel": sw / s,
                         "ate_rmse_m": float(np.sqrt((e**2).mean()))})
        t0 += window_step_s
        if t0 >= t[-1]:
            break
    out["windows"] = wins
    if wins:
        sr = np.array([w["scale_rel"] for w in wins])
        out["window_scale_rel_min_max"] = [float(sr.min()), float(sr.max())]
        out["window_scale_rel_std_pct"] = float(100 * sr.std())
        out["window_scale_rel_range_pct"] = float(100 * (sr.max() - sr.min()))
    out["_align"] = {"s": s, "R": R.tolist(), "t": tr.tolist()}
    return out

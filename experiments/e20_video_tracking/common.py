"""Shared helpers for e20: frames, caches, Sim(3) alignment and continuous-tracking metrics.

Imports only numpy (plus cv2 lazily) so it loads in all three venvs (project, cache/e20/.venv,
cache/e13/.venv). Poses are camera-to-world 4x4 in the OpenCV camera convention of the cached
960x720 landscape frames, like ARKit T_wc in frames.json.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FRAMES = ROOT / "cache/frames"
CACHE = ROOT / "cache/e20"
OUT = ROOT / "experiments/e20_video_tracking/out"
CAPTURES = ["single_room", "single_scan_with_ceiling", "single_scan_floor_only"]
MOGE_DIRS = [CACHE / "moge", ROOT / "cache/e08/moge"]  # e08 cache: same model, settings and format
MOGE_WH = (480, 360)


def load_frames(capture: str) -> list[dict]:
    return json.loads((FRAMES / capture / "frames.json").read_text())["frames"]


def img_path(capture: str, row: int) -> Path:
    return FRAMES / capture / "rgb" / f"{row:06d}.jpg"


def median_K(frames) -> np.ndarray:
    return np.median(np.array([f["K"] for f in frames]), axis=0)


def moge_path(capture: str, row: int) -> Path | None:
    for d in MOGE_DIRS:
        p = d / capture / f"{row:06d}.npy"
        if p.exists():
            return p
    return None


_PAGEOUT_HIST: list[tuple[float, int]] = []


def _vm() -> tuple[int, int, float]:
    """(free pages, pageouts + swapouts, swap used MB). Swapouts are counted with pageouts because
    anonymous (model) memory leaves RAM through the compressor/swap, not through Pageouts."""
    vs = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    get = lambda key: int([l for l in vs.splitlines() if l.startswith(key)][0].split()[-1].rstrip("."))  # noqa: E731
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return get("Pages free"), get("Pageouts") + get("Swapouts"), float(sw.split("used = ")[1].split("M")[0])


def mem_ok(min_free_pages: int = 50000, max_pageouts_per_30s: int = 5000) -> tuple[bool, str]:
    """Pause condition: free pages < 50k, or pageouts+swapouts rising > 5000 within 30 s (sustained paging).

    Swap usage alone is not used: it stays high after a model-load peak without ongoing pressure.
    The pageout rate is measured between successive calls, scaled to 30 s (calls >= 5 s apart).
    """
    free, pageouts, swap = _vm()
    now = time.time()
    _PAGEOUT_HIST.append((now, pageouts))
    while len(_PAGEOUT_HIST) > 2 and now - _PAGEOUT_HIST[1][0] >= 30:
        _PAGEOUT_HIST.pop(0)
    rate = None
    t_old, p_old = _PAGEOUT_HIST[0]
    if now - t_old >= 5:
        rate = (pageouts - p_old) * 30.0 / (now - t_old)
    ok = free >= min_free_pages and (rate is None or rate <= max_pageouts_per_30s)
    r = "n/a" if rate is None else f"{rate:.0f}"
    return ok, f"swap {swap:.0f}MB free_pages {free} page+swapouts/30s {r}"


def wait_for_memory(log=print, max_wait_s: int = 600) -> str:
    t0 = time.time()
    while True:
        ok, msg = mem_ok()
        if ok:
            return msg
        log(f"[mem] waiting: {msg}")
        if time.time() - t0 > max_wait_s:
            raise SystemExit(f"memory did not recover in {max_wait_s}s: {msg}")
        time.sleep(10)


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares (s, R, t) with dst ~= s * R @ src + t. src, dst: (N, 3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / ((xs ** 2).sum() / len(src))) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def apply_sim3(s, R, t, c: np.ndarray) -> np.ndarray:
    return s * c @ R.T + t


def rot_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def pieces_from_poses(poses: dict[int, np.ndarray], piece_of: dict[int, int] | None = None) -> list[list[int]]:
    """Group rows into pieces (each piece has its own Sim(3) frame); sorted by size, largest first."""
    groups: dict[int, list[int]] = {}
    for r in poses:
        groups.setdefault(0 if piece_of is None else piece_of[r], []).append(r)
    return sorted((sorted(g) for g in groups.values()), key=len, reverse=True)


def continuity(t_all: np.ndarray, t_piece: np.ndarray, gap_s: float = 1.0) -> dict:
    """Longest run inside a piece with no pose gap > gap_s, and the gaps themselves."""
    t_piece = np.sort(t_piece)
    dt = np.diff(t_piece)
    cut = np.where(dt > gap_s)[0]
    starts = np.r_[0, cut + 1]
    ends = np.r_[cut, len(t_piece) - 1]
    runs = t_piece[ends] - t_piece[starts]
    return {"longest_run_s": float(runs.max()) if len(runs) else 0.0, "n_gaps_gt_1s": int(len(cut)),
            "gaps_s": [[round(float(t_piece[i] - t_all[0]), 1), round(float(dt[i]), 1)] for i in cut][:30]}


def traj_eval(t: np.ndarray, T_est: np.ndarray, T_gt: np.ndarray, metric: bool = False,
              window_s: float = 20.0, step_s: float = 10.0) -> dict:
    """Metrics for ONE continuous trajectory (est in its own Sim(3) frame) vs ARKit (metres).

    metric=True means the estimate claims metres, so its global Sim(3) scale is itself a result.
    """
    c_e, c_g = T_est[:, :3, 3], T_gt[:, :3, 3]
    s, R, tr = umeyama(c_e, c_g)
    c_al = apply_sim3(s, R, tr, c_e)
    err = np.linalg.norm(c_al - c_g, axis=1)
    out = {"n": int(len(t)), "span_s": float(t[-1] - t[0]), "sim3_scale": s,
           "ate_rmse_m": float(np.sqrt((err ** 2).mean())), "ate_median_m": float(np.median(err)),
           "ate_max_m": float(err.max()),
           "gt_path_m": float(np.linalg.norm(np.diff(c_g, axis=0), axis=1).sum())}
    if metric:
        out["metric_scale_err_pct"] = 100 * (1 / s - 1)  # est length / true length - 1
    # 20 s windows: local Sim(3) ATE and local scale relative to the global fit.
    wins, t0 = [], t[0]
    while t0 < t[-1]:
        m = (t >= t0) & (t < t0 + window_s)
        if m.sum() >= 10 and (t[m][-1] - t[m][0]) > 0.6 * window_s:
            ext = float(np.linalg.norm(c_g[m].max(0) - c_g[m].min(0)))
            if ext > 0.5:
                sw, Rw, tw = umeyama(c_e[m], c_g[m])
                e = np.linalg.norm(apply_sim3(sw, Rw, tw, c_e[m]) - c_g[m], axis=1)
                wins.append({"t0": round(float(t0 - t[0]), 1), "n": int(m.sum()), "gt_extent_m": round(ext, 2),
                             "scale_rel": sw / s, "ate_rmse_m": float(np.sqrt((e ** 2).mean()))})
        t0 += step_s
    out["windows"] = wins
    if wins:
        sr = np.array([w["scale_rel"] for w in wins])
        out["win_ate_median_m"] = float(np.median([w["ate_rmse_m"] for w in wins]))
        out["win_ate_max_m"] = float(np.max([w["ate_rmse_m"] for w in wins]))
        out["win_scale_minmax"] = [float(sr.min()), float(sr.max())]
        out["win_scale_spread_pct"] = float(100 * (sr.max() - sr.min()))
        out["win_scale_std_pct"] = float(100 * sr.std())
    # Drift: Sim(3) fitted on the first 30 s only, error growth afterwards.
    m30 = t - t[0] <= 30.0
    if m30.sum() >= 10 and np.linalg.norm(c_g[m30].max(0) - c_g[m30].min(0)) > 0.5:
        s3, R3, t3 = umeyama(c_e[m30], c_g[m30])
        e3 = np.linalg.norm(apply_sim3(s3, R3, t3, c_e) - c_g, axis=1)
        out["drift_fit30"] = {"err_end_m": float(e3[-1]), "err_max_m": float(e3.max()),
                              "err_at_s": {str(k): float(e3[np.argmin(np.abs(t - t[0] - k))])
                                           for k in (60, 120, 180) if t[-1] - t[0] >= k},
                              "err_per_gt_path_pct": float(100 * e3[-1] / max(out["gt_path_m"], 1e-6))}
    # Long-range distances between camera centres far apart in time (proxy for room-scale lengths):
    # after the single global scale, how wrong is |c_i - c_j| when |c_i - c_j|_gt > 1.5 m?
    rng = np.random.default_rng(0)
    i = rng.integers(0, len(t), 4000)
    j = rng.integers(0, len(t), 4000)
    dg = np.linalg.norm(c_g[i] - c_g[j], axis=1)
    keep = (np.abs(t[i] - t[j]) > 20) & (dg > 1.5)
    if keep.sum() > 20:
        de = s * np.linalg.norm(c_e[i[keep]] - c_e[j[keep]], axis=1)
        rel = 100 * (de / dg[keep] - 1)
        out["longrange_dist_err_pct"] = {"n": int(keep.sum()), "median_abs": float(np.median(np.abs(rel))),
                                         "p90_abs": float(np.percentile(np.abs(rel), 90))}
        if metric:
            de_m = np.linalg.norm(c_e[i[keep]] - c_e[j[keep]], axis=1)
            rel_m = 100 * (de_m / dg[keep] - 1)
            out["longrange_dist_err_pct_native_scale"] = {"median": float(np.median(rel_m)),
                                                          "median_abs": float(np.median(np.abs(rel_m))),
                                                          "p90_abs": float(np.percentile(np.abs(rel_m), 90))}
    # 1 s relative rotation error.
    rot = []
    for a in range(len(t)):
        b = np.searchsorted(t, t[a] + 1.0)
        if b >= len(t):
            break
        Rg = T_gt[a, :3, :3].T @ T_gt[b, :3, :3]
        Re = T_est[a, :3, :3].T @ T_est[b, :3, :3]
        rot.append(rot_deg(Rg, Re))
    out["rpe_1s_rot_deg_median"] = float(np.median(rot)) if rot else None
    out["_align"] = {"s": s, "R": R.tolist(), "t": tr.tolist()}
    return out


def evaluate_method(capture: str, rows_eval: list[int], poses: dict[int, np.ndarray],
                    piece_of: dict[int, int] | None = None, metric: bool = False, extra: dict | None = None) -> dict:
    """Coverage/continuity over rows_eval (the keyframes the method was given) + traj_eval of the largest piece."""
    by_row = {f["row"]: f for f in load_frames(capture)}
    t_all = np.array([by_row[r]["t"] for r in rows_eval])
    pcs = [p for p in pieces_from_poses(poses, piece_of) if len(p) >= 2]
    res = {"capture": capture, "n_input": len(rows_eval), "n_pieces": len(pcs),
           "piece_sizes": [len(p) for p in pcs][:20], "duration_s": float(t_all[-1] - t_all[0])}
    if not pcs:
        return res
    big = pcs[0]
    tb = np.array([by_row[r]["t"] for r in big])
    res["coverage_largest_pct"] = 100 * len(big) / len(rows_eval)
    res["coverage_any_pct"] = 100 * sum(len(p) for p in pcs) / len(rows_eval)
    res.update(continuity(t_all, tb))
    longest = 0.0
    for p in pcs:
        tp = np.array([by_row[r]["t"] for r in p])
        longest = max(longest, continuity(t_all, tp)["longest_run_s"])
    res["longest_run_any_piece_s"] = longest
    T_e = np.stack([poses[r] for r in big])
    T_g = np.stack([np.array(by_row[r]["T_wc"]) for r in big])
    res["largest"] = traj_eval(tb, T_e, T_g, metric=metric)
    res["largest"]["t_range_s"] = [float(tb[0] - t_all[0]), float(tb[-1] - t_all[0])]
    if extra:
        res.update(extra)
    return res


def save_poses(path: Path, poses: dict[int, np.ndarray], piece_of: dict[int, int] | None = None, **meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    items = [{"row": int(r), "piece": int(piece_of[r]) if piece_of else 0, "T_wc": np.round(T, 6).tolist()}
             for r, T in sorted(poses.items())]
    path.write_text(json.dumps({"meta": meta, "poses": items}))


def load_poses(path: Path):
    d = json.loads(path.read_text())
    poses = {it["row"]: np.array(it["T_wc"]) for it in d["poses"]}
    piece_of = {it["row"]: it["piece"] for it in d["poses"]}
    return poses, piece_of, d["meta"]

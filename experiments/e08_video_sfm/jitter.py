"""Pose density and position noise of SfM fragments vs ARKit (pseudo-GT), for IMU-based scale use.

For each model: pose rate (registered poses / time span) and gaps; position residual after
  - global Sim3 (ATE), - local Sim3 over sliding 2 s windows (local accuracy, removes drift),
  - jitter: RMS of the second difference of the local residual / sqrt(6) (white-noise estimate),
all in metres using the model's global Sim3 scale.
Usage: uv run python experiments/e08_video_sfm/jitter.py <capture> <cfg_dir> <variant>
"""
import json
import sys

import numpy as np

from common import CACHE, OUT, load_frames, umeyama
from scale import load_models
from sfm import cam_from_world


def main():
    capture, cfg, variant = sys.argv[1:4]
    frames = {f["row"]: f for f in load_frames(capture)}
    cache_dt = np.median(np.diff(sorted(f["t"] for f in frames.values())))
    res = {"capture": capture, "cfg": cfg, "variant": variant, "cached_frame_rate_hz": float(1 / cache_dt), "models": []}
    for mi, rec in enumerate(load_models(CACHE / cfg / variant)):
        ims = sorted([im for im in rec.images.values() if im.has_pose], key=lambda im: im.name)
        if len(ims) < 30:
            continue
        t = np.array([frames[int(im.name[:6])]["t"] for im in ims])
        c = np.stack([np.linalg.inv(cam_from_world(im))[:3, 3] for im in ims])
        g = np.stack([np.array(frames[int(im.name[:6])]["T_wc"])[:3, 3] for im in ims])
        s, R, tr = umeyama(c, g)
        dt = np.diff(t)
        loc_res, jit = [], []
        for t0 in np.arange(t[0], t[-1] - 2.0, 1.0):
            m = (t >= t0) & (t < t0 + 2.0)
            if m.sum() < 12 or np.linalg.norm(g[m].max(0) - g[m].min(0)) < 0.05:
                continue
            # Local rigid fit with the global scale fixed: residual = local shape error, not scale drift.
            _, Rl, tl = umeyama(s * c[m], g[m], with_scale=False)
            r = (Rl @ (s * c[m]).T).T + tl - g[m]
            loc_res.append(np.sqrt((r ** 2).sum(1).mean()))
            # Uniformly spaced runs only, so the second difference is a noise estimate.
            if np.all(np.abs(np.diff(t[m]) - cache_dt) < 0.3 * cache_dt):
                d2 = r[2:] - 2 * r[1:-1] + r[:-2]
                jit.append(np.sqrt((d2 ** 2).sum(1).mean() / 6))
        # Continuous runs: consecutive poses no further apart than 2.5 cached-frame intervals.
        breaks = np.where(dt > 2.5 * cache_dt)[0]
        starts, ends = np.r_[0, breaks + 1], np.r_[breaks, len(t) - 1]
        run_len = t[ends] - t[starts]
        res["models"].append({
            "continuous_runs": int(len(run_len)), "continuous_run_s_max": float(run_len.max()),
            "continuous_run_s_median": float(np.median(run_len)), "continuous_covered_s": float(run_len.sum()),
            "pose_rate_within_runs_hz": float(((ends - starts).sum()) / max(run_len.sum(), 1e-9)),
            "model": mi, "n_poses": len(ims), "t_span_s": float(t[-1] - t[0]),
            "pose_rate_hz": float((len(ims) - 1) / (t[-1] - t[0])),
            "frac_gaps_over_0p2s": float((dt > 0.2).mean()), "max_gap_s": float(dt.max()),
            "ate_rmse_global_sim3_m": float(np.sqrt((((s * (R @ c.T)).T + tr - g) ** 2).sum(1).mean())),
            "local_2s_rigid_rmse_m_median": float(np.median(loc_res)) if loc_res else None,
            "local_2s_rigid_rmse_m_p90": float(np.percentile(loc_res, 90)) if loc_res else None,
            "jitter_white_m_median": float(np.median(jit)) if jit else None,
            "jitter_white_m_p90": float(np.percentile(jit, 90)) if jit else None,
            "n_windows": len(loc_res)})
        print(json.dumps(res["models"][-1]))
    (OUT / f"{capture}_pose_noise.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

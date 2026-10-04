"""IMU scale vs keyframe position noise (proxy for SfM trajectories before real SfM poses exist).

ARKit trajectory subsampled to ~3 fps keyframes (every 15th row), Gaussian jitter added to
positions (rotations kept), then the lever-arm model (lever_arm.py) at sigma 0.4 / 0.6 s.
Reported: forward, reverse and their geometric mean (errors-in-variables bracket).
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from imu_scale import estimate_extrinsic_and_offset
from lever_arm import build, fit
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e11_imu_scale/out"
rng = np.random.default_rng(1); res = {}
for name in ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]:
    cap = StrayCapture.load(ROOT / name); T = cap.T_wc(); t = cap.timestamps
    off, R_ci, _ = estimate_extrinsic_and_offset(t, T[:, :3, :3], cap.imu)
    idx = np.arange(0, len(cap), 15); r = {}
    for noise in [0.0, 0.005, 0.01, 0.02]:
        for sg in [0.4, 0.6]:
            gm = []
            for rep in range(5):
                p = T[idx, :3, 3] + rng.normal(0, noise, (len(idx), 3))
                a, b, m, ma = build(t[idx], p, T[idx, :3, :3], cap.imu, R_ci, off, sg)
                fw, _ = fit(a, b, m, ma); rv, _ = fit(a, b, m, ma, reverse=True)
                gm.append(np.sqrt(fw * rv) - 1)
            r[f"noise{int(noise*1000)}mm_sigma{sg}"] = {"geo_mean_err_pct_median": round(100 * float(np.median(gm)), 2), "spread_pct": round(100 * float(np.ptp(gm)), 2)}
    res[name] = r; print(name, json.dumps(r))
(OUT / "noise_robustness.json").write_text(json.dumps(res, indent=1))

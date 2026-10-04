"""Fuse depth into a voxelised world cloud per capture; top-down + height histogram views."""
import json, sys
from pathlib import Path
import numpy as np, open3d as o3d, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e02_fusion/out"; OUT.mkdir(parents=True, exist_ok=True)
STEP = 5
summary = {}
for name in sys.argv[1:] or ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]:
    cap = StrayCapture.load(ROOT / name)
    pts = np.concatenate([cap.points_world(i, min_conf=2, max_depth=4.0, stride=2) for i in range(0, len(cap), STEP)])
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).voxel_down_sample(0.02)
    o3d.io.write_point_cloud(str(OUT / f"{name}_vox2cm.ply"), pc)
    P = np.asarray(pc.points); T = cap.T_wc()
    cam = T[:, :3, 3]
    # which world axis is vertical: camera optical axes are mostly horizontal-ish; IMU-free check via
    # the axis along which the cloud has two dominant planes (floor/ceiling) and the trajectory is flat
    hist = {ax: np.histogram(P[:, k], bins=np.arange(P[:, k].min(), P[:, k].max() + 0.02, 0.02)) for k, ax in enumerate("xyz")}
    peak_frac = {ax: float(h[0].max() / h[0].sum()) for ax, (h) in hist.items()}
    h, e = hist["y"]
    top = np.argsort(h)[::-1][:6]
    summary[name] = {"n_points": len(P), "bbox_min": P.min(0).round(2).tolist(), "bbox_max": P.max(0).round(2).tolist(),
                     "peak_bin_fraction_by_axis": peak_frac,
                     "y_peaks": [(round(float(e[i] + 0.01), 3), int(h[i])) for i in sorted(top)],
                     "cam_y_median": float(np.median(cam[:, 1]))}
    fig, ax = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [3, 1]})
    sel = (P[:, 1] > np.percentile(P[:, 1], 5)) & (P[:, 1] < np.percentile(P[:, 1], 95))
    ax[0].hist2d(P[sel, 0], P[sel, 2], bins=[int(np.ptp(P[:, 0]) / 0.03), int(np.ptp(P[:, 2]) / 0.03)], cmap="gray_r", vmax=40)
    ax[0].plot(cam[:, 0], cam[:, 2], "r-", lw=0.6); ax[0].plot(cam[0, 0], cam[0, 2], "go")
    ax[0].set_aspect("equal"); ax[0].set_title(f"{name}: top-down (x,z), mid-height points; red=trajectory")
    ax[1].barh(e[:-1], h, height=0.02); ax[1].axhline(np.median(cam[:, 1]), color="r"); ax[1].set_title("point count vs world y (m)")
    ax[1].invert_yaxis()
    fig.tight_layout(); fig.savefig(OUT / f"{name}_topdown.png", dpi=90); plt.close(fig)
    print(name, json.dumps(summary[name]))
(OUT / "fusion_summary.json").write_text(json.dumps(summary, indent=1))

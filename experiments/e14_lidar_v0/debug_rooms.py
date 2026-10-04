"""Per-room debug view for lidar_pipeline v0: points (top view, room frame), wall lines, region, polygon."""
import sys
from pathlib import Path
import numpy as np, matplotlib
from scipy import ndimage as ndi
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from roomscan import geometry as G
from roomscan.lidar_pipeline import _visits, STEP
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e14_lidar_v0/out"; OUT.mkdir(parents=True, exist_ok=True)
name = sys.argv[1] if len(sys.argv) > 1 else "single_scan_with_ceiling"
cap = StrayCapture.load(ROOT / name); T = cap.T_wc(); t = cap.timestamps
pc = G.fuse(cap, T, range(0, len(cap), STEP)); P, N = np.asarray(pc.points), np.asarray(pc.normals)
seg = G.segment_rooms(P, N); centers = seg.grid.centers()
lab_d = ndi.grey_dilation(seg.labels, size=(7, 7))
cij = np.clip(seg.grid.cell(T[:, [0, 2], 3]), 0, np.array(seg.grid.shape) - 1)
rof = np.where(seg.labels[cij[:, 0], cij[:, 1]] > 0, seg.labels[cij[:, 0], cij[:, 1]], lab_d[cij[:, 0], cij[:, 1]])
L = seg.labels.max(); fig, axs = plt.subplots(2, (L + 1) // 2, figsize=(5 * ((L + 1) // 2), 10)); axs = axs.ravel()
for label in range(1, L + 1):
    ax = axs[label - 1]; region = seg.labels == label
    near = ndi.binary_dilation(region, iterations=10)
    ij = np.clip(seg.grid.cell(P[:, [0, 2]]), 0, np.array(seg.grid.shape) - 1); k = near[ij[:, 0], ij[:, 1]]
    RP, RN = P[k], N[k]; yaw = G.manhattan_yaw(RN); Rm = G._rot(yaw)
    UV = RP[:, [0, 2]] @ Rm.T; NUV = RN[:, [0, 2]] @ Rm.T; Hh = RP[:, 1] - seg.grid.floor_y
    us, _ = G.wall_lines(UV, NUV, Hh, 0); vs, _ = G.wall_lines(UV, NUV, Hh, 1)
    vert = (np.abs(RN[:, 1]) < 0.3) & (Hh > 0.3)
    ax.scatter(UV[vert, 0], UV[vert, 1], s=0.2, c=Hh[vert], cmap="viridis")
    reg = centers[G.fill_region(seg.labels, label)] @ Rm.T; ax.scatter(reg[:, 0], reg[:, 1], s=1, c="orange", alpha=0.3)
    for u in us: ax.axvline(u, color="r", lw=0.6)
    for v in vs: ax.axhline(v, color="b", lw=0.6)
    struct = (np.abs(RN[:, 1]) < 0.3) & (Hh > G.H_STRUCT)
    other = centers[(seg.labels > 0) & (seg.labels != label)] @ Rm.T
    poly = G.room_polygon(reg, us, vs, structural_uv=UV[struct], other_uv=other)
    if poly is not None: q = np.vstack([poly, poly[:1]]); ax.plot(q[:, 0], q[:, 1], "k-", lw=2)
    vis = _visits(rof, t, label)
    ax.set_title(f"label {label}: {len(us)}u/{len(vs)}v lines, visits={len(vis)}, poly={'none' if poly is None else len(poly)}"); ax.set_aspect("equal")
fig.tight_layout(); fig.savefig(OUT / f"{name}_rooms_debug.png", dpi=60); print("ok")

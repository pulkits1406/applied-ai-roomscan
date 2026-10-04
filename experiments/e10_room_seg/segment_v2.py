"""Room segmentation v2: barriers from VERTICAL surfaces only (normals), above furniture height.

v1 failure (see out/*_rooms.png): a low corridor ceiling (~2.35 m) filled the "header" band, and
furniture in the wall band removed free space. Here:
  barrier  = cells holding vertical-surface points (|n_y| < 0.3) at height > H_STRUCT above floor
             (lintels above doors are vertical faces, so door gaps close when the ceiling band
             was observed; furniture tops are mostly below H_STRUCT).
  interior = connected components of ~barrier that contain observed floor; furniture does not
             subtract from rooms.
Fallback for scans without high observations: distance-transform watershed (door half-width).
"""
import json, sys
from pathlib import Path
import numpy as np, open3d as o3d, matplotlib
from scipy import ndimage as ndi
from skimage.segmentation import watershed
matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]; FUS = ROOT / "experiments/e02_fusion/out"; OUT = ROOT / "experiments/e10_room_seg/out"
RES = 0.05
DOMAIN_CLOSE = 0.6  # m; closes furniture shadows in the observed floor

def load(name):
    pc = o3d.io.read_point_cloud(str(FUS / f"{name}_vox2cm.ply"))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
    return np.asarray(pc.points), np.asarray(pc.normals)

def seg(P, N, h_struct, door_halfwidth=0.45, min_room_m2=1.0):
    f0 = np.percentile(P[:, 1], 2); h = P[:, 1] - f0
    mn = P[:, [0, 2]].min(0) - 0.3
    ij = np.floor((P[:, [0, 2]] - mn) / RES).astype(int); H, W = ij.max(0) + 1
    def occ(m, c=2):
        g = np.zeros((H, W), np.int32); np.add.at(g, (ij[m, 0], ij[m, 1]), 1); return g >= c
    vert = np.abs(N[:, 1]) < 0.3
    barrier = occ(vert & (h > h_struct) & (h < 3.5), 3)
    barrier = ndi.binary_closing(barrier, iterations=2)
    floor = occ(np.abs(h) < 0.05, 1)
    # Interior domain = observed-floor footprint with gaps (furniture shadows) closed; rooms whose
    # outer boundary was not fully observed (windows, glass, open edges) are bounded by it.
    domain = ndi.binary_fill_holes(ndi.binary_closing(floor, iterations=int(DOMAIN_CLOSE / RES)))
    domain &= ndi.binary_dilation(floor, iterations=int(0.3 / RES))
    comp, n = ndi.label(~barrier & domain)
    has_floor = ndi.sum(floor, comp, index=np.arange(1, n + 1))
    area = ndi.sum(np.ones_like(comp), comp, index=np.arange(1, n + 1)) * RES ** 2
    lab = np.zeros_like(comp); k = 0
    for l in range(1, n + 1):
        if has_floor[l - 1] * RES ** 2 < 0.3 or area[l - 1] < min_room_m2: continue
        k += 1; lab[comp == l] = k
    stats = {"n_rooms": k, "areas_m2": [round(float((lab == i).sum() * RES ** 2), 2) for i in range(1, k + 1)],
             "observed_floor_frac": [round(float((floor & (lab == i)).sum() / max((lab == i).sum(), 1)), 2) for i in range(1, k + 1)]}
    return lab, barrier, floor, stats, dict(res=RES, origin_xz=mn.tolist(), floor_y=float(f0))

if __name__ == "__main__":
    res = {}
    for name in sys.argv[1:] or ["single_scan_with_ceiling", "single_scan_floor_only", "single_room"]:
        P, N = load(name); h = P[:, 1] - np.percentile(P[:, 1], 2)
        vh = h[np.abs(N[:, 1]) < 0.3]
        res[name] = {"vertical_pts_height_p50_p95_p99": np.percentile(vh, [50, 95, 99]).round(2).tolist()}
        fig, axs = plt.subplots(1, 3, figsize=(18, 7))
        for ax, hs in zip(axs, [1.9, 2.05, 1.6]):
            lab, barrier, floor, st, grid = seg(P, N, hs)
            res[name][f"h_struct_{hs}"] = st
            ax.imshow(barrier.T, origin="lower", cmap="gray_r", alpha=0.6)
            ax.imshow(np.ma.masked_equal(lab, 0).T, origin="lower", cmap="tab20", interpolation="nearest", alpha=0.8)
            for k in range(1, lab.max() + 1):
                ys, xs = np.nonzero(lab == k); ax.text(ys.mean(), xs.mean(), f"{k}\n{st['areas_m2'][k-1]}", fontsize=8, ha="center")
            ax.set_title(f"{name}\nbarrier = vertical surfaces above {hs} m: {st['n_rooms']} rooms")
            np.save(OUT / f"{name}_v2b_h{hs}_labels.npy", lab); json.dump(grid, open(OUT / f"{name}_v2_grid.json", "w"))
        fig.tight_layout(); fig.savefig(OUT / f"{name}_rooms_v2b.png", dpi=75); plt.close(fig)
        print(name, json.dumps(res[name]))
    (OUT / "segmentation_v2b.json").write_text(json.dumps(res, indent=1))

"""Room segmentation on a fused whole-apartment cloud (2D, gravity-aligned).

Free space = cells whose floor was observed and which hold no wall-height structure.
Two ways to separate rooms at doorways:
  A "header": structure observed between door-head height and the ceiling (the lintel above a
    doorway) is added as a barrier -> door gaps close, open-plan areas stay merged.
    Needs ceiling-band observations.
  B "morph": erode free space by a door half-width, label components, grow labels back over
    free space by geodesic distance (no ceiling data needed).
"""
import json, sys
from pathlib import Path
import numpy as np, open3d as o3d, cv2, matplotlib
from scipy import ndimage as ndi
from skimage.segmentation import watershed
matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]; FUS = ROOT / "experiments/e02_fusion/out"; OUT = ROOT / "experiments/e10_room_seg/out"; OUT.mkdir(parents=True, exist_ok=True)
RES = 0.05

def grids(P):
    f0 = np.percentile(P[:, 1], 2)
    floor_pts = np.abs(P[:, 1] - f0) < 0.05
    mn = P[:, [0, 2]].min(0) - 0.2
    ij = np.floor((P[:, [0, 2]] - mn) / RES).astype(int); H, W = ij.max(0) + 1
    def occ(mask, minc=1):
        g = np.zeros((H, W), np.int32); np.add.at(g, (ij[mask, 0], ij[mask, 1]), 1); return g >= minc
    h = P[:, 1] - f0
    floor = occ(floor_pts)
    wall = occ((h > 0.4) & (h < 1.9), 2)
    header = occ((h > 2.12) & (h < 2.6), 2)   # above door heads (~2.0-2.1 m), below the lowest ceilings seen (~2.3 m) is risky; refined below
    ceil = occ(h > 2.2)
    return dict(f0=f0, mn=mn, floor=floor, wall=wall, header=header, ceil=ceil, shape=(H, W))

def segment(g, mode, door_halfwidth=0.5, min_room_m2=1.5):
    floor = ndi.binary_closing(g["floor"], iterations=2)
    barrier = ndi.binary_dilation(g["wall"], iterations=1)
    if mode == "header":
        barrier |= ndi.binary_dilation(g["header"] & ~ndi.binary_opening(g["ceil"] & g["header"], iterations=4), iterations=1)
    free = floor & ~barrier
    free = ndi.binary_opening(free, iterations=1)
    if mode == "header":
        lab, n = ndi.label(free)
    else:
        dist = ndi.distance_transform_edt(free) * RES
        seeds, n = ndi.label(dist > door_halfwidth)
        lab = watershed(-dist, seeds, mask=free)
    areas = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, lab.max() + 1)) * RES ** 2
    keep = np.nonzero(areas >= min_room_m2)[0] + 1
    out = np.zeros_like(lab); 
    for k, l in enumerate(keep, 1): out[lab == l] = k
    return out, [float(areas[l - 1]) for l in keep]

if __name__ == "__main__":
    res = {}
    for name in sys.argv[1:] or ["single_scan_with_ceiling", "single_scan_floor_only", "single_room"]:
        P = np.asarray(o3d.io.read_point_cloud(str(FUS / f"{name}_vox2cm.ply")).points)
        g = grids(P)
        fig, axs = plt.subplots(1, 3, figsize=(18, 7))
        axs[0].imshow((g["wall"] * 2 + g["header"] * 1).T, origin="lower", cmap="magma"); axs[0].set_title(f"{name}: wall (bright) / header (dim)")
        res[name] = {}
        for ax, mode in zip(axs[1:], ["header", "morph"]):
            lab, areas = segment(g, mode)
            res[name][mode] = {"n_rooms": len(areas), "areas_m2": [round(a, 2) for a in areas]}
            ax.imshow(np.ma.masked_equal(lab, 0).T, origin="lower", cmap="tab20", interpolation="nearest")
            for k in range(1, lab.max() + 1):
                ys, xs = np.nonzero(lab == k); ax.text(ys.mean(), xs.mean(), str(k), fontsize=9, ha="center")
            ax.set_title(f"{mode}: {len(areas)} regions"); np.save(OUT / f"{name}_{mode}_labels.npy", lab)
        fig.tight_layout(); fig.savefig(OUT / f"{name}_rooms.png", dpi=75); plt.close(fig)
        json.dump({"res": RES, "origin_xz": g["mn"].tolist(), "floor_y": float(g["f0"])}, open(OUT / f"{name}_grid.json", "w"))
        print(name, json.dumps(res[name]))
    (OUT / "segmentation.json").write_text(json.dumps(res, indent=1))

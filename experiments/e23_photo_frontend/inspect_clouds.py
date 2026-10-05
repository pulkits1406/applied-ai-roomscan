"""Diagnostics for photo-tier room clouds (reads the e23 model cache, CPU only): gravity check,
floor/ceiling height histograms of horizontal surfaces, top-down view with camera positions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roomscan import geometry as G
from roomscan.photo import inference, reconstruct as RC
from roomscan.photo.images import load_photo, room_folders

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/e23_photo_frontend/out"


def main(folder="cache/photo_sim/single_scan_with_ceiling_K6", cache="runs/e23/cache", mode="moge"):
    rep = {}
    fs = room_folders(ROOT / folder)
    fig, axs = plt.subplots(2, len(fs), figsize=(4 * len(fs), 8))
    for k, (name, paths) in enumerate(fs.items()):
        photos = [load_photo(p) for p in paths]
        pc = RC.room_cloud(inference.moge(photos, ROOT / cache / name), inference.mapanything(photos, ROOT / cache / name), mode)
        P, N = pc.P, pc.N
        hor = np.abs(N[:, 1]) > 0.9
        f0 = float(np.percentile(P[:, 1], 2))
        up_f = hor & (N[:, 1] > 0); dn_f = hor & (N[:, 1] < 0)
        floor_mode = float(np.median(P[up_f, 1])) if up_f.any() else None
        rep[name] = {"n": int(len(P)), "frac_horizontal": round(float(hor.mean()), 3), "frac_vertical": round(float((np.abs(N[:, 1]) < 0.3).mean()), 3),
                     "p2_y": round(f0, 3), "floor_median_y(up-normals)": floor_mode,
                     "ceiling_median_y(down-normals)": float(np.median(P[dn_f, 1])) if dn_f.any() else None,
                     "frac_within_5cm_of_p2": round(float((np.abs(P[:, 1] - f0) < 0.05).mean()), 3),
                     "cam_height_above_floor": None if floor_mode is None else np.round(pc.cams[:, 1] - floor_mode, 2).tolist()}
        ax = axs[0, k]
        ax.hist(P[up_f, 1], bins=100, alpha=0.6, label="up-facing"); ax.hist(P[dn_f, 1], bins=100, alpha=0.6, label="down-facing")
        ax.axvline(f0, color="k", ls=":"); ax.set_title(f"{name}: y of horizontal surfaces"); ax.legend(fontsize=6)
        ax = axs[1, k]
        s = np.random.default_rng(0).choice(len(P), min(60000, len(P)), replace=False)
        ax.scatter(P[s, 0], P[s, 2], s=0.2, c=P[s, 1], cmap="viridis")
        ax.scatter(pc.cams[:, 0], pc.cams[:, 2], c="r", s=20, marker="^")
        ax.set_aspect("equal"); ax.set_title(f"{name} top-down (x, z)")
    fig.tight_layout(); fig.savefig(OUT / f"inspect_{mode}.png", dpi=90)
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:])

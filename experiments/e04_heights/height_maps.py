"""Floor-height and ceiling-height maps on a 0.3 m grid.

Floor flatness across the apartment exposes vertical drift / gravity error (a real floor is
flat to a few mm); the ceiling-minus-floor map shows how many distinct ceiling heights exist.
"""
import json
from pathlib import Path
import numpy as np, open3d as o3d, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]; FUS = ROOT / "experiments/e02_fusion/out"; OUT = ROOT / "experiments/e04_heights/out"; OUT.mkdir(parents=True, exist_ok=True)
G = 0.3
res = {}
for name in ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]:
    P = np.asarray(o3d.io.read_point_cloud(str(FUS / f"{name}_vox2cm.ply")).points)
    f0 = np.percentile(P[:, 1], 2)
    ij = np.floor(P[:, [0, 2]] / G).astype(int); ij -= ij.min(0)
    H, W = ij.max(0) + 1
    floor = np.full((H, W), np.nan); ceil = np.full((H, W), np.nan)
    key = ij[:, 0] * W + ij[:, 1]
    order = np.argsort(key); key, Ps = key[order], P[order]
    starts = np.r_[0, np.nonzero(np.diff(key))[0] + 1, len(key)]
    for s, e in zip(starts[:-1], starts[1:]):
        y = Ps[s:e, 1]; k = key[s]
        lo = y[y < f0 + 0.15]
        if len(lo) >= 30: floor.flat[k] = np.median(lo)
        hi = y[y > f0 + 2.1]
        if len(hi) >= 30:
            h, edges = np.histogram(hi, bins=np.arange(hi.min(), hi.max() + 0.02, 0.01))
            c = edges[np.argmax(h)] + 0.005; ceil.flat[k] = np.median(hi[np.abs(hi - c) < 0.02])
    fl = floor[~np.isnan(floor)]
    r = {"floor_y_p5_p50_p95_mm": (1000 * np.percentile(fl, [5, 50, 95])).round(1).tolist(), "floor_cells": int(len(fl))}
    if np.isfinite(ceil).any():
        ch = (ceil - np.nanmedian(fl))
        v = ch[np.isfinite(ch)]
        h, e = np.histogram(v, bins=np.arange(2.0, 3.5, 0.02))
        r["ceiling_height_modes_m"] = [(round(float(e[i] + 0.01), 2), int(h[i])) for i in np.argsort(h)[::-1][:5]]
    res[name] = r; print(name, json.dumps(r))
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    im = ax[0].imshow(1000 * (floor - np.nanmedian(fl)).T, origin="lower", cmap="coolwarm", vmin=-30, vmax=30); plt.colorbar(im, ax=ax[0], label="floor y - median (mm)")
    ax[0].set_title(f"{name}: floor height deviation")
    im = ax[1].imshow((ceil - np.nanmedian(fl)).T, origin="lower", cmap="viridis", vmin=2.3, vmax=3.1); plt.colorbar(im, ax=ax[1], label="ceiling - floor (m)")
    ax[1].set_title("ceiling height")
    fig.tight_layout(); fig.savefig(OUT / f"{name}_heights.png", dpi=80); plt.close(fig)
(OUT / "height_maps.json").write_text(json.dumps(res, indent=1))

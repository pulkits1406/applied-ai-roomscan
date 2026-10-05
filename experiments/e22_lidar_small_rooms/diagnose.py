"""Stage-by-stage evidence for why a room yields no polygon (reads cache/e22 from dump_rooms.py).

Per room: Manhattan yaw, every density peak of wall-facing points per axis/side with its vertical
extent and top (the two structural tests in geometry.wall_lines), the accepted lines, the
arrangement-cell coverage, and a top-down figure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks

from roomscan import geometry as G
from roomscan.lidar import walls as WL

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/e22_lidar_small_rooms/out"


def peaks_detail(UV, NUV, H, axis, sign):
    m = sign * NUV[:, axis] > 0.85
    x = UV[m, axis]
    if len(x) < 50:
        return []
    bins = np.arange(x.min() - 0.05, x.max() + 0.06, 0.01)
    hist, edges = np.histogram(x, bins=bins)
    sm = ndi.gaussian_filter1d(hist.astype(float), 1.0)
    pk, _ = find_peaks(sm, height=max(8.0, 0.02 * sm.max()), distance=6)
    rows = []
    for p in pk:
        c = edges[p] + 0.005
        sel = np.abs(x - c) < 0.02
        h = H[m][sel]
        ext = float(np.percentile(h, 95) - np.percentile(h, 5))
        top = float(np.percentile(h, 98))
        rows.append({"coord": round(float(c), 3), "n": int(sel.sum()), "extent": round(ext, 2), "top": round(top, 2),
                     "accepted": bool(ext >= 1.2 and top >= 2.0)})
    return rows


def main(capture="single_scan_with_ceiling", rooms="2,6,3,4", tag=""):
    d = ROOT / "cache/e22" / (capture + tag)
    seg = np.load(d / "seg.npz")
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for r in map(int, rooms.split(",")):
        z = np.load(d / f"room{r}.npz")
        P, N, floor_y = z["P"], z["N"], float(seg["floor_y"])
        yaw = G.manhattan_yaw(N)
        Rm = G._rot(yaw)
        UV, NUV, H = P[:, [0, 2]] @ Rm.T, N[:, [0, 2]] @ Rm.T, P[:, 1] - floor_y
        reg = z["region"] @ Rm.T
        rep = {"n_points": int(len(P)), "region_m2": round(len(z["region"]) * G.GRID**2, 2), "yaw_deg": round(np.degrees(yaw), 1),
               "frac_points_above_2m": round(float((H > 2.0).mean()), 3),
               "frac_vertical_above_1.9": round(float(((np.abs(N[:, 1]) < 0.3) & (H > 1.9)).mean()), 3),
               "region_u_range": [round(float(reg[:, 0].min()), 2), round(float(reg[:, 0].max()), 2)],
               "region_v_range": [round(float(reg[:, 1].min()), 2), round(float(reg[:, 1].max()), 2)]}
        for axis, nm in ((0, "u"), (1, "v")):
            for sign in (1, -1):
                rep[f"peaks_{nm}{'+' if sign > 0 else '-'}"] = peaks_detail(UV, NUV, H, axis, sign)
        report[r] = rep
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))
        for ax, (lo, hi, ttl) in zip(axs, ((0.3, 1.9, "vertical pts 0.3-1.9 m"), (1.9, 3.5, "vertical pts > 1.9 m"))):
            m = (np.abs(N[:, 1]) < 0.3) & (H > lo) & (H < hi)
            ax.scatter(reg[:, 0], reg[:, 1], s=2, c="0.85")
            oth = z["other"] @ Rm.T
            near = np.linalg.norm(oth[:, None] - reg.mean(0)[None], axis=-1).ravel() < 5
            ax.scatter(oth[near, 0], oth[near, 1], s=1, c="mistyrose")
            ax.scatter(UV[m, 0], UV[m, 1], s=0.3, c=H[m], cmap="viridis")
            for key, col in (("peaks_u+", "r"), ("peaks_u-", "m"), ("peaks_v+", "b"), ("peaks_v-", "c")):
                for p in rep[key]:
                    ls = "-" if p["accepted"] else ":"
                    (ax.axvline if key.startswith("peaks_u") else ax.axhline)(p["coord"], color=col, ls=ls, lw=0.8)
            ax.set_xlim(reg[:, 0].min() - 1.5, reg[:, 0].max() + 1.5); ax.set_ylim(reg[:, 1].min() - 1.5, reg[:, 1].max() + 1.5)
            ax.set_aspect("equal"); ax.set_title(f"room {r}: {ttl} (solid = accepted wall line)")
        fig.tight_layout(); fig.savefig(OUT / f"diag{tag}_room{r}.png", dpi=110); plt.close(fig)
    (OUT / f"diagnose{tag}.json").write_text(json.dumps(report, indent=1))
    for r, rep in report.items():
        print(f"--- room {r}", {k: v for k, v in rep.items() if not k.startswith("peaks")})
        for k, v in rep.items():
            if k.startswith("peaks"):
                print("  ", k, [(p["coord"], p["n"], p["extent"], p["top"], "OK" if p["accepted"] else "x") for p in v])


if __name__ == "__main__":
    main(*sys.argv[1:])

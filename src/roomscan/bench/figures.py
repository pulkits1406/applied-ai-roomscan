"""Visual evaluation: predicted vs reference wall lengths with their 90 % intervals, one panel per
capture, coloured by status (measured / inferred), uncovered references marked."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def wall_figure(site, results: list[dict], out: str | Path):
    caps = [r for r in results if r.get("walls")]
    if not caps:
        return
    n = len(caps)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), squeeze=False)
    for ax, r in zip(axs.ravel(), caps):
        w = [x for x in r["walls"] if x.get("pred") is not None]
        miss = [x for x in r["walls"] if x.get("pred") is None]
        lim = max([x["gt"] for x in r["walls"]] + [x["pred"] for x in w] + [1.0]) * 1.1
        ax.plot([0, lim], [0, lim], "k:", lw=0.8)
        for x in w:
            hw = x.get("half_width") or 0.0
            col = "tab:orange" if x.get("status") == "inferred" else "tab:blue"
            ax.errorbar(x["gt"], x["pred"], yerr=hw, fmt="o", color=col, ms=4, capsize=2, mfc=col if x.get("covered") else "white")
        for x in miss:
            ax.plot(x["gt"], 0.02 * lim, "x", color="tab:red")
        ttl = f"{r['capture']}" + (f"\n{r.get('question')}: {r.get('variant')}" if r.get("question") else "")
        if any(m.get("method") == "perimeter" for m in r.get("room_match", [])):
            ttl += "\nrooms size-matched (identity unverified)"
        ax.set_title(ttl, fontsize=8); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("reference (m)", fontsize=7); ax.set_ylabel("predicted (m)", fontsize=7); ax.tick_params(labelsize=7)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"{site.site_id} ({site.purpose}, gt={site.gt_kind}): filled = interval covers reference, "
                 "open = not covered, orange = inferred, red x = unmatched", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)

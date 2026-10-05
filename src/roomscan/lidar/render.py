"""Render a Plan as a dimensioned floor-plan PNG (any tier)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roomscan.model import Plan


def render(plan: Plan, out: str | Path, title: str | None = None):
    surf = {s.id: s for s in plan.surfaces}
    fig, ax = plt.subplots(figsize=(10, 10))
    for r in plan.rooms:
        P = np.array(r.polygon)
        ax.fill(P[:, 0], P[:, 1], alpha=0.12)
        Q = np.vstack([P, P[:1]])
        ax.plot(Q[:, 0], Q[:, 1], "k-", lw=1.4)
        for wid in r.wall_ids:
            s = surf[wid]
            a, b = np.array(s.polygon[0][:2]), np.array(s.polygon[1][:2])
            m, L = 0.5 * (a + b), s.length
            if L.value is not None and L.value > 0.3:
                d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
                nrm = np.array([-d[1], d[0]])
                ax.text(*(m + 0.12 * nrm), f"{100 * L.value:.0f}±{100 * (L.interval.hi - L.value):.0f}", fontsize=6, ha="center", va="center",
                        rotation=np.degrees(np.arctan2(d[1], d[0])) % 180 - (180 if np.degrees(np.arctan2(d[1], d[0])) % 180 > 90 else 0))
        c = P.mean(0)
        ch = r.ceiling_height
        ax.text(*c, f"{r.label or r.id}\n{r.floor_area.value:.2f} m²\nh {'—' if ch.value is None else f'{ch.value:.2f}'} m", ha="center", fontsize=8, weight="bold")
    for o in plan.openings:
        rooms = [r for r in plan.rooms if r.id in o.room_ids]
        if len(rooms) == 2:
            a, b = (np.array(r.polygon).mean(0) for r in rooms)
            ax.plot(*np.stack([a, b]).T, ":", color="tab:green", lw=1)
            ax.text(*(0.5 * (a + b)), f"{o.kind} {100 * o.width.value:.0f} cm", fontsize=6, color="tab:green", ha="center")
    ax.set_aspect("equal"); ax.axis("off")
    cal = any(m.interval and m.interval.calibrated for r in plan.rooms for m in (r.floor_area,))
    ax.set_title((title or f"{plan.capture.tier} plan") + ("" if cal else "\n(dimensions in cm; intervals UNCALIBRATED)"), fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)

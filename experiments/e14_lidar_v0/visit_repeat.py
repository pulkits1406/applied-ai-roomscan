"""Within-capture repeatability proxy: room walls rebuilt from visit 1 vs visit 2 (independent frames).

Not the assessment's repeatability gate (that needs two captures made per our protocol), but it
isolates how much wall positions move between independent observations of the same room inside
one ARKit session. Compared quantity: the room's extent along each Manhattan axis (outermost wall
lines of the polygon) and, when both polygons have 4 walls, each wall length.
Gate reference: |diff| <= max(1 cm, 0.5 % of length).
"""
import json
from pathlib import Path
import numpy as np
from roomscan.lidar_pipeline import build_plan

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e14_lidar_v0/out"; OUT.mkdir(parents=True, exist_ok=True)
CAP = ROOT / "single_scan_with_ceiling"
plans = {k: build_plan(CAP, visit_rank=k) for k in (0, 1)}
def dims(plan, dbg):
    out = {}
    for r in plan.rooms:
        lab = int(r.id[1:]); yaw = np.radians(dbg[f"room{lab}"]["yaw_deg"])
        P = np.array(r.polygon); P = np.stack([P[:, 0], -P[:, 1]], 1)        # back to world (x, z)
        c, s = np.cos(yaw), np.sin(yaw); uv = P @ np.array([[c, s], [-s, c]]).T
        out[r.id] = {"extent_u": float(np.ptp(uv[:, 0])), "extent_v": float(np.ptp(uv[:, 1])), "n_walls": len(r.wall_ids)}
    return out
d0, d1 = dims(*plans[0]), dims(*plans[1])
rows = []
for rid in d0.keys() & d1.keys():
    for k in ("extent_u", "extent_v"):
        a, b = d0[rid][k], d1[rid][k]; diff = abs(a - b); tol = max(0.01, 0.005 * 0.5 * (a + b))
        rows.append({"room": rid, "dim": k, "visit0_m": round(a, 4), "visit1_m": round(b, 4), "diff_mm": round(1000 * diff, 1), "tol_mm": round(1000 * tol, 1), "pass": diff <= tol,
                     "n_walls": [d0[rid]["n_walls"], d1[rid]["n_walls"]]})
res = {"label": "within-capture visit-to-visit proxy (not the repeatability gate)", "rows": sorted(rows, key=lambda r: (r["room"], r["dim"])),
       "pass_frac": float(np.mean([r["pass"] for r in rows])) if rows else None,
       "median_diff_mm": float(np.median([r["diff_mm"] for r in rows])) if rows else None}
(OUT / "visit_repeat.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))

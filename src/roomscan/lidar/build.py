"""Assemble the Plan (schema v0.1) for a Stray LiDAR capture.

Intervals are explicitly UNCALIBRATED (Interval.calibrated = False): widths come from the
provisional error budget below, derived from GT-free consistency (e18) and placeholders for terms
only laser GT can measure. They are replaced by empirical calibration once laser GT exists; the
evaluator's calibration_residuals are the input for that.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from roomscan.assemble import OpeningInput, RoomInput, assemble
from roomscan.lidar import layout, openings as OP, roomgeo, rooms as RM
from roomscan.model import CaptureInfo, Plan
from roomscan.stray import StrayCapture

# Provisional 1-sigma budget (metres or relative)
SIG_FACE = 0.007        # one wall-face position: e18 V6 half-to-half median 9.8 mm / sqrt(2)
SIG_SCALE = 0.004       # LiDAR/VIO scale, relative; unknown without laser GT (pseudo-GT disagrees ~3 %, e08)
SIG_LEVEL = 0.005       # floor or ceiling plane level
SIG_JAMB = 0.007        # one jamb face
# face position sigma by evidence: measured from interior-facing points / line from topology only / never observed
FACE_SIGMA = {"close": SIG_FACE, "all": SIG_FACE, "unmeasured": 0.03, "inferred": 0.10}
METHOD = "lidar_v1_uncal"


def face_terms(sources: list[str], st) -> list[dict]:
    """Per face: its measurement sigma by evidence ('inferred' replaces it for never-observed faces)
    and the stability excess over that measurement noise (quality.py)."""
    out = []
    for src, s in zip(sources, st.sigma_m):
        base = FACE_SIGMA[src]
        t = {"inferred" if src == "inferred" else "measurement": base}
        excess = float(np.sqrt(max(0.0, s ** 2 - base ** 2)))
        if excess > 0:
            t["stability"] = excess
        out.append(t)
    return out


def _to_plan_xy(P_xz):
    return np.stack([P_xz[:, 0], -P_xz[:, 1]], 1)


def build(cap_dir: str | Path, placement: str = "arkit_yaw", ladder: bool = True,
          seg_pad: tuple[float, float] = (RM.SEG_PAD_M, RM.SEG_PAD_M)) -> tuple[Plan, dict]:
    """ladder=False reproduces v1.0 (rooms whose topology fails are dropped instead of escalated);
    seg_pad=(0.3, 0.0) reproduces the segmentation before the border-erosion fix."""
    cap = StrayCapture.load(cap_dir)
    c = RM.prepare(cap, seg_pad=seg_pad)
    seg = c.seg
    debug = {"high_observed": seg.high_observed, "n_labels": int(seg.labels.max()), "ladder": ladder, "rooms": {}}
    geo = {}
    for label in range(1, seg.labels.max() + 1):
        g = roomgeo.measure(c, label, ladder)
        if isinstance(g, str):
            debug["rooms"][label] = g
            continue
        geo[label] = g
        debug["rooms"][label] = {"step": g.step, "n_walls": len(g.walls.poly_uv), "wall_sources": g.walls.source,
                                 "visits": len(g.cloud.visits), "unaligned_visits": sum(1 for a in g.cloud.alignment if not a.get("aligned")),
                                 "ceiling_points": g.n_ceil, "alignment": g.cloud.alignment}
    stab = {}
    for label, g in geo.items():
        stab[label] = roomgeo.stability(g)
        debug["rooms"][label]["stability"] = {"sigma_m": [round(x, 4) for x in stab[label].sigma_m],
                                              "missing_frac": stab[label].missing_frac, "runs_without_outline": stab[label].runs_without_outline,
                                              "detail": stab[label].detail}
    placed = layout.place({r: g.walls.poly_uv @ g.Rm for r, g in geo.items()}, {r: g.walls.yaw for r, g in geo.items()}, placement)

    rooms, openings = [], []
    for label, g in geo.items():
        src = g.walls.source
        rooms.append(RoomInput(
            id=f"r{label}", polygon=_to_plan_xy(placed[label]),
            face_sigma=[FACE_SIGMA[x] for x in src], face_status=["inferred" if x == "inferred" else "measured" for x in src],
            face_terms=face_terms(src, stab[label]),
            scale_sigma_rel=SIG_SCALE, ceiling_h=g.ceil_h, ceiling_sigma=float(np.hypot(SIG_LEVEL, SIG_LEVEL)),
            method=METHOD, placement_sigma_m=layout.PLACEMENT_SIGMA_M))
    for k, d in enumerate(OP.detect(seg)):
        a, b = d.rooms
        if a not in geo or b not in geo:
            continue
        widths, edge_a = [], None
        for r in (a, b):
            g = geo[r]
            m = OP.measure(g.walls.poly_uv, g.Rm, d.centre_xz, g.P, g.N, g.floor_y)
            if m is not None and m.width is not None:
                widths.append(m.width)
            if r == a and m is not None:
                edge_a = m.edge
        if widths:
            o = OpeningInput(f"o{k}", "doorway", [f"r{a}", f"r{b}"], edge_a or 0, float(np.mean(widths)),
                             float(np.hypot(SIG_JAMB, SIG_JAMB) / np.sqrt(len(widths))), "lidar_jamb_gap_uncal",
                             0.8 if len(widths) == 2 else 0.6, {"jambs": float(np.hypot(SIG_JAMB, SIG_JAMB))})
        else:   # coarse grid width only: wide interval, low confidence
            o = OpeningInput(f"o{k}", "doorway", [f"r{a}", f"r{b}"], edge_a or 0, d.coarse_width, 0.05, "lidar_grid_doorway_uncal", 0.3)
        openings.append(o)
        debug.setdefault("openings", []).append({"id": o.id, "rooms": [a, b], "widths_from_rooms": widths, "coarse": d.coarse_width})
    flags = []
    if not seg.high_observed:
        flags.append("upper_walls_not_observed: almost no wall surface above 1.9 m was seen; room separation and wall "
                     "existence need the upper walls and ceiling in view (re-capture sweeping each wall's top edge)")
    for label, why in debug["rooms"].items():
        if isinstance(why, str):
            flags.append(f"room_region_{label}_{why}: a free-space region was found but no room geometry could be built")
    for label, g in geo.items():
        if g.ceil_h is None:
            flags.append(f"r{label}_ceiling_not_observed: too few ceiling points ({g.n_ceil}); sweep the ceiling")
        if "unmeasured" in g.walls.source:
            flags.append(f"r{label}_walls_unmeasured: {g.walls.source.count('unmeasured')} wall(s) placed from topology only (face sigma {FACE_SIGMA['unmeasured']} m)")
        if stab[label].unstable_topology:
            flags.append(f"r{label}_topology_unstable: the room outline changes under small irrelevant perturbations "
                         f"({stab[label].runs_without_outline}/{stab[label].n_runs} runs without an outline); intervals widened")
        elif stab[label].fragile_topology:
            flags.append(f"r{label}_topology_fragile: {stab[label].runs_without_outline}/{stab[label].n_runs} perturbed runs "
                         "(e.g. half the points) found no outline; wall existence depends on coverage — capture more of each wall's top edge")
        if g.step == "reregistered":
            flags.append(f"r{label}_visits_reregistered: wall faces of separate visits disagreed beyond ICP range; visits were re-registered "
                         "(e22) before the room's walls could be found")
        if g.step == "inferred":
            flags.append(f"r{label}_walls_inferred: {g.walls.source.count('inferred')} wall(s) never observed at structural height; placed at the "
                         f"observed-floor boundary, status inferred (face sigma {FACE_SIGMA['inferred']} m); capture each wall's top edge")
    plan = assemble("lidar", rooms, openings, CaptureInfo(tier="lidar", app="Stray Scanner", inputs=[str(cap_dir)]), METHOD, flags,
                    f"room-local geometry (per-room registration of visits); placement={placement}",
                    params={"placement": placement, "ladder": ladder, "seg_pad": list(seg_pad)})
    return plan, debug

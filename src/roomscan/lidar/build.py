"""Assemble the Plan (schema v0.1) for a Stray LiDAR capture.

Intervals are explicitly UNCALIBRATED (Interval.calibrated = False): widths come from the
provisional error budget below, derived from GT-free consistency (e18) and placeholders for terms
only laser GT can measure. They are replaced by empirical calibration once laser GT exists; the
evaluator's calibration_residuals are the input for that.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from roomscan import geometry as G
from roomscan.lidar import layout, levels as LV, openings as OP, rooms as RM, walls as WL
from roomscan.model import (Adjacency, CaptureInfo, Interval, Measurement, Opening, Plan, Plane, Room,
                            Surface)
from roomscan.stray import StrayCapture

Z90 = 1.645
# Provisional 1-sigma budget (metres or relative)
SIG_FACE = 0.007        # one wall-face position: e18 V6 half-to-half median 9.8 mm / sqrt(2)
SIG_SCALE = 0.004       # LiDAR/VIO scale, relative; unknown without laser GT (pseudo-GT disagrees ~3 %, e08)
SIG_LEVEL = 0.005       # floor or ceiling plane level
SIG_JAMB = 0.007        # one jamb face
METHOD = "lidar_v1_uncal"


def meas(v, sigma, unit, method, budget=None):
    if v is None:
        return Measurement(value=None, unit=unit, interval=None, status="not_observed", method=method)
    return Measurement(value=float(v), unit=unit, method=method, error_budget=budget or {},
                       interval=Interval(lo=float(v - Z90 * sigma), hi=float(v + Z90 * sigma), level=0.9, calibrated=False))


def _to_plan_xy(P_xz):
    return np.stack([P_xz[:, 0], -P_xz[:, 1]], 1)


def build(cap_dir: str | Path, placement: str = "arkit_yaw") -> tuple[Plan, dict]:
    cap = StrayCapture.load(cap_dir)
    c = RM.prepare(cap)
    seg = c.seg
    centers = seg.grid.centers()
    debug = {"high_observed": seg.high_observed, "n_labels": int(seg.labels.max()), "rooms": {}}
    geo = {}
    for label in range(1, seg.labels.max() + 1):
        rc = RM.room_cloud(c, label)
        if rc is None or len(rc.all.points) < 2000:
            debug["rooms"][label] = "insufficient_points"
            continue
        P, N = np.asarray(rc.all.points), np.asarray(rc.all.normals)
        CP, CN = np.asarray(rc.close.points), np.asarray(rc.close.normals)
        region = centers[G.fill_region(seg.labels, label)]
        other = centers[(seg.labels > 0) & (seg.labels != label)]
        # first pass for walls with a provisional ceiling, then levels inside the polygon
        w = WL.extract(P, N, CP, CN, seg.grid.floor_y, None, region, other)
        if w is None:
            debug["rooms"][label] = "no_polygon"
            continue
        Rm = G._rot(w.yaw)
        floor_y, ceil_y, n_ceil = LV.levels(P, N, w.poly_uv, Rm, seg.grid.floor_y)
        ceil_h = None if ceil_y is None else ceil_y - floor_y
        w = WL.extract(P, N, CP, CN, floor_y, ceil_h, region, other) or w
        geo[label] = {"walls": w, "Rm": G._rot(w.yaw), "floor_y": floor_y, "ceil_h": ceil_h, "n_ceil": n_ceil, "P": P, "N": N,
                      "poly_xz": w.poly_uv @ G._rot(w.yaw)}
        debug["rooms"][label] = {"n_walls": len(w.poly_uv), "wall_sources": w.source, "visits": len(rc.visits),
                                 "unaligned_visits": sum(1 for a in rc.alignment if not a.get("aligned")), "ceiling_points": n_ceil}
    placed = layout.place({r: g["poly_xz"] for r, g in geo.items()}, {r: g["walls"].yaw for r, g in geo.items()}, placement)

    rooms, surfaces, openings, adjacency = [], [], [], []
    for label, g in geo.items():
        rid = f"r{label}"
        poly = _to_plan_xy(placed[label])
        if G.polygon_area(poly) < 0:
            poly = poly[::-1]
        h = g["ceil_h"]
        zt = h if h is not None else 2.4
        wall_ids = []
        for k in range(len(poly)):
            p0, p1 = poly[k], poly[(k + 1) % len(poly)]
            L = float(np.linalg.norm(p1 - p0))
            sig = float(np.hypot(np.sqrt(2) * SIG_FACE, SIG_SCALE * L))
            n2 = np.array([p1[1] - p0[1], -(p1[0] - p0[0])]) / max(L, 1e-9)
            sid = f"{rid}_w{k}"
            wall_ids.append(sid)
            surfaces.append(Surface(
                id=sid, kind="wall", room_id=rid, plane=Plane(normal=(float(n2[0]), float(n2[1]), 0.0), offset=float(-(n2 @ p0))),
                polygon=[(float(p0[0]), float(p0[1]), 0.0), (float(p1[0]), float(p1[1]), 0.0), (float(p1[0]), float(p1[1]), zt), (float(p0[0]), float(p0[1]), zt)],
                area=meas(L * zt, sig * zt, "m2", METHOD),
                length=meas(L, sig, "m", METHOD, {"face_positions": float(np.sqrt(2) * SIG_FACE), "scale": SIG_SCALE * L}),
                height=meas(h, np.hypot(SIG_LEVEL, SIG_LEVEL), "m", METHOD)))
        ring = [(float(x), float(y), 0.0) for x, y in poly]
        area = abs(G.polygon_area(poly))
        perim = float(np.linalg.norm(np.roll(poly, -1, 0) - poly, axis=1).sum())
        sig_area = float(np.hypot(perim * SIG_FACE / 2, 2 * SIG_SCALE * area))
        surfaces.append(Surface(id=f"{rid}_floor", kind="floor", room_id=rid, plane=Plane(normal=(0, 0, 1.0), offset=0.0), polygon=ring,
                                area=meas(area, sig_area, "m2", METHOD)))
        if h is not None:
            surfaces.append(Surface(id=f"{rid}_ceiling", kind="ceiling", room_id=rid, plane=Plane(normal=(0, 0, -1.0), offset=float(h)),
                                    polygon=[(x, y, float(h)) for x, y, _ in ring], area=meas(area, sig_area, "m2", METHOD)))
        rooms.append(Room(id=rid, polygon=[tuple(map(float, p)) for p in poly],
                          floor_area=meas(area, sig_area, "m2", METHOD, {"face_positions": perim * SIG_FACE / 2, "scale": 2 * SIG_SCALE * area}),
                          perimeter=meas(perim, float(np.hypot(np.sqrt(len(poly)) * SIG_FACE, SIG_SCALE * perim)), "m", METHOD),
                          ceiling_height=meas(h, np.hypot(SIG_LEVEL, SIG_LEVEL), "m", METHOD, {"floor_level": SIG_LEVEL, "ceiling_level": SIG_LEVEL}),
                          wall_ids=wall_ids, placement_sigma_m=layout.PLACEMENT_SIGMA_M))

    for k, d in enumerate(OP.detect(seg)):
        a, b = d.rooms
        if a not in geo or b not in geo:
            continue
        widths, edge_a = [], None
        for r in (a, b):
            g = geo[r]
            m = OP.measure(g["walls"].poly_uv, g["Rm"], d.centre_xz, g["P"], g["N"], g["floor_y"])
            if m is not None and m.width is not None:
                widths.append(m.width)
            if r == a and m is not None:
                edge_a = m.edge
        oid = f"o{k}"
        wall = f"r{a}_w{edge_a if edge_a is not None else 0}"
        if widths:
            wv = float(np.mean(widths))
            width = meas(wv, float(np.hypot(SIG_JAMB, SIG_JAMB) / np.sqrt(len(widths))), "m", "lidar_jamb_gap_uncal", {"jambs": float(np.hypot(SIG_JAMB, SIG_JAMB))})
            conf = 0.8 if len(widths) == 2 else 0.6
        else:   # coarse grid width only: wide interval, low confidence
            width = meas(d.coarse_width, 0.05, "m", "lidar_grid_doorway_uncal")
            conf = 0.3
        openings.append(Opening(id=oid, kind="doorway", wall_id=wall, room_ids=[f"r{a}", f"r{b}"], width=width, detection_confidence=conf))
        adjacency.append(Adjacency(room_a=f"r{a}", room_b=f"r{b}", via_opening_id=oid))
        for r in rooms:
            if r.id in (f"r{a}", f"r{b}"):
                r.opening_ids.append(oid)
        debug.setdefault("openings", []).append({"id": oid, "rooms": [a, b], "widths_from_rooms": widths, "coarse": d.coarse_width})
    flags = []
    if not seg.high_observed:
        flags.append("upper_walls_not_observed: almost no wall surface above 1.9 m was seen; room separation and wall "
                     "existence need the upper walls and ceiling in view (re-capture sweeping each wall's top edge)")
    for label, why in debug["rooms"].items():
        if isinstance(why, str):
            flags.append(f"room_region_{label}_{why}: a free-space region was found but no room geometry could be built")
    for label, g in geo.items():
        if g["ceil_h"] is None:
            flags.append(f"r{label}_ceiling_not_observed: too few ceiling points ({g['n_ceil']}); sweep the ceiling")
        if "unmeasured" in g["walls"].source:
            flags.append(f"r{label}_walls_unmeasured: {g['walls'].source.count('unmeasured')} wall(s) placed from topology only")
    fp = sum(r.floor_area.value for r in rooms)
    fp_sig = float(np.sqrt(sum(((r.floor_area.interval.hi - r.floor_area.interval.lo) / (2 * Z90)) ** 2 for r in rooms)))
    plan = Plan(capture=CaptureInfo(tier="lidar", app="Stray Scanner", inputs=[str(cap_dir)]), rooms=rooms, surfaces=surfaces,
                openings=openings, adjacency=adjacency, footprint_area=meas(fp, fp_sig, "m2", METHOD),
                drift_correction=f"room-local geometry (per-room ICP of visits); placement={placement}", quality_flags=flags)
    return plan, debug

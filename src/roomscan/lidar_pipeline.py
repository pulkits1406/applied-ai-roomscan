"""LiDAR tier, prototype v0: Stray capture -> Plan JSON.

    uv run python -m roomscan.lidar_pipeline <stray_dir> <out_dir> [--labels-from plan_or_npz]

Steps: fuse every STEP-th frame -> segment rooms (2-D) -> per room, fuse only its longest
continuous visit (other visits rigidly ICP-aligned to it) -> Manhattan wall lines -> rectilinear
polygon -> floor/ceiling levels -> geometric doorways -> Plan.

Intervals in v0 come from a provisional, UNCALIBRATED error model (method suffix
"_uncal_v0"); they must be replaced by intervals calibrated on laser ground truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from roomscan import geometry as G
from roomscan.model import (Adjacency, CaptureInfo, Interval, Measurement, Opening, Plan, Plane, Room,
                            Surface)
from roomscan.stray import StrayCapture

STEP = 5
Z90 = 1.645
# Provisional 1-sigma terms (metres / relative). Plane offset noise from e03 short-baseline
# consistency (~6-8 mm per surface); scale term placeholder for unknown LiDAR/VIO scale bias.
SIG_PLANE = 0.006
SIG_SCALE = 0.004


def _meas(v, sigma, unit, method, budget=None, status="measured"):
    if v is None:
        return Measurement(value=None, unit=unit, interval=None, status="not_observed", method=method)
    return Measurement(value=float(v), unit=unit, interval=Interval(lo=float(v - Z90 * sigma), hi=float(v + Z90 * sigma), level=0.9),
                       status=status, method=method, error_budget=budget or {})


def _visits(room_of_frame: np.ndarray, t: np.ndarray, label: int, gap_s=1.0):
    """Contiguous time spans with the camera inside room `label` (gaps < gap_s bridged)."""
    idx = np.nonzero(room_of_frame == label)[0]
    if len(idx) == 0:
        return []
    spans, s = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if t[b] - t[a] > gap_s:
            spans.append((s, a)); s = b
    spans.append((s, idx[-1]))
    return sorted(spans, key=lambda ab: t[ab[1]] - t[ab[0]], reverse=True)


def build_plan(cap_dir: str | Path, T: np.ndarray | None = None, drift: str = "none", visit_rank: int = 0) -> tuple[Plan, dict]:
    """visit_rank selects which visit (0 = longest) supplies each room's wall geometry."""
    cap = StrayCapture.load(cap_dir)
    T = cap.T_wc() if T is None else T
    t = cap.timestamps
    rows = range(0, len(cap), STEP)
    pc = G.fuse(cap, T, rows)
    P, N = np.asarray(pc.points), np.asarray(pc.normals)
    seg = G.segment_rooms(P, N)
    debug = {"high_observed": seg.high_observed, "n_labels": int(seg.labels.max())}
    # camera -> room label (nearest labelled cell within 0.3 m)
    lab_d = ndi.grey_dilation(seg.labels, size=(7, 7))
    cij = np.clip(seg.grid.cell(T[:, [0, 2], 3]), 0, np.array(seg.grid.shape) - 1)
    room_of_frame = np.where(seg.labels[cij[:, 0], cij[:, 1]] > 0, seg.labels[cij[:, 0], cij[:, 1]], lab_d[cij[:, 0], cij[:, 1]])

    rooms, surfaces, openings, adjacency, geoms = [], [], [], [], {}
    centers = seg.grid.centers()
    for label in range(1, seg.labels.max() + 1):
        vis = _visits(room_of_frame, t, label)
        if not vis:
            continue
        if len(vis) <= visit_rank:
            continue
        a, b = vis[visit_rank]
        vrows = list(range(a, b + 1, 3))
        if len(vrows) < 10:
            continue
        rpc = G.fuse(cap, T, vrows, max_depth=5.0)
        RP, RN = np.asarray(rpc.points), np.asarray(rpc.normals)
        region = seg.labels == label
        near = ndi.binary_dilation(region, iterations=int(0.5 / G.GRID))
        ij = np.clip(seg.grid.cell(RP[:, [0, 2]]), 0, np.array(seg.grid.shape) - 1)
        keep = near[ij[:, 0], ij[:, 1]]
        RP, RN = RP[keep], RN[keep]
        if len(RP) < 2000:
            continue
        yaw = G.manhattan_yaw(RN)
        Rm = G._rot(yaw)
        # Wall EXISTENCE from all visits (structural support needs observations above 2 m, which
        # one visit may lack); wall POSITION from the single visit (drift-consistent within it).
        gij = np.clip(seg.grid.cell(P[:, [0, 2]]), 0, np.array(seg.grid.shape) - 1)
        gk = near[gij[:, 0], gij[:, 1]]
        GP, GN = P[gk], N[gk]
        GUV, GNUV, GH = GP[:, [0, 2]] @ Rm.T, GN[:, [0, 2]] @ Rm.T, GP[:, 1] - seg.grid.floor_y
        UV = RP[:, [0, 2]] @ Rm.T
        NUV = RN[:, [0, 2]] @ Rm.T
        us, _ = G.wall_lines(GUV, GNUV, GH, 0)
        vs, _ = G.wall_lines(GUV, GNUV, GH, 1)
        us = G.refine_lines(us, UV, NUV, 0)
        vs = G.refine_lines(vs, UV, NUV, 1)
        reg_uv = centers[G.fill_region(seg.labels, label)] @ Rm.T
        struct = (np.abs(GN[:, 1]) < 0.3) & (GH > G.H_STRUCT)
        other = centers[(seg.labels > 0) & (seg.labels != label)] @ Rm.T
        poly_uv = G.room_polygon(reg_uv, us, vs, structural_uv=GUV[struct], other_uv=other)
        if poly_uv is None or len(poly_uv) < 4:
            debug[f"room{label}"] = "no_polygon"
            continue
        poly_xz = poly_uv @ Rm  # back to world (x, z)
        # interior points for floor/ceiling: inside polygon shrunk by 0.15 m
        from shapely.geometry import Point, Polygon as SPoly
        # floor/ceiling from ALL visits (vertical drift is small; the longest visit may not
        # include the ceiling sweep)
        from shapely import contains_xy
        shp = SPoly(poly_uv).buffer(-0.15)
        guv = P[:, [0, 2]] @ Rm.T
        Pin = P[contains_xy(shp, guv[:, 0], guv[:, 1])]
        floor_y, ceil_y, csup = G.floor_ceiling(Pin, seg.grid.floor_y)
        if csup < 0.03:
            ceil_y = None
        geoms[label] = G.RoomGeom(label, yaw, poly_xz, floor_y, ceil_y, csup, n_points=len(RP))
        rid = f"r{label}"
        height = None if ceil_y is None else ceil_y - floor_y
        # property-frame polygon (x, y) = (world x, -world z); world (x,z) polygon is CCW in (u,v)
        poly_xy = np.stack([poly_xz[:, 0], -poly_xz[:, 1]], 1)
        if G.polygon_area(poly_xy) < 0:
            poly_xy = poly_xy[::-1]
        wall_ids = []
        for k in range(len(poly_xy)):
            p0, p1 = poly_xy[k], poly_xy[(k + 1) % len(poly_xy)]
            L = float(np.linalg.norm(p1 - p0))
            sig = float(np.hypot(np.sqrt(2) * SIG_PLANE, SIG_SCALE * L))
            n2 = np.array([p1[1] - p0[1], -(p1[0] - p0[0])]) / max(L, 1e-9)
            zt = height if height is not None else 2.4
            sid = f"{rid}_w{k}"
            wall_ids.append(sid)
            surfaces.append(Surface(
                id=sid, kind="wall", room_id=rid,
                plane=Plane(normal=(float(n2[0]), float(n2[1]), 0.0), offset=float(-(n2 @ p0))),
                polygon=[(float(p0[0]), float(p0[1]), 0.0), (float(p1[0]), float(p1[1]), 0.0), (float(p1[0]), float(p1[1]), zt), (float(p0[0]), float(p0[1]), zt)],
                area=_meas(L * zt, sig * zt, "m2", "lidar_wall_rect_uncal_v0"),
                length=_meas(L, sig, "m", "lidar_wall_lines_uncal_v0", {"plane_offsets": np.sqrt(2) * SIG_PLANE, "scale": SIG_SCALE * L}),
                height=_meas(height, np.sqrt(2) * SIG_PLANE, "m", "lidar_floor_ceiling_uncal_v0")))
        area = abs(G.polygon_area(poly_xy))
        perim = float(sum(np.linalg.norm(np.roll(poly_xy, -1, 0) - poly_xy, axis=1)))
        sig_area = float(np.hypot(perim * SIG_PLANE, 2 * SIG_SCALE * area))
        rooms.append(Room(
            id=rid, polygon=[tuple(map(float, p)) for p in poly_xy],
            floor_area=_meas(area, sig_area, "m2", "lidar_polygon_uncal_v0"),
            perimeter=_meas(perim, float(np.hypot(len(poly_xy) * SIG_PLANE, SIG_SCALE * perim)), "m", "lidar_polygon_uncal_v0"),
            ceiling_height=_meas(height, np.sqrt(2) * SIG_PLANE + 0.002, "m", "lidar_floor_ceiling_uncal_v0"),
            wall_ids=wall_ids))
        debug[f"room{label}"] = {"visit_s": round(float(t[b] - t[a]), 1), "n_visits": len(vis), "yaw_deg": round(np.degrees(yaw), 2),
                                 "n_walls": len(poly_xy), "area_m2": round(area, 2), "ceiling_support": round(csup, 3)}
    # geometric doorways: connected doorway cells touching exactly two rooms
    dlab, nd = ndi.label(seg.doorway)
    room_ids = {r.id for r in rooms}
    for k in range(1, nd + 1):
        cells = dlab == k
        ring = ndi.binary_dilation(cells, iterations=3) & ~cells
        touching = sorted({int(x) for x in np.unique(seg.labels[ring]) if x > 0 and f"r{int(x)}" in room_ids})
        if len(touching) != 2:
            continue
        pts = centers[cells]
        if len(pts) < 3:
            continue
        ev, evec = np.linalg.eigh(np.cov(pts.T))
        along = pts @ evec[:, -1]
        width = float(np.ptp(along) + G.GRID)
        if not 0.5 <= width <= 2.5:
            continue
        ra, rb = f"r{touching[0]}", f"r{touching[1]}"
        wall = next(s.id for s in surfaces if s.room_id == ra)  # v0: wall assignment not resolved
        oid = f"o{k}"
        openings.append(Opening(id=oid, kind="doorway", wall_id=wall, room_ids=[ra, rb],
                                width=_meas(width, 0.03, "m", "lidar_grid_doorway_uncal_v0"), detection_confidence=0.5))
        adjacency.append(Adjacency(room_a=ra, room_b=rb, via_opening_id=oid))
        for r in rooms:
            if r.id in (ra, rb):
                r.opening_ids.append(oid)
    fp = sum(r.floor_area.value for r in rooms)
    plan = Plan(capture=CaptureInfo(tier="lidar", app="Stray Scanner", inputs=[str(cap_dir)]),
                rooms=rooms, surfaces=surfaces, openings=openings, adjacency=adjacency,
                footprint_area=_meas(fp, float(np.sqrt(sum(r.floor_area.error_budget.get("x", 0) for r in rooms)) + 0.02 * fp), "m2", "sum_rooms_uncal_v0"),
                drift_correction=drift)
    return plan, debug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture"); ap.add_argument("out")
    a = ap.parse_args()
    plan, dbg = build_plan(a.capture)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(plan.model_dump_json(indent=1))
    (out / "debug.json").write_text(json.dumps(dbg, indent=1))
    print(json.dumps(dbg, indent=1))


if __name__ == "__main__":
    main()

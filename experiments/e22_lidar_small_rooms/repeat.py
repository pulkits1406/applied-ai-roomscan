"""Production-code version of the e18 protocol (GT-free): topology + wall lengths + ceilings per
room from all data, then wall lengths re-measured from each of two disjoint halves (2 s chunks
alternating inside every visit), each half built through the same production registration and
ICP-aligned onto the full room cloud. Also records which rooms produce a polygon at all.

    uv run python experiments/e22_lidar_small_rooms/repeat.py <capture> <config>[,<config>...]
configs: v1_0 (icp + V6 rules = current production), reg (verified registration + V6 rules),
         rules (icp + window 3 cm + inferred sides), reg_rules (both), ladder (production
         roomgeo.measure: v1_0, escalating only for rooms whose topology fails)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from roomscan import geometry as G
from roomscan.lidar import levels as LV, roomgeo, rooms as RM, walls as WL
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/e22_lidar_small_rooms/out"
reg = o3d.pipelines.registration
CHUNK_S = 2.0
NEW_RULES = WL.Rules(face_window=0.03, infer_missing=True)
CONFIGS = {"v1_0": ("icp", WL.V1_0), "reg": ("coarse", WL.V1_0), "rules": ("icp", NEW_RULES), "reg_rules": ("coarse", NEW_RULES), "ladder": (None, None)}
tol = lambda L: max(0.01, 0.005 * L)


def geometry(c, label, rc, rules):
    seg = c.seg
    centers = seg.grid.centers()
    P, N = np.asarray(rc.all.points), np.asarray(rc.all.normals)
    CP, CN = np.asarray(rc.close.points), np.asarray(rc.close.normals)
    region = centers[G.fill_region(seg.labels, label)]
    other = centers[(seg.labels > 0) & (seg.labels != label)]
    w = WL.extract(P, N, CP, CN, seg.grid.floor_y, None, region, other, rules)
    if w is None:
        return None
    Rm = G._rot(w.yaw)
    fy, cy, _ = LV.levels(P, N, w.poly_uv, Rm, seg.grid.floor_y)
    ch = None if cy is None else cy - fy
    w = WL.extract(P, N, CP, CN, fy, ch, region, other, rules) or w
    return {"walls": w, "floor_y": fy, "ceil_h": ch}


def halves(c, spans):
    t = c.cap.timestamps
    order = sorted(spans, key=lambda ab: t[ab[1]] - t[ab[0]], reverse=True)
    A, B, k = [], [], 0
    for a, b in order:
        ra, rb, s0 = [], [], a
        while s0 <= b:
            e0 = max(min(int(np.searchsorted(t, t[s0] + CHUNK_S)), b + 1) - 1, s0)
            (ra if k % 2 == 0 else rb).extend(range(s0, e0 + 1, RM.ROOM_STEP)); k += 1; s0 = e0 + 1
        if ra: A.append(ra)
        if rb: B.append(rb)
    return A, B


def to_full(pc, full):
    r = reg.registration_icp(pc, full, 0.05, np.eye(4), reg.TransformationEstimationPointToPlane())
    r = reg.registration_icp(pc, full, 0.02, r.transformation, reg.TransformationEstimationPointToPlane())
    return r.transformation, r.fitness


def remeasure(g, rc_half, T):
    w = g["walls"]; Rm = G._rot(w.yaw)
    pa = o3d.geometry.PointCloud(rc_half.all).transform(T); pc = o3d.geometry.PointCloud(rc_half.close).transform(T)
    P, N, CP, CN = (np.asarray(x) for x in (pa.points, pa.normals, pc.points, pc.normals))
    UV, NUV, H = P[:, [0, 2]] @ Rm.T, N[:, [0, 2]] @ Rm.T, P[:, 1] - g["floor_y"]
    CUV, CNUV, CH = CP[:, [0, 2]] @ Rm.T, CN[:, [0, 2]] @ Rm.T, CP[:, 1] - g["floor_y"]
    top = (g["ceil_h"] - 0.1) if g["ceil_h"] else 2.6
    poly = w.poly_uv; Q = poly.copy(); n = len(poly)
    for k in range(n):
        cpos, s = WL.edge_position(poly, k, CUV, CNUV, CH, top)
        if cpos is None or s < WL.CLOSE_MIN_SUPPORT:
            cpos, s = WL.edge_position(poly, k, UV, NUV, H, top)
        ax = WL.edge_axis(poly[k], poly[(k + 1) % n])
        if cpos is not None:
            Q[k][ax] = cpos; Q[(k + 1) % n][ax] = cpos
    fy, cy, _ = LV.levels(P, N, poly, Rm, g["floor_y"])
    return np.linalg.norm(np.roll(Q, -1, 0) - Q, axis=1), (None if cy is None else cy - fy)


def main(capture="single_scan_with_ceiling", configs="v1_0,reg,rules,reg_rules"):
    t0 = time.time()
    c = RM.prepare(StrayCapture.load(ROOT / capture))
    res, summary = {}, {}
    for name in configs.split(","):
        registration, rules = CONFIGS[name]
        rows = {}
        for label in range(1, c.seg.labels.max() + 1):
            spans = RM.visits(c, label)
            if not spans:
                continue
            if name == "ladder":
                rg = roomgeo.measure(c, label)
                if isinstance(rg, str):
                    rows[label] = {"status": rg}; continue
                rc, reg_used = rg.cloud, rg.registration
                g = {"walls": rg.walls, "floor_y": rg.floor_y, "ceil_h": rg.ceil_h}
            else:
                rc, reg_used = RM.room_cloud(c, label, registration=registration), registration
                if len(rc.all.points) < 2000:
                    rows[label] = {"status": "insufficient_points"}; continue
                g = geometry(c, label, rc, rules)
                if g is None:
                    rows[label] = {"status": "no_polygon"}; continue
            w = g["walls"]
            full_len = np.linalg.norm(np.roll(w.poly_uv, -1, 0) - w.poly_uv, axis=1)
            row = {"status": "ok", "n_walls": len(w.poly_uv), "sources": w.source, "area_m2": round(abs(G.polygon_area(w.poly_uv)), 3),
                   "ceiling_m": None if g["ceil_h"] is None else round(g["ceil_h"], 4), "lengths_m": full_len.round(4).tolist()}
            A, B = halves(c, spans)
            hs = []
            for rows_h in (A, B):
                rch = RM.room_cloud_rows(c, label, rows_h, registration=reg_used)
                T, fit = to_full(rch.all, rc.all)
                hs.append(remeasure(g, rch, T) + (fit,))
            d = np.abs(hs[0][0] - hs[1][0])
            row.update({"wall_diffs_mm": (1000 * d).round(1).tolist(),
                        "wall_pass": [bool(x <= tol(0.5 * (p + q))) for x, p, q in zip(d, hs[0][0], hs[1][0])],
                        "measured_wall": [s in ("close", "all") for s in w.source],
                        "ceiling_diff_mm": round(1000 * abs(hs[0][1] - hs[1][1]), 1) if hs[0][1] and hs[1][1] else None})
            rows[label] = row
            print(name, label, {k: row[k] for k in ("n_walls", "area_m2", "ceiling_m", "wall_diffs_mm", "ceiling_diff_mm")}, flush=True)
        ok = [r for r in rows.values() if r["status"] == "ok"]
        wd = [x for r in ok for x in r["wall_diffs_mm"]]; wp = [x for r in ok for x in r["wall_pass"]]
        cd = [r["ceiling_diff_mm"] for r in ok if r["ceiling_diff_mm"] is not None]
        summary[name] = {"rooms_with_polygon": sorted(int(k) for k, r in rows.items() if r["status"] == "ok"),
                         "rooms_failed": {int(k): r["status"] for k, r in rows.items() if r["status"] != "ok"},
                         "n_walls": len(wd), "wall_diff_mm_median": round(float(np.median(wd)), 1) if wd else None,
                         "wall_diff_mm_p90": round(float(np.percentile(wd, 90)), 1) if wd else None,
                         "wall_pass_frac": round(float(np.mean(wp)), 3) if wp else None,
                         "ceiling_pass_frac_1cm": round(float(np.mean([x <= 10 for x in cd])), 3) if cd else None,
                         "ceiling_diff_mm_median": round(float(np.median(cd)), 1) if cd else None}
        res[name] = rows
        print(name, json.dumps(summary[name]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"label": "GT-free, production code: topology from all data; wall lengths from 2 s chunk halves", "capture": capture,
           "sec": round(time.time() - t0, 1), "summary": summary, "rooms": res}
    (OUT / f"repeat_{capture}_{configs.replace(',', '_')}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:])

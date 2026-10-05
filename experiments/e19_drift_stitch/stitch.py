"""Drift ablation: (A) raw ARKit room placement vs (B) plane-anchored stitch. GT-free.

Rooms are built twice from DISJOINT observation sets of the same capture (e18 visit split), so
drift differs between the two halves. For each half:
  A raw:       each room polygon placed where its own (longest) visit put it.
  B anchored:  every room rotated about its centroid to the common Manhattan yaw (circular median
               of room yaws mod 90 deg); then per-room translations + one shared wall thickness tau
               solved by least squares from
                 shared wall:  opposite-facing parallel faces of adjacent rooms are tau apart
                 doorway:      the gap each room observes in its own wall points lines up along the wall
               with the largest room fixed.
Metrics:
  placement disagreement: per-room centroid distance between halves after a best rigid (2-D)
     alignment of the whole layout -> drift that a method failed to remove shows up here
  shared-wall gap spread: std of face-to-face gaps over shared walls (real walls are ~uniform)
  pairwise room overlap area
Footprint figure: raw vs anchored for both halves.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import Polygon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e18_lidar_repeat"))
import roomgeo as RG  # noqa: E402
from roomscan import geometry as G  # noqa: E402
from roomscan.stray import StrayCapture  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/e19_drift_stitch/out"
OUT.mkdir(parents=True, exist_ok=True)


def rot2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def world_poly(g):
    """Room polygon in world (x, z): poly_uv @ Rm with Rm = G._rot(yaw), or an explicit placement."""
    if "_poly_world" in g:
        return g["_poly_world"]
    return g["poly_uv"] @ G._rot(g["yaw"])


def common_yaw(yaws):
    a = np.mod(np.array(yaws), np.pi / 2) * 4                      # 90-deg periodic -> circle
    m = np.arctan2(np.sin(a).mean(), np.cos(a).mean()) / 4
    return float(np.mod(m, np.pi / 2))


def snap_yaw(P, yaw, target):
    """Rotate world polygon about its centroid so its Manhattan yaw equals target (shortest turn)."""
    d = np.mod(target - yaw + np.pi / 4, np.pi / 2) - np.pi / 4
    c = P.mean(0)
    # world (x,z) -> room (u,v) is [[c, s], [-s, c]]; increasing yaw rotates the room frame by -yaw
    return (P - c) @ rot2(-d).T + c, d


def axis_edges(P, yaw):
    """Edges of a polygon (in the common Manhattan frame) as (axis, coord, lo, hi, inward_sign)."""
    R = G._rot(yaw); Q = P @ R.T
    ccw = G.polygon_area(Q) > 0
    out = []
    for k in range(len(Q)):
        a, b = Q[k], Q[(k + 1) % len(Q)]
        axis = 0 if abs(b[0] - a[0]) < abs(b[1] - a[1]) else 1    # edge perpendicular to axis
        d = b - a
        inward = np.array([-d[1], d[0]]) if ccw else np.array([d[1], -d[0]])
        lo, hi = sorted([a[1 - axis], b[1 - axis]])
        out.append((axis, float(0.5 * (a[axis] + b[axis])), float(lo), float(hi), int(np.sign(inward[axis]))))
    return out


def shared_walls(rooms, yaw, max_gap=0.45, min_overlap=0.3):
    """Pairs of opposite-facing parallel edges of different rooms, close and overlapping."""
    E = {r: axis_edges(P, yaw) for r, P in rooms.items()}
    pairs = []
    keys = sorted(rooms)
    for i, ra in enumerate(keys):
        for rb in keys[i + 1:]:
            for ea in E[ra]:
                for eb in E[rb]:
                    if ea[0] != eb[0] or ea[4] == eb[4]:
                        continue
                    gap = (eb[1] - ea[1]) * (-ea[4])          # distance from A's face outward to B's face
                    ov = min(ea[3], eb[3]) - max(ea[2], eb[2])
                    if 0.0 < gap < max_gap and ov > min_overlap:
                        pairs.append({"a": ra, "b": rb, "axis": ea[0], "sign_a": ea[4], "gap": float(gap), "overlap": float(ov),
                                      "lo": max(ea[2], eb[2]), "hi": min(ea[3], eb[3]), "coord_a": ea[1], "coord_b": eb[1]})
    return pairs


def doorway_centre(P, N, H, yaw, wall, side, tol=0.06):
    """Along-wall centre of the largest gap in this room's own mid-height wall points on the shared
    face (common Manhattan frame), restricted to the shared span. None if no gap of door size."""
    R = G._rot(yaw); UV = P[:, [0, 2]] @ R.T; NUV = N[:, [0, 2]] @ R.T
    ax = wall["axis"]; o = 1 - ax
    c = wall["coord_a"] if side == "a" else wall["coord_b"]
    sgn = wall["sign_a"] if side == "a" else -wall["sign_a"]
    m = (np.abs(UV[:, ax] - c) < tol) & (sgn * NUV[:, ax] > 0.7) & (H > 0.3) & (H < 1.8)
    lo, hi = wall["lo"], wall["hi"]
    bins = np.arange(lo, hi + 0.02, 0.02)
    if len(bins) < 10:
        return None
    occ = np.histogram(UV[m, o], bins=bins)[0] > 0
    gaps, s = [], None
    for k, f in enumerate(occ):
        if not f and s is None:
            s = k
        if f and s is not None:
            gaps.append((s, k)); s = None
    if s is not None:
        gaps.append((s, len(occ)))
    gaps = [(a, b) for a, b in gaps if 0.55 <= (b - a) * 0.02 <= 1.3 and a > 0 and b < len(occ)]   # door-sized, bounded both sides
    if not gaps:
        return None
    a, b = max(gaps, key=lambda g: g[1] - g[0])
    return float(bins[a] + (b - a) * 0.01)


def solve_translations(rooms, walls, doors, fixed):
    """Least squares over per-room 2-D translations (common Manhattan frame) + wall thickness tau."""
    keys = sorted(rooms); idx = {r: i for i, r in enumerate(keys)}
    n = 2 * len(keys) + 1
    A, b = [], []
    for w in walls:
        row = np.zeros(n); ax = w["axis"]
        # (coord_b + t_b) - (coord_a + t_a) = -sign_a * tau
        row[2 * idx[w["b"]] + ax] += 1; row[2 * idx[w["a"]] + ax] -= 1; row[-1] = w["sign_a"]
        A.append(row); b.append(w["coord_a"] - w["coord_b"])
    for d in doors:
        row = np.zeros(n); o = 1 - d["axis"]
        row[2 * idx[d["b"]] + o] += 1; row[2 * idx[d["a"]] + o] -= 1
        A.append(3 * row); b.append(3 * (d["centre_a"] - d["centre_b"]))     # doorways weigh more (direct observation)
    for ax in (0, 1):                                                       # gauge: fixed room stays put
        row = np.zeros(n); row[2 * idx[fixed] + ax] = 10; A.append(row); b.append(0.0)
    for r in keys:                                                          # weak prior: small corrections
        for ax in (0, 1):
            row = np.zeros(n); row[2 * idx[r] + ax] = 0.05; A.append(row); b.append(0.0)
    x, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    return {r: x[2 * idx[r]:2 * idx[r] + 2] for r in keys}, float(x[-1])


def full_rooms(cap, T, seg, spans_all, variant):
    """Room shape from ALL visits (shape is shared by both halves; only placement differs)."""
    geo, clouds = {}, {}
    for label, spans in spans_all.items():
        g, _ = RG.room_geometry(cap, T, seg, label, spans, variant)
        if g is None:
            continue
        region = ndi.binary_dilation(seg.labels == label, iterations=int(0.5 / G.GRID))
        pc, _ = RG.aligned_room_cloud(cap, T, spans, region, seg, oriented=variant in ("V2", "V3"))
        geo[label] = g; clouds[label] = pc
    return geo, clouds


def place_half(cap, T, seg, geo_full, clouds_full, spans_half, variant):
    """Where this half's own raw poses put each room: ICP of the half's room cloud onto the full
    room cloud; the full polygon is mapped back with the inverse (planar part) transform."""
    import open3d as o3d
    reg = o3d.pipelines.registration
    geo, clouds, info = {}, {}, {}
    for label, spans in spans_half.items():
        if label not in geo_full:
            continue
        region = ndi.binary_dilation(seg.labels == label, iterations=int(0.5 / G.GRID))
        hc, _ = RG.aligned_room_cloud(cap, T, spans, region, seg, oriented=variant in ("V2", "V3"))
        if len(hc.points) < 2000:
            continue
        F = clouds_full[label]
        if not F.has_normals():
            F.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
        r = reg.registration_icp(hc, F, 0.08, np.eye(4), reg.TransformationEstimationPointToPlane())
        r = reg.registration_icp(hc, F, 0.03, r.transformation, reg.TransformationEstimationPointToPlane())
        Ti = np.linalg.inv(r.transformation)                       # full coords -> half coords
        R2 = Ti[np.ix_([0, 2], [0, 2])]; t2 = Ti[[0, 2], 3]
        u, _, vt = np.linalg.svd(R2); R2 = u @ vt                  # planar rotation part
        g = dict(geo_full[label])
        P = world_poly(g) @ R2.T + t2
        dyaw = np.arctan2(R2[1, 0], R2[0, 0])
        # world (x,z) -> room (u,v) uses G._rot(yaw); rotating the world polygon by R2 changes yaw by -dyaw
        g["yaw"] = float(g["yaw"] - dyaw); g["_poly_world"] = P
        geo[label] = g; clouds[label] = (np.asarray(hc.points), np.asarray(hc.normals))
        info[int(label)] = {"icp_fitness": round(r.fitness, 3), "shift_mm": round(1000 * float(np.linalg.norm(t2 + 0)), 1)}
    return geo, clouds, info


def anchored(geo, clouds, floor_y):
    raw = {r: world_poly(g) for r, g in geo.items()}
    yaw = common_yaw([g["yaw"] for g in geo.values()])
    snapped, turns = {}, {}
    for r, g in geo.items():
        snapped[r], turns[r] = snap_yaw(raw[r], g["yaw"], yaw)
    walls = shared_walls(snapped, yaw)
    doors = []
    R = G._rot(yaw)
    for w in walls:
        cents = []
        for side, r in (("a", w["a"]), ("b", w["b"])):
            P, N = clouds[r]
            # apply the same yaw snap to this room's points (rotation about the raw polygon centroid)
            c = raw[r].mean(0); d = turns[r]
            xz = (P[:, [0, 2]] - c) @ rot2(-d).T + c; nxz = N[:, [0, 2]] @ rot2(-d).T
            P2 = P.copy(); P2[:, [0, 2]] = xz; N2 = N.copy(); N2[:, [0, 2]] = nxz
            cents.append(doorway_centre(P2, N2, P[:, 1] - floor_y, yaw, w, side))
        if None not in cents:
            doors.append({**{k: w[k] for k in ("a", "b", "axis")}, "centre_a": cents[0], "centre_b": cents[1]})
    fixed = max(geo, key=lambda r: geo[r]["area"])
    tr, tau = solve_translations(snapped, walls, doors, fixed)
    out = {r: snapped[r] + tr[r] @ R for r in snapped}            # translation from Manhattan frame back to world
    return raw, out, {"common_yaw_deg": round(np.degrees(yaw), 2), "yaw_turns_deg": {int(r): round(float(np.degrees(t)), 2) for r, t in turns.items()},
                      "n_shared_walls": len(walls), "n_doorway_constraints": len(doors), "tau_m": round(tau, 3),
                      "translations_mm": {int(r): (1000 * v).round(1).tolist() for r, v in tr.items()}}


def layout_metrics(P, yaw):
    walls = shared_walls(P, yaw)
    gaps = [w["gap"] for w in walls]
    polys = {r: Polygon(p) for r, p in P.items()}
    ks = sorted(polys)
    ov = sum(polys[a].intersection(polys[b]).area for i, a in enumerate(ks) for b in ks[i + 1:])
    return {"n_shared_walls": len(walls), "gap_mm_median": round(1000 * float(np.median(gaps)), 1) if gaps else None,
            "gap_mm_std": round(1000 * float(np.std(gaps)), 1) if gaps else None, "overlap_m2": round(float(ov), 4)}


def rigid_disagreement(PA, PB):
    common = sorted(set(PA) & set(PB))
    X = np.array([PA[r].mean(0) for r in common]); Y = np.array([PB[r].mean(0) for r in common])
    mx, my = X.mean(0), Y.mean(0)
    U, _, Vt = np.linalg.svd((Y - my).T @ (X - mx)); D = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    d = np.linalg.norm(Y - ((X - mx) @ R.T + my), axis=1)
    return {int(r): round(1000 * float(v), 1) for r, v in zip(common, d)}, round(1000 * float(np.median(d)), 1), round(1000 * float(d.max()), 1)


def main(name="single_scan_with_ceiling", variant="V3"):
    cap = StrayCapture.load(ROOT / name); T = cap.T_wc(); t = cap.timestamps
    pc = G.fuse(cap, T, range(0, len(cap), 5)); seg = G.segment_rooms(np.asarray(pc.points), np.asarray(pc.normals))
    lab_d = ndi.grey_dilation(seg.labels, size=(7, 7))
    cij = np.clip(seg.grid.cell(T[:, [0, 2], 3]), 0, np.array(seg.grid.shape) - 1)
    rof = np.where(seg.labels[cij[:, 0], cij[:, 1]] > 0, seg.labels[cij[:, 0], cij[:, 1]], lab_d[cij[:, 0], cij[:, 1]])
    halves = {"A": {}, "B": {}}; spans_all = {}
    for label in range(1, seg.labels.max() + 1):
        sp = sorted(RG.visits(rof, t, label), key=lambda ab: t[ab[0]])
        if not sp:
            continue
        spans_all[label] = sp
        if len(sp) >= 2:
            halves["A"][label], halves["B"][label] = sp[0::2], sp[1::2]
        # single-visit rooms have no independent second placement; they are left out of the halves
    geo_full, clouds_full = full_rooms(cap, T, seg, spans_all, variant)
    res, lay = {"variant": variant, "capture": name, "label": "GT-free drift ablation (visit-split placements, shared shapes)",
                "rooms_full": sorted(int(r) for r in geo_full)}, {}
    for h, spans in halves.items():
        geo, clouds, pinfo = place_half(cap, T, seg, geo_full, clouds_full, spans, variant)
        raw, anch, info = anchored(geo, clouds, seg.grid.floor_y)
        yaw = common_yaw([g["yaw"] for g in geo.values()])
        lay[h] = {"raw": raw, "anchored": anch}
        res[h] = {"placement_icp": pinfo, "stitch": info, "raw_metrics": layout_metrics({r: snap_yaw(raw[r], geo[r]["yaw"], yaw)[0] for r in raw}, yaw),
                  "anchored_metrics": layout_metrics(anch, yaw),
                  "raw_room_yaw_spread_deg": round(float(np.degrees(np.ptp([np.mod(g["yaw"] - yaw + np.pi / 4, np.pi / 2) - np.pi / 4 for g in geo.values()]))), 2)}
    for k in ("raw", "anchored"):
        per, med, mx = rigid_disagreement(lay["A"][k], lay["B"][k])
        res[f"halves_disagreement_{k}"] = {"per_room_mm": per, "median_mm": med, "max_mm": mx}
    (OUT / f"ablation_{name}_{variant}.json").write_text(json.dumps(res, indent=1))
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    for ax, k in zip(axs, ("raw", "anchored")):
        for h, col in (("A", "tab:blue"), ("B", "tab:red")):
            for r, P in lay[h][k].items():
                Q = np.vstack([P, P[:1]]); ax.plot(Q[:, 0], Q[:, 1], color=col, lw=1.2, alpha=0.8)
                if h == "A":
                    ax.text(*P.mean(0), str(r), fontsize=9)
        ax.set_aspect("equal"); ax.set_title(f"{k}: half A (blue) vs half B (red)\nmedian room disagreement {res[f'halves_disagreement_{k}']['median_mm']} mm")
    fig.tight_layout(); fig.savefig(OUT / f"footprint_{name}_{variant}.png", dpi=80)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))

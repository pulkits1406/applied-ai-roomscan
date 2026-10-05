"""Room-geometry variants for the LiDAR repeatability study (e18).

All variants share room identity from one segmentation of the full capture (identity only; no
geometry is shared between the two halves of a split). Variants:
  V0  baseline: points from one visit, unoriented normals (e14 behaviour, but existence + position
      from the same subset so halves are independent)
  V1  all visits of the subset, rigidly ICP-aligned per room to the subset's longest visit
  V2  V1 + camera-oriented normals (a face supports only the room it faces) + robust offsets from a
      clutter-free height band restricted to the edge's own extent
  V3  V2 + topology regularisation: merge same-facing lines < MERGE_M apart, absorb notches
      shallower than NOTCH_M
"""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy import ndimage as ndi
from shapely import contains_xy
from shapely.geometry import Polygon as SPoly

from roomscan import geometry as G
from roomscan.stray import StrayCapture

reg = o3d.pipelines.registration
MERGE_M = 0.12
ALIGN_MIN_FITNESS = 0.15
KEEP_UNALIGNED = True
NOTCH_M = 0.15
BAND = (0.9, None)  # robust wall offsets use points from 0.9 m above floor up to ceiling - 0.1


# ------------------------------------------------------------------ oriented fusion
def frame_points_normals(cap: StrayCapture, T, i, stride=2, max_depth=4.5, min_cos=0.15):
    """World points and camera-facing normals from one depth frame (normals from the depth grid)."""
    d = cap.depth_m(i)[::stride, ::stride]
    c = cap.confidence(i)[::stride, ::stride]
    K = cap.K_depth(i); K[:2] /= stride
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    X = np.stack([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d], -1)
    dx = np.zeros_like(X); dy = np.zeros_like(X)
    dx[:, 1:-1] = X[:, 2:] - X[:, :-2]; dy[1:-1] = X[2:] - X[:-2]
    n = np.cross(dx, dy)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(nn, 1e-12)
    flip = (n * X).sum(-1) > 0          # make normals face the camera (n . (0 - X) > 0)
    n[flip] *= -1
    ok = (c == 2) & (d < max_depth) & (nn[..., 0] > 0)
    ok[0] = ok[-1] = ok[:, 0] = ok[:, -1] = False
    # reject grazing/edge normals: depth discontinuities produce unreliable normals
    ok &= np.abs((n * X).sum(-1)) / np.maximum(np.linalg.norm(X, axis=-1), 1e-9) > min_cos
    R, t = T[i, :3, :3], T[i, :3, 3]
    return X[ok] @ R.T + t, n[ok] @ R.T


CLOSE = {"max_depth": 4.5, "min_cos": 0.15}   # V5 sets close-range / near-normal observation selection


def fuse_oriented(cap, T, rows, voxel=0.02):
    P, N = [], []
    for i in rows:
        p, n = frame_points_normals(cap, T, i, max_depth=CLOSE["max_depth"], min_cos=CLOSE["min_cos"])
        P.append(p); N.append(n)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(P)))
    pc.normals = o3d.utility.Vector3dVector(np.concatenate(N))
    pc = pc.voxel_down_sample(voxel)   # averages normals within a voxel
    N = np.asarray(pc.normals); N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    pc.normals = o3d.utility.Vector3dVector(N)
    return pc


def fuse_unoriented(cap, T, rows):
    return G.fuse(cap, T, rows, max_depth=4.5)


# ------------------------------------------------------------------ visits / alignment
def visits(room_of_frame, t, label, gap_s=1.0):
    idx = np.nonzero(room_of_frame == label)[0]
    if len(idx) == 0:
        return []
    spans, s = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if t[b] - t[a] > gap_s:
            spans.append((s, a)); s = b
    spans.append((s, idx[-1]))
    return [sp for sp in spans if t[sp[1]] - t[sp[0]] >= 1.0]


def crop(pc, mask, seg):
    P = np.asarray(pc.points)
    ij = np.clip(seg.grid.cell(P[:, [0, 2]]), 0, np.array(seg.grid.shape) - 1)
    return pc.select_by_index(np.nonzero(mask[ij[:, 0], ij[:, 1]])[0])


def _groups(spans):
    """Normalise spans to visit groups: each group is a list of (a, b) row ranges of one visit."""
    return [g if isinstance(g, list) else [g] for g in spans]


def _rows(group, step):
    return [i for a, b in group for i in range(a, b + 1, step)]


def _dur(t, group):
    return sum(t[b] - t[a] for a, b in group)


def aligned_room_cloud(cap, T, spans, mask, seg, oriented, step=3):
    """Fuse each visit separately, ICP-align every visit to the longest one (room crop only)."""
    t = cap.timestamps
    groups = sorted(_groups(spans), key=lambda g: _dur(t, g), reverse=True)
    fuse = fuse_oriented if oriented else fuse_unoriented
    clouds = [crop(fuse(cap, T, _rows(g, step)), mask, seg) for g in groups]
    ref = clouds[0]
    if not ref.has_normals() or not oriented:
        ref.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
    out, info = [ref], []
    for c in clouds[1:]:
        if len(c.points) < 500:
            info.append({"skipped": True}); continue
        r = reg.registration_icp(c, ref, 0.05, np.eye(4), reg.TransformationEstimationPointToPlane())
        r = reg.registration_icp(c, ref, 0.02, r.transformation, reg.TransformationEstimationPointToPlane())
        ok = r.fitness > ALIGN_MIN_FITNESS and r.inlier_rmse < 0.025 and np.linalg.norm(r.transformation[:3, 3]) < 0.3
        info.append({"fitness": round(r.fitness, 3), "shift_mm": round(1000 * float(np.linalg.norm(r.transformation[:3, 3])), 1), "aligned": bool(ok)})
        c = o3d.geometry.PointCloud(c)
        if ok:
            c.transform(r.transformation)
        if ok or KEEP_UNALIGNED:
            # an unaligned visit still carries coverage (e.g. the upper-wall sweep) that wall
            # existence depends on; dropping it removed every wall line in e18 run 2
            out.append(c)
    merged = out[0]
    for c in out[1:]:
        merged += c
    merged = merged.voxel_down_sample(0.02)
    if not oriented:
        merged.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
    return merged, info


# ------------------------------------------------------------------ walls
def lines_signed(UV, NUV, H, axis, oriented, min_extent=1.2, min_top=2.0):
    """Wall-face candidates perpendicular to `axis`. With oriented normals, faces are split by the
    side they face (+1: normal along +axis). Returns list of (coord, sign)."""
    if not oriented:
        return [(c, 0) for c in G.wall_lines(UV, NUV, H, axis, min_extent, min_top)[0]]
    out = []
    for sign in (+1, -1):
        m = sign * NUV[:, axis] > 0.85
        if m.sum() < 50:
            continue
        cs, _ = G.wall_lines(UV[m], NUV[m], H[m], axis, min_extent, min_top)
        out += [(c, sign) for c in cs]
    return out


def merge_lines(lines, tol=MERGE_M):
    """Merge same-facing lines closer than tol (duplicate faces: drift, skirting, double visits)."""
    out = []
    for sign in {s for _, s in lines}:
        cs = sorted(c for c, s in lines if s == sign)
        group = [cs[0]] if cs else []
        for c in cs[1:]:
            if c - group[-1] < tol:
                group.append(c)
            else:
                out.append((float(np.median(group)), sign)); group = [c]
        if group:
            out.append((float(np.median(group)), sign))
    return out


def edge_support_signed(S, Sn, axis, c, lo, hi, into_sign, tol=0.03, bin_m=0.05):
    """Coverage of segment {coord[axis]=c, other in [lo,hi]} by structural points whose normal
    points to `into_sign` along axis (0 = ignore orientation)."""
    o = 1 - axis
    m = (np.abs(S[:, axis] - c) < tol) & (S[:, o] >= lo) & (S[:, o] <= hi)
    if into_sign:
        m &= into_sign * Sn[:, axis] > 0.7
    nb = max(int(np.ceil((hi - lo) / bin_m)), 1)
    return len(np.unique(np.clip(((S[m, o] - lo) / bin_m).astype(int), 0, nb - 1))) / nb


def polygon_from_lines(reg_uv, us, vs, S, Sn, other_uv, oriented, inside_frac=0.35, min_support=0.3):
    """Arrangement-cell union grown across edges lacking (correctly facing) structural support."""
    us, vs = np.sort(np.unique(np.round(us, 4))), np.sort(np.unique(np.round(vs, 4)))
    if len(us) < 2 or len(vs) < 2:
        return None
    area = np.outer(np.diff(us), np.diff(vs))

    def cov(pts):
        c = np.zeros(area.shape)
        if pts is None or not len(pts):
            return c
        iu = np.searchsorted(us, pts[:, 0]) - 1; iv = np.searchsorted(vs, pts[:, 1]) - 1
        ok = (iu >= 0) & (iu < len(us) - 1) & (iv >= 0) & (iv < len(vs) - 1)
        np.add.at(c, (iu[ok], iv[ok]), 1)
        return c * G.GRID**2 / np.maximum(area, 1e-9)

    cell = cov(reg_uv) >= inside_frac
    forbid = cov(other_uv) >= 0.3
    for _ in range(30):
        add = np.zeros_like(cell)
        for i, j in zip(*np.nonzero(cell)):
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if not (0 <= a < cell.shape[0] and 0 <= b < cell.shape[1]) or cell[a, b] or forbid[a, b]:
                    continue
                # wall between inside (i,j) and outside (a,b) must face the inside: normal points from
                # outside toward inside, i.e. along -di / -dj
                if di:
                    sup = edge_support_signed(S, Sn, 0, us[max(i, a)], vs[j], vs[j + 1], -di if oriented else 0)
                else:
                    sup = edge_support_signed(S, Sn, 1, vs[max(j, b)], us[i], us[i + 1], -dj if oriented else 0)
                if sup < min_support:
                    add[a, b] = True
        if not add.any():
            break
        cell |= add
    lab, n = ndi.label(cell)
    if n == 0:
        return None
    sizes = ndi.sum(area, lab, index=np.arange(1, n + 1))
    return G._trace_rectilinear(lab == (np.argmax(sizes) + 1), us, vs)


def absorb_notches(poly, min_depth=NOTCH_M):
    """Remove rectilinear notches/bumps shallower than min_depth: an edge shorter than min_depth whose
    neighbours turn in opposite directions (a step) is collapsed by snapping to the longer side."""
    P = poly.copy()
    changed = True
    while changed and len(P) > 4:
        changed = False
        n = len(P)
        L = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
        for k in np.argsort(L):
            if L[k] >= min_depth:
                break
            a, b = P[k], P[(k + 1) % n]
            prev, nxt = P[k - 1], P[(k + 2) % n]
            # step: the edges before and after are parallel to each other (perpendicular to this edge)
            e0, e1 = a - prev, nxt - b
            if abs(np.cross(e0, e1)) > 1e-9:
                continue
            # move the shorter neighbouring edge onto the longer one's line
            axis = 0 if abs(b[0] - a[0]) > abs(b[1] - a[1]) else 1
            if np.linalg.norm(e0) < np.linalg.norm(e1):
                P[k - 1][axis] = b[axis]; P[k][axis] = b[axis]
            else:
                P[(k + 1) % n][axis] = a[axis]; P[(k + 2) % n][axis] = a[axis]
            keep = [m for m in range(n) if not (np.allclose(P[m], P[(m + 1) % n]))]
            P = P[keep]
            keep = [m for m in range(len(P)) if not G._collinear(P[m - 1], P[m], P[(m + 1) % len(P)])]
            P = P[keep]
            changed = True
            break
    return P


def outermost_peak(x, inward_sign, bin_m=0.005, rel=0.3):
    """Wall face = the most OUTWARD density peak with >= rel of the strongest peak's support
    (furniture/skirting faces lie on the interior side of the wall face)."""
    if len(x) < 30:
        return None
    bins = np.arange(x.min() - 0.01, x.max() + 0.011, bin_m)
    h = ndi.gaussian_filter1d(np.histogram(x, bins=bins)[0].astype(float), 1.5)
    from scipy.signal import find_peaks
    pk, _ = find_peaks(h, height=rel * h.max())
    if not len(pk):
        return None
    c = bins[pk] + bin_m / 2
    c = c[np.argmin(c * inward_sign)]            # outward = against the inward direction
    near = np.abs(x - c) < 0.01
    return float(np.median(x[near]))


def refine_offsets(poly, UV, NUV, H, ceil_h, oriented, win=0.05, mode="median"):
    """Robust per-edge offset: median coordinate of points within win of the edge line, facing the
    interior (oriented), in the clutter-free band, within the edge's extent. Returns polygon with
    refined offsets (vertices recomputed as intersections) and per-edge support counts."""
    n = len(poly); P = poly.copy()
    ccw = G.polygon_area(P) > 0
    top = (ceil_h - 0.1) if ceil_h else 2.6
    band = (H > BAND[0]) & (H < top)
    offs, sup = [], []
    for k in range(n):
        a, b = P[k], P[(k + 1) % n]
        axis = 0 if abs(b[0] - a[0]) < 1e-9 else 1      # edge perpendicular to this axis
        c = a[axis]; o = 1 - axis
        lo, hi = sorted([a[o], b[o]])
        d = b - a
        inward = np.array([-d[1], d[0]]) if ccw else np.array([d[1], -d[0]])   # left normal for CCW
        into = int(np.sign(inward[axis]))
        m = band & (np.abs(UV[:, axis] - c) < win) & (UV[:, o] > lo + 0.05) & (UV[:, o] < hi - 0.05)
        m &= (into * NUV[:, axis] > 0.85) if oriented else (np.abs(NUV[:, axis]) > 0.85)
        if m.sum() >= 30:
            v = float(np.median(UV[m, axis])) if mode == "median" else outermost_peak(UV[m, axis], into)
            offs.append(v if v is not None else float(c)); sup.append(int(m.sum()))
        else:
            offs.append(float(c)); sup.append(int(m.sum()))
    Q = P.copy()
    for k in range(n):
        a, b = P[k], P[(k + 1) % n]
        axis = 0 if abs(b[0] - a[0]) < 1e-9 else 1
        Q[k][axis] = offs[k]; Q[(k + 1) % n][axis] = offs[k]
    return Q, sup


def floor_ceiling_robust(P, N, poly_uv, Rm, floor_guess):
    """Floor / dominant ceiling from horizontal-normal points inside the polygon (shrunk 0.2 m)."""
    shp = SPoly(poly_uv).buffer(-0.2)
    if shp.is_empty:
        shp = SPoly(poly_uv)
    uv = P[:, [0, 2]] @ Rm.T
    m = contains_xy(shp, uv[:, 0], uv[:, 1]) & (np.abs(N[:, 1]) > 0.9)
    y = P[m, 1]
    fl = y[np.abs(y - floor_guess) < 0.08]
    floor = float(np.median(fl)) if len(fl) > 200 else None
    hi = y[y > (floor or floor_guess) + 2.0]
    if len(hi) < 300 or floor is None:
        return floor, None, 0.0
    h, e = np.histogram(hi, bins=np.arange(hi.min(), hi.max() + 0.01, 0.005))
    c = e[np.argmax(ndi.gaussian_filter1d(h.astype(float), 1))] + 0.0025
    near = hi[np.abs(hi - c) < 0.015]
    return floor, float(np.median(near)), float(len(near) / max(m.sum(), 1))


# ------------------------------------------------------------------ one room, one variant
def room_geometry(cap, T, seg, label, spans, variant):
    oriented = variant in ("V2", "V3", "V4", "V5")
    CLOSE.update({"max_depth": 2.5, "min_cos": 0.5} if variant == "V5" else {"max_depth": 4.5, "min_cos": 0.15})
    region = seg.labels == label
    near = ndi.binary_dilation(region, iterations=int(0.5 / G.GRID))
    t = cap.timestamps
    if variant == "V0":
        # e14 v0 logic within this half: existence/lines from all of the half's points (no
        # alignment), positions refined from its longest visit
        groups = _groups(spans)
        pc = crop(fuse_unoriented(cap, T, [i for g in groups for i in _rows(g, 3)]), near, seg); info = []
        longest = max(groups, key=lambda g: _dur(t, g))
        lpc = crop(fuse_unoriented(cap, T, _rows(longest, 3)), near, seg)
    else:
        pc, info = aligned_room_cloud(cap, T, spans, near, seg, oriented)
    P, N = np.asarray(pc.points), np.asarray(pc.normals)
    if len(P) < 2000:
        return None, {"reason": "too_few_points"}
    yaw = G.manhattan_yaw(N); Rm = G._rot(yaw)
    UV, NUV, H = P[:, [0, 2]] @ Rm.T, N[:, [0, 2]] @ Rm.T, P[:, 1] - seg.grid.floor_y
    lu = lines_signed(UV, NUV, H, 0, oriented); lv = lines_signed(UV, NUV, H, 1, oriented)
    if variant == "V3":
        lu, lv = merge_lines(lu), merge_lines(lv)
    us = np.array([c for c, _ in lu]); vs = np.array([c for c, _ in lv])
    vert = np.abs(N[:, 1]) < 0.3
    struct = vert & (H > G.H_STRUCT)
    reg_uv = seg.grid.centers()[G.fill_region(seg.labels, label)] @ Rm.T
    other = seg.grid.centers()[(seg.labels > 0) & (seg.labels != label)] @ Rm.T
    if variant == "V0":
        LP, LN = np.asarray(lpc.points), np.asarray(lpc.normals)
        LUV, LNUV = LP[:, [0, 2]] @ Rm.T, LN[:, [0, 2]] @ Rm.T
        us, vs = G.refine_lines(us, LUV, LNUV, 0), G.refine_lines(vs, LUV, LNUV, 1)
    poly = polygon_from_lines(reg_uv, us, vs, UV[struct], NUV[struct], other, oriented)
    if poly is None or len(poly) < 4:
        return None, {"reason": "no_polygon", "n_lines": [len(us), len(vs)]}
    if variant == "V3":
        poly = absorb_notches(poly)
    floor, ceil, csup = floor_ceiling_robust(P, N, poly, Rm, seg.grid.floor_y)
    ceil_h = (ceil - floor) if (ceil is not None and floor is not None) else None
    if variant in ("V2", "V3", "V4", "V5"):
        poly, sup = refine_offsets(poly, UV, NUV, H, ceil_h, oriented, win=0.08 if variant == "V4" else 0.05,
                                   mode="outermost" if variant == "V4" else "median")
    lens = np.linalg.norm(np.roll(poly, -1, 0) - poly, axis=1)
    return {"poly_uv": poly, "yaw": yaw, "lengths": lens, "n_walls": len(poly), "extent_u": float(np.ptp(poly[:, 0])),
            "extent_v": float(np.ptp(poly[:, 1])), "ceiling_h": ceil_h, "area": abs(G.polygon_area(poly))}, {"align": info, "n_lines": [len(us), len(vs)]}


# ------------------------------------------------------------------ fixed-topology position test
def room_cloud(cap, T, seg, label, spans, variant):
    """The point set a variant measures from (V0: unaligned union; V1+: per-visit ICP-aligned)."""
    oriented = variant in ("V2", "V3", "V4", "V5")
    CLOSE.update({"max_depth": 2.5, "min_cos": 0.5} if variant == "V5" else {"max_depth": 4.5, "min_cos": 0.15})
    near = ndi.binary_dilation(seg.labels == label, iterations=int(0.5 / G.GRID))
    if variant == "V0":
        groups = _groups(spans)
        return crop(fuse_unoriented(cap, T, [i for g in groups for i in _rows(g, 3)]), near, seg)
    pc, _ = aligned_room_cloud(cap, T, spans, near, seg, oriented)
    return pc


def edge_positions(poly, UV, NUV, H, ceil_h, variant, win=0.05):
    """Per-edge wall position under a variant's measurement rule, topology given.
    V0/V1: median of axis-aligned-normal points within win of the edge line, any height, within the
           edge extent; V2/V3: refine_offsets (interior-facing, clutter-free band, own extent)."""
    if variant in ("V2", "V3", "V4", "V5"):
        Q, sup = refine_offsets(poly, UV, NUV, H, ceil_h, True, 0.08 if variant == "V4" else win,
                                mode="outermost" if variant == "V4" else "median")
        return Q, sup
    n = len(poly); offs, sup = [], []
    for k in range(n):
        a, b = poly[k], poly[(k + 1) % n]
        axis = 0 if abs(b[0] - a[0]) < 1e-9 else 1; o = 1 - axis
        lo, hi = sorted([a[o], b[o]])
        m = (np.abs(UV[:, axis] - a[axis]) < win) & (UV[:, o] > lo + 0.05) & (UV[:, o] < hi - 0.05) & (np.abs(NUV[:, axis]) > 0.85) & (H > 0.1)
        offs.append(float(np.median(UV[m, axis])) if m.sum() >= 30 else float(a[axis])); sup.append(int(m.sum()))
    Q = poly.copy()
    for k in range(n):
        a, b = poly[k], poly[(k + 1) % n]
        axis = 0 if abs(b[0] - a[0]) < 1e-9 else 1
        Q[k][axis] = offs[k]; Q[(k + 1) % n][axis] = offs[k]
    return Q, sup

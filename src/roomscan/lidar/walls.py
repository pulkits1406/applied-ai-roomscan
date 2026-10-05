"""Plane/wall extraction for one room (e18 variant V6).

Topology (which walls exist) comes from all observations; a wall face must reach structural height
and only supports the room it faces. Wall positions come from close-range near-normal observations
where a wall has enough of them, else from all observations (e18: close-range 7.5 mm vs 11 mm
median half-to-half difference, but too sparse to carry topology).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from roomscan import geometry as G

BAND_LOW = 0.9          # m above floor; below this, furniture dominates wall faces
CLOSE_MIN_SUPPORT = 60  # points for a close-range wall position to be used
WIN = 0.05              # m, half-width of the band around a wall line used for its position


@dataclass
class Walls:
    yaw: float                 # room Manhattan yaw (world x,z -> room u,v via G._rot)
    poly_uv: np.ndarray        # rectilinear polygon in room frame, CCW
    support: list[int]         # points used per edge
    source: list[str]          # "close" | "all" | "unmeasured" per edge


def signed_lines(UV, NUV, H, axis):
    """Wall-face candidates perpendicular to axis, split by the side they face."""
    out = []
    for sign in (+1, -1):
        m = sign * NUV[:, axis] > 0.85
        if m.sum() >= 50:
            out += [(c, sign) for c in G.wall_lines(UV[m], NUV[m], H[m], axis)[0]]
    return out


def _support(S, Sn, axis, c, lo, hi, into, tol=0.03, bin_m=0.05):
    o = 1 - axis
    m = (np.abs(S[:, axis] - c) < tol) & (S[:, o] >= lo) & (S[:, o] <= hi) & (into * Sn[:, axis] > 0.7)
    nb = max(int(np.ceil((hi - lo) / bin_m)), 1)
    return len(np.unique(np.clip(((S[m, o] - lo) / bin_m).astype(int), 0, nb - 1))) / nb


def polygon(region_uv, us, vs, S, Sn, other_uv, inside_frac=0.35, min_support=0.3):
    """Union of line-arrangement cells covered by the room's free space, grown across any edge not
    covered by structural points facing into the room; cells of other rooms are never added."""
    us, vs = np.unique(np.round(us, 4)), np.unique(np.round(vs, 4))
    if len(us) < 2 or len(vs) < 2:
        return None
    area = np.outer(np.diff(us), np.diff(vs))

    def cov(pts):
        c = np.zeros(area.shape)
        if pts is None or not len(pts):
            return c
        iu, iv = np.searchsorted(us, pts[:, 0]) - 1, np.searchsorted(vs, pts[:, 1]) - 1
        ok = (iu >= 0) & (iu < len(us) - 1) & (iv >= 0) & (iv < len(vs) - 1)
        np.add.at(c, (iu[ok], iv[ok]), 1)
        return c * G.GRID**2 / np.maximum(area, 1e-9)

    cell, forbid = cov(region_uv) >= inside_frac, cov(other_uv) >= 0.3
    for _ in range(30):
        add = np.zeros_like(cell)
        for i, j in zip(*np.nonzero(cell)):
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if not (0 <= a < cell.shape[0] and 0 <= b < cell.shape[1]) or cell[a, b] or forbid[a, b]:
                    continue
                if di:
                    sup = _support(S, Sn, 0, us[max(i, a)], vs[j], vs[j + 1], -di)
                else:
                    sup = _support(S, Sn, 1, vs[max(j, b)], us[i], us[i + 1], -dj)
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


def edge_axis(a, b):
    return 0 if abs(b[0] - a[0]) < 1e-9 else 1      # axis the edge is perpendicular to


def inward_sign(poly, k):
    a, b = poly[k], poly[(k + 1) % len(poly)]
    d = b - a
    inward = np.array([-d[1], d[0]]) if G.polygon_area(poly) > 0 else np.array([d[1], -d[0]])
    return int(np.sign(inward[edge_axis(a, b)]))


def edge_position(poly, k, UV, NUV, H, top):
    """Median coordinate of interior-facing wall points near edge k, in the clutter-free band and
    the edge's own extent. Returns (coord or None, support)."""
    a, b = poly[k], poly[(k + 1) % len(poly)]
    ax = edge_axis(a, b); o = 1 - ax
    lo, hi = sorted([a[o], b[o]])
    into = inward_sign(poly, k)
    m = (H > BAND_LOW) & (H < top) & (np.abs(UV[:, ax] - a[ax]) < WIN) & (UV[:, o] > lo + 0.05) & (UV[:, o] < hi - 0.05) & (into * NUV[:, ax] > 0.85)
    return (float(np.median(UV[m, ax])) if m.sum() >= 30 else None), int(m.sum())


def extract(all_P, all_N, close_P, close_N, floor_y, ceil_h, region_uv, other_uv) -> Walls | None:
    yaw = G.manhattan_yaw(all_N)
    Rm = G._rot(yaw)
    UV, NUV, H = all_P[:, [0, 2]] @ Rm.T, all_N[:, [0, 2]] @ Rm.T, all_P[:, 1] - floor_y
    lu, lv = signed_lines(UV, NUV, H, 0), signed_lines(UV, NUV, H, 1)
    struct = (np.abs(all_N[:, 1]) < 0.3) & (H > G.H_STRUCT)
    poly = polygon(region_uv @ Rm.T, np.array([c for c, _ in lu]), np.array([c for c, _ in lv]),
                   UV[struct], NUV[struct], other_uv @ Rm.T)
    if poly is None or len(poly) < 4:
        return None
    top = (ceil_h - 0.1) if ceil_h else 2.6
    CUV, CNUV, CH = close_P[:, [0, 2]] @ Rm.T, close_N[:, [0, 2]] @ Rm.T, close_P[:, 1] - floor_y
    Q, sup, src = poly.copy(), [], []
    n = len(poly)
    for k in range(n):
        c, s = edge_position(poly, k, CUV, CNUV, CH, top)
        if c is not None and s >= CLOSE_MIN_SUPPORT:
            source = "close"
        else:
            c, s = edge_position(poly, k, UV, NUV, H, top)
            source = "all" if c is not None else "unmeasured"
            c = poly[k][edge_axis(poly[k], poly[(k + 1) % n])] if c is None else c
        ax = edge_axis(poly[k], poly[(k + 1) % n])
        Q[k][ax] = c
        Q[(k + 1) % n][ax] = c
        sup.append(s); src.append(source)
    return Walls(yaw, Q, sup, src)

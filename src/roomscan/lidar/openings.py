"""Doorways between rooms: detection from the 2-D segmentation, width from each room's own wall face.

Detection: structural barrier (lintel) with no wall at mid-height and floor beneath, touching two
rooms (geometry.segment_rooms). Width: on the wall edge of a room nearest the doorway, the gap in
interior-facing mid-height wall points (1 cm bins) that contains the doorway centre, i.e. the
jamb-to-jamb clear opening seen from that room. Windows are not detected (glass defeats LiDAR).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from roomscan import geometry as G
from roomscan.lidar.walls import edge_axis, inward_sign

BIN = 0.01
DOOR_W = (0.5, 1.6)


@dataclass
class Doorway:
    rooms: tuple[int, int]
    centre_xz: np.ndarray
    coarse_width: float


@dataclass
class OpeningMeasure:
    edge: int
    width: float | None
    centre_along: float | None


def detect(seg) -> list[Doorway]:
    centers = seg.grid.centers()
    lab, n = ndi.label(seg.doorway)
    out = []
    for k in range(1, n + 1):
        cells = lab == k
        ring = ndi.binary_dilation(cells, iterations=3) & ~cells
        touching = sorted({int(x) for x in np.unique(seg.labels[ring]) if x > 0})
        if len(touching) != 2 or cells.sum() < 3:
            continue
        pts = centers[cells]
        ev, evec = np.linalg.eigh(np.cov(pts.T))
        w = float(np.ptp(pts @ evec[:, -1]) + G.GRID)
        if 0.5 <= w <= 2.5:
            out.append(Doorway((touching[0], touching[1]), pts.mean(0), w))
    return out


def measure(poly_uv, Rm, centre_xz, P, N, floor_y, max_dist=0.4) -> OpeningMeasure | None:
    c = centre_xz @ Rm.T
    UV, NUV, H = P[:, [0, 2]] @ Rm.T, N[:, [0, 2]] @ Rm.T, P[:, 1] - floor_y
    best = None
    for k in range(len(poly_uv)):
        a, b = poly_uv[k], poly_uv[(k + 1) % len(poly_uv)]
        ax = edge_axis(a, b); o = 1 - ax
        lo, hi = sorted([a[o], b[o]])
        d = abs(c[ax] - a[ax])
        if lo - 0.2 <= c[o] <= hi + 0.2 and d < max_dist and (best is None or d < best[0]):
            best = (d, k, ax, o, lo, hi, a[ax])
    if best is None:
        return None
    _, k, ax, o, lo, hi, coord = best
    into = inward_sign(poly_uv, k)
    m = (np.abs(UV[:, ax] - coord) < 0.06) & (into * NUV[:, ax] > 0.7) & (H > 0.3) & (H < 1.8)
    bins = np.arange(lo, hi + BIN, BIN)
    if len(bins) < 20:
        return OpeningMeasure(k, None, None)
    occ = np.histogram(UV[m, o], bins=bins)[0] > 0
    occ = ndi.binary_closing(occ, iterations=2)        # bridge 1-2 cm sampling holes, not doorways
    gaps, s = [], None
    for i, f in enumerate(occ):
        if not f and s is None:
            s = i
        if f and s is not None:
            gaps.append((s, i)); s = None
    ci = (c[o] - lo) / BIN
    for g0, g1 in gaps:                                # gap must be bounded by jambs on both sides
        w = (g1 - g0) * BIN
        if g0 <= ci + 10 and g1 >= ci - 10 and DOOR_W[0] <= w <= DOOR_W[1]:
            return OpeningMeasure(k, float(w), float(bins[g0] + w / 2))
    return OpeningMeasure(k, None, None)

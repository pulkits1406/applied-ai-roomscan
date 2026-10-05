"""LiDAR-tier geometry prototype: fused points -> rooms -> rectilinear walls -> ceiling -> doorways.

Frames: inputs are in the ARKit world (metres, +y up). Plans are emitted in the property frame
(x, y horizontal, z up) via ARKIT_TO_PROPERTY.

Evidence behind the design (ENGINEERING_LOG, e09/e10):
- Room separation uses vertical surfaces above furniture height: door lintels close doorways.
- Drift is close to rigid per room and accumulates between rooms, so each room's geometry comes
  from its longest continuous visit, with other visits rigidly aligned to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy import ndimage as ndi
from scipy.signal import find_peaks

from roomscan.stray import StrayCapture

ARKIT_TO_PROPERTY = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
GRID = 0.05            # m, 2-D segmentation grid
H_STRUCT = 1.9         # m above floor; vertical surfaces above this are structural (walls/lintels)
DOOR_LOW = (0.3, 1.8)  # m; band where a doorway has no wall
reg = o3d.pipelines.registration


# ---------------------------------------------------------------- fusion
def fuse(cap: StrayCapture, T: np.ndarray, rows, stride=2, max_depth=4.0, voxel=0.02):
    pts = []
    for i in rows:
        d = cap.depth_m(i)[::stride, ::stride]
        c = cap.confidence(i)[::stride, ::stride]
        K = cap.K_depth(i)
        K[:2] /= stride
        v, u = np.nonzero((c == 2) & (d < max_depth))
        z = d[v, u]
        pc = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
        pts.append(pc @ T[i, :3, :3].T + T[i, :3, 3])
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts))).voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
    return pc


# ---------------------------------------------------------------- segmentation
@dataclass
class Grid2D:
    origin: np.ndarray  # world (x, z) of cell (0, 0)
    shape: tuple[int, int]
    floor_y: float

    def cell(self, xz: np.ndarray) -> np.ndarray:
        return np.floor((xz - self.origin) / GRID).astype(int)

    def centers(self) -> np.ndarray:
        i, j = np.mgrid[0:self.shape[0], 0:self.shape[1]]
        return np.stack([i, j], -1) * GRID + self.origin + GRID / 2


@dataclass
class Segmentation:
    grid: Grid2D
    labels: np.ndarray
    barrier: np.ndarray
    floor: np.ndarray
    doorway: np.ndarray
    high_observed: bool


def segment_rooms(P: np.ndarray, N: np.ndarray, min_room_m2=1.0, pad_m: float = 0.3, pad_high_m: float = 0.0) -> Segmentation:
    """pad_m / pad_high_m: empty margin below / above the data extent. The morphological closing
    (0.6 m) treats outside the grid as empty, so with less padding than that the region is eroded
    near the grid border (e.g. 0.27 m / 0.53 m inset with the v0 defaults, which are kept for v0)."""
    f0 = float(np.percentile(P[:, 1], 2))
    h = P[:, 1] - f0
    origin = P[:, [0, 2]].min(0) - pad_m
    ij = np.floor((P[:, [0, 2]] - origin) / GRID).astype(int)
    shape = tuple(ij.max(0) + 1 + int(round(pad_high_m / GRID)))

    def occ(m, c=2):
        g = np.zeros(shape, np.int32)
        np.add.at(g, (ij[m, 0], ij[m, 1]), 1)
        return g >= c

    vert = np.abs(N[:, 1]) < 0.3
    barrier = ndi.binary_closing(occ(vert & (h > H_STRUCT) & (h < 3.5), 3), iterations=2)
    low = ndi.binary_dilation(occ(vert & (h > DOOR_LOW[0]) & (h < DOOR_LOW[1]), 2), iterations=1)
    floor = occ(np.abs(h) < 0.05, 1)
    domain = ndi.binary_fill_holes(ndi.binary_closing(floor, iterations=int(0.6 / GRID)))
    domain &= ndi.binary_dilation(floor, iterations=int(0.3 / GRID))
    comp, n = ndi.label(~barrier & domain)
    idx = np.arange(1, n + 1)
    has_floor = ndi.sum(floor, comp, index=idx) * GRID**2
    area = ndi.sum(np.ones_like(comp), comp, index=idx) * GRID**2
    labels = np.zeros_like(comp)
    k = 0
    for l in idx:
        if has_floor[l - 1] >= 0.3 and area[l - 1] >= min_room_m2:
            k += 1
            labels[comp == l] = k
    # doorway: structural (high) barrier with no wall at mid-height and floor beneath
    doorway = barrier & ~low & ndi.binary_dilation(floor, iterations=2)
    high_observed = bool((vert & (h > H_STRUCT)).sum() > 0.02 * vert.sum())
    return Segmentation(Grid2D(origin, shape, f0), labels, barrier, floor, doorway, high_observed)


# ---------------------------------------------------------------- per-room layout
@dataclass
class RoomGeom:
    label: int
    yaw: float                      # rotation of the room's Manhattan frame about world +y
    polygon_xz: np.ndarray          # (K, 2) world x, z, counter-clockwise in plan view
    floor_y: float
    ceiling_y: float | None
    ceiling_support: float          # fraction of interior cells with ceiling observations
    wall_support: list[float] = field(default_factory=list)
    n_points: int = 0


def manhattan_yaw(N: np.ndarray) -> float:
    v = N[np.abs(N[:, 1]) < 0.2][:, [0, 2]]
    ang = np.mod(np.arctan2(v[:, 1], v[:, 0]), np.pi / 2)
    hist, edges = np.histogram(ang, bins=180, range=(0, np.pi / 2))
    hist = ndi.gaussian_filter1d(hist.astype(float), 2, mode="wrap")
    return float(edges[np.argmax(hist)] + (edges[1] - edges[0]) / 2)


def _rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s], [-s, c]])  # world (x, z) -> room (u, v)


def wall_lines(UV: np.ndarray, Nuv: np.ndarray, H: np.ndarray, axis: int, min_extent=1.2, min_top=2.0):
    """Coordinates of wall faces perpendicular to `axis`: peaks of point density along the axis,
    kept only if the supporting points span >= min_extent vertically AND reach min_top above the
    floor (furniture faces are short or stop below structural height)."""
    m = np.abs(Nuv[:, axis]) > 0.85
    x = UV[m, axis]
    if len(x) < 50:
        return np.array([]), np.array([])
    bins = np.arange(x.min() - 0.05, x.max() + 0.06, 0.01)
    hist, edges = np.histogram(x, bins=bins)
    sm = ndi.gaussian_filter1d(hist.astype(float), 1.0)
    peaks, props = find_peaks(sm, height=max(8.0, 0.02 * sm.max()), distance=6)
    lines, sup = [], []
    for p in peaks:
        c = edges[p] + 0.005
        sel = m.copy()
        sel[m] = np.abs(x - c) < 0.02
        if H[sel].size and (np.percentile(H[sel], 95) - np.percentile(H[sel], 5)) >= min_extent and np.percentile(H[sel], 98) >= min_top:
            # refine to the mean of nearby points
            lines.append(float(np.mean(UV[sel, axis])))
            sup.append(float(sel.sum()))
    return np.array(lines), np.array(sup)


def refine_lines(lines: np.ndarray, UV: np.ndarray, Nuv: np.ndarray, axis: int, win=0.05, min_pts=30) -> np.ndarray:
    """Re-estimate each wall coordinate from a subset of points (one visit); keep the input
    coordinate where that subset has too little support."""
    out = []
    m = np.abs(Nuv[:, axis]) > 0.85
    x = UV[m, axis]
    for c in lines:
        sel = np.abs(x - c) < win
        out.append(float(np.median(x[sel])) if sel.sum() >= min_pts else float(c))
    return np.array(out)


def fill_region(labels: np.ndarray, label: int, close_m=0.5) -> np.ndarray:
    """Room region with furniture holes and notches closed, never growing into another room."""
    r = labels == label
    k = int(close_m / GRID)
    filled = ndi.binary_fill_holes(ndi.binary_closing(np.pad(r, k), iterations=k))[k:-k, k:-k]
    return filled & ((labels == 0) | r)


def _edge_support(S: np.ndarray, axis: int, c: float, lo: float, hi: float, tol=0.03, bin_m=0.05) -> float:
    """Fraction of the segment {coord[axis] = c, other coord in [lo, hi]} covered by structural
    points S (within tol of the line), measured in bin_m bins."""
    o = 1 - axis
    m = (np.abs(S[:, axis] - c) < tol) & (S[:, o] >= lo) & (S[:, o] <= hi)
    nb = max(int(np.ceil((hi - lo) / bin_m)), 1)
    hit = np.unique(np.clip(((S[m, o] - lo) / bin_m).astype(int), 0, nb - 1))
    return len(hit) / nb


def room_polygon(region_uv_pts: np.ndarray, us: np.ndarray, vs: np.ndarray, inside_frac=0.35,
                 structural_uv: np.ndarray | None = None, other_uv: np.ndarray | None = None, min_support=0.3):
    """Union of arrangement cells (between consecutive wall lines) mostly covered by the room's
    free-space samples, then grown across any boundary edge that structural points do not cover
    (a notch is only real if a wall exists along that edge segment). Cells mostly covered by
    another room are never added. Returns the outer boundary as a rectilinear polygon in (u, v)."""
    if len(us) < 2 or len(vs) < 2:
        return None
    us, vs = np.sort(us), np.sort(vs)
    area = np.outer(np.diff(us), np.diff(vs))

    def coverage(pts):
        c = np.zeros(area.shape)
        if pts is None or len(pts) == 0:
            return c
        iu = np.searchsorted(us, pts[:, 0]) - 1
        iv = np.searchsorted(vs, pts[:, 1]) - 1
        ok = (iu >= 0) & (iu < len(us) - 1) & (iv >= 0) & (iv < len(vs) - 1)
        np.add.at(c, (iu[ok], iv[ok]), 1)
        return c * GRID**2 / np.maximum(area, 1e-9)

    cell_in = coverage(region_uv_pts) >= inside_frac
    forbid = coverage(other_uv) >= 0.3
    if structural_uv is not None and len(structural_uv):
        for _ in range(20):
            add = np.zeros_like(cell_in)
            I, J = np.nonzero(cell_in)
            for i, j in zip(I, J):
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = i + di, j + dj
                    if not (0 <= a < cell_in.shape[0] and 0 <= b < cell_in.shape[1]) or cell_in[a, b] or forbid[a, b]:
                        continue
                    if di:  # shared edge lies on u = us[max(i, a)], spanning v in [vs[j], vs[j+1]]
                        sup = _edge_support(structural_uv, 0, us[max(i, a)], vs[j], vs[j + 1])
                    else:
                        sup = _edge_support(structural_uv, 1, vs[max(j, b)], us[i], us[i + 1])
                    if sup < min_support:
                        add[a, b] = True
            if not add.any():
                break
            cell_in |= add
    lab, n = ndi.label(cell_in)
    if n == 0:
        return None
    sizes = ndi.sum(area, lab, index=np.arange(1, n + 1))
    cell_in = lab == (np.argmax(sizes) + 1)
    return _trace_rectilinear(cell_in, us, vs)


def _trace_rectilinear(mask: np.ndarray, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """Outer boundary of a 4-connected cell mask on a non-uniform grid, counter-clockwise."""
    m = np.pad(mask, 1)
    edges = {}
    for i in range(1, m.shape[0] - 1):
        for j in range(1, m.shape[1] - 1):
            if not m[i, j]:
                continue
            # directed edges keep the interior on the left (counter-clockwise outer boundary)
            if not m[i, j - 1]: edges[(i - 1, j - 1)] = (i, j - 1)
            if not m[i + 1, j]: edges[(i, j - 1)] = (i, j)
            if not m[i, j + 1]: edges[(i, j)] = (i - 1, j)
            if not m[i - 1, j]: edges[(i - 1, j)] = (i - 1, j - 1)
    start = min(edges)
    loop, cur = [start], edges[start]
    while cur != start and len(loop) <= len(edges):
        loop.append(cur)
        cur = edges[cur]
    pts = np.array([[us[a], vs[b]] for a, b in loop])
    keep = [k for k in range(len(pts)) if not _collinear(pts[k - 1], pts[k], pts[(k + 1) % len(pts)])]
    return pts[keep]


def _collinear(a, b, c):
    return abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) < 1e-9


def floor_ceiling(Pin: np.ndarray, floor_guess: float):
    """Floor and dominant ceiling levels (world y) from interior points."""
    h = Pin[:, 1]
    fl = h[np.abs(h - floor_guess) < 0.06]
    floor_y = float(np.median(fl)) if len(fl) > 200 else floor_guess
    hi = h[h > floor_y + 2.0]
    if len(hi) < 300:
        return floor_y, None, 0.0
    hist, edges = np.histogram(hi, bins=np.arange(hi.min(), hi.max() + 0.01, 0.005))
    c = edges[np.argmax(ndi.gaussian_filter1d(hist.astype(float), 1))] + 0.0025
    near = hi[np.abs(hi - c) < 0.015]
    return floor_y, float(np.median(near)), float(len(near) / max(len(Pin), 1))


def polygon_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

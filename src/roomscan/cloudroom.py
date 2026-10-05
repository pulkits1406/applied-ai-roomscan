"""Room geometry from a gravity-aligned metric point cloud (photo and video tiers).

Input convention is the LiDAR one: world +y up, horizontal plane (x, z), metres, normals oriented
toward the camera that observed each point. The same segmentation (geometry.segment_rooms),
wall-face topology/positions (lidar.walls) and level estimation (lidar.levels) as the LiDAR tier
are applied, so the tiers differ only in where the points come from. Points from monocular depth
have no "close-range" subset with better accuracy, so all points serve both roles.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from roomscan import geometry as G, quality
from roomscan.lidar import levels as LV, walls as WL

INFER = WL.Rules(infer_missing=True)


@dataclass
class CloudRoom:
    walls: WL.Walls
    floor_y: float
    ceil_h: float | None
    n_ceil: int
    step: str                   # "observed" | "inferred" (sides closed at the free-space boundary)
    region_label: int


def point_normals(X: np.ndarray, valid: np.ndarray, min_cos: float = 0.15):
    """Camera-frame point map (H, W, 3) -> (points, normals) oriented toward the camera; grazing
    and depth-edge samples rejected as in lidar.fusion."""
    dx = np.zeros_like(X); dy = np.zeros_like(X)
    dx[:, 1:-1] = X[:, 2:] - X[:, :-2]
    dy[1:-1] = X[2:] - X[:-2]
    n = np.cross(dx, dy)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(nn, 1e-12)
    n[(n * X).sum(-1) > 0] *= -1
    cos = np.abs((n * X).sum(-1)) / np.maximum(np.linalg.norm(X, axis=-1), 1e-9)
    ok = valid & np.isfinite(X).all(-1) & np.isfinite(n).all(-1) & (nn[..., 0] > 0) & (cos > min_cos)
    ok[0] = ok[-1] = ok[:, 0] = ok[:, -1] = False
    return X[ok], n[ok]


def gravity_rotation(cam_R: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Rotation taking the reconstruction frame to a +y-up frame. Initial up = mean of the cameras'
    image-up axes (-y of an OpenCV camera; photos and video are held upright), refined as the mean
    direction of the surface normals within 25 deg of it (floor, sign-folded ceiling)."""
    up = -cam_R[:, :, 1].mean(0)
    up /= np.linalg.norm(up)
    for _ in range(3):
        c = N @ up
        m = np.abs(c) > np.cos(np.radians(25))
        if m.sum() < 100:
            break
        v = (N[m] * np.sign(c[m])[:, None]).mean(0)
        up = v / np.linalg.norm(v)
    y = np.array([0.0, 1.0, 0.0])
    ax = np.cross(up, y)
    s, c = np.linalg.norm(ax), float(up @ y)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]) / s
    return np.eye(3) + s * K + (1 - c) * K @ K


def voxel(P, N, size=0.02):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.normals = o3d.utility.Vector3dVector(N)
    pc = pc.voxel_down_sample(size)
    Nn = np.asarray(pc.normals)
    return np.asarray(pc.points), Nn / np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-9)


def floor_level(P: np.ndarray, N: np.ndarray, cams: np.ndarray | None = None) -> float | None:
    """Lowest well-supported level of upward-facing horizontal surfaces below the cameras (5 cm bins,
    >= 5 % of those points). Monocular depth puts scattered points below the floor (reflections,
    edge bleeding), so a low height percentile, which LiDAR segmentation uses, is not usable."""
    up = N[:, 1] > 0.9
    if cams is not None and len(cams):
        up &= P[:, 1] < cams[:, 1].min() - 0.3
    y = P[up, 1]
    if len(y) < 200:
        return None
    h, e = np.histogram(y, bins=np.arange(y.min(), y.max() + 0.05, 0.05))
    ok = np.nonzero(h >= 0.05 * len(y))[0]
    if not len(ok):
        return None
    sel = y[(y >= e[ok[0]] - 0.05) & (y < e[ok[0] + 1] + 0.05)]
    return float(np.median(sel))


def _clean(P, N, cams):
    fl = floor_level(P, N, cams)
    if fl is None:
        return None
    keep = P[:, 1] > fl - 0.05
    return P[keep], N[keep]


def _room(seg, label, P, N, infer):
    centers = seg.grid.centers()
    region = centers[G.fill_region(seg.labels, label)]
    other = centers[(seg.labels > 0) & (seg.labels != label)]
    for step, rules in (("observed", WL.V1_0), ("inferred", INFER)) if infer else (("observed", WL.V1_0),):
        w = WL.extract(P, N, P, N, seg.grid.floor_y, None, region, other, rules)
        if w is None:
            continue
        fy, cy, nc = LV.levels(P, N, w.poly_uv, G._rot(w.yaw), seg.grid.floor_y)
        ch = None if cy is None else cy - fy
        w = WL.extract(P, N, P, N, fy, ch, region, other, rules) or w
        return CloudRoom(w, fy, ch, nc, step, label)
    return None


def _camera_labels(seg, cams):
    cij = np.clip(seg.grid.cell(cams[:, [0, 2]]), 0, np.array(seg.grid.shape) - 1)
    return seg.labels[cij[:, 0], cij[:, 1]]


def room_from_cloud(P: np.ndarray, N: np.ndarray, cams: np.ndarray, infer: bool = True) -> CloudRoom | str:
    """P, N gravity-aligned; cams (C, 3) camera centres. The room is the segmented region holding
    most camera positions (a photo set is captured from inside its room). Points more than 5 cm
    below the detected floor are dropped before segmentation."""
    c = _clean(P, N, cams)
    if c is None:
        return "no_floor"
    P, N = c
    seg = G.segment_rooms(P, N, pad_m=1.0, pad_high_m=1.0)
    if seg.labels.max() == 0:
        return "no_region"
    labs = _camera_labels(seg, cams)
    labs = labs[labs > 0]
    label = int(np.bincount(labs).argmax()) if len(labs) else int(np.argmax(np.bincount(seg.labels.ravel())[1:]) + 1)
    return _room(seg, label, P, N, infer) or "no_polygon"


def camera_height_spread(P: np.ndarray, N: np.ndarray, cams: np.ndarray) -> float | None:
    """Std of camera heights above the detected floor (m). Cameras of one handheld capture are at
    nearly one height, so the spread measures how inconsistent the reconstruction's poses are."""
    fl = floor_level(P, N, cams)
    return None if fl is None else float(np.std(cams[:, 1] - fl))


def stability(P: np.ndarray, N: np.ndarray, cams: np.ndarray, nominal: CloudRoom) -> quality.FaceStability:
    """Face stability of room_from_cloud under quality.perturbations; the perturbed runs repeat the
    whole step (floor, segmentation, region choice, walls), where photo/video rooms were seen to flip."""
    zeros = np.zeros_like(cams)

    def rebuild(pc, region, other):
        (Pp, Np), (Cp, _) = pc
        g = room_from_cloud(Pp, Np, Cp)
        return None if isinstance(g, str) else (g.walls.poly_uv, g.walls.yaw)

    return quality.face_stability(nominal.walls.poly_uv, nominal.walls.yaw, rebuild, [(P, N), (cams, zeros)],
                                  np.zeros((0, 2)), np.zeros((0, 2)), subsample=[True, False])


def face_terms(sources: list[str], st: quality.FaceStability | None, measured_sigma: float, inferred_sigma: float,
               registration_sigma: float | None) -> list[dict]:
    """Per-face independent sigma terms for assemble: measurement (or inferred), stability excess
    over the measurement noise, and the reconstruction's registration inconsistency."""
    out = []
    for k, src in enumerate(sources):
        observed = src in ("close", "all")
        base = measured_sigma if observed else inferred_sigma
        t = {"measurement" if observed else "inferred": base}
        if st is not None:
            ex = float(np.sqrt(max(0.0, st.sigma_m[k] ** 2 - base ** 2)))
            if ex > 0:
                t["stability"] = ex
        if registration_sigma:
            t["registration"] = float(registration_sigma)
        out.append(t)
    return out


def rooms_from_cloud(P: np.ndarray, N: np.ndarray, cams: np.ndarray, min_cams: int = 5, infer: bool = True):
    """Every segmented region that min_cams camera positions lie in (a video piece can pass through
    several rooms). Returns ({label: CloudRoom | reason}, camera labels)."""
    c = _clean(P, N, cams)
    if c is None:
        return {}, np.zeros(len(cams), int)
    P, N = c
    seg = G.segment_rooms(P, N, pad_m=1.0, pad_high_m=1.0)
    labs = _camera_labels(seg, cams)
    out = {}
    for label, n in zip(*np.unique(labs[labs > 0], return_counts=True)):
        if n >= min_cams:
            out[int(label)] = _room(seg, int(label), P, N, infer) or "no_polygon"
    return out, labs

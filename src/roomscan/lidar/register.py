"""Rigid registration of one room visit onto another (gravity is shared, so only yaw and the
horizontal translation are searched).

Point-to-plane ICP alone converges only within its correspondence distance (5 cm). Revisits of a
room after 1-2 minutes can be 10-25 cm apart (e22: room 2, 22 cm along one axis), and two visits
often see disjoint height bands (one low, one high), so ICP finds too few correspondences and the
visit is merged unaligned, doubling every wall face.

Hypothesise and verify:
  candidates  per room axis, the strongest peaks of the cross-correlation of wall-facing point
              profiles (each facing side separately, so a face only matches a face seen from the
              same side) within +-MAX_SHIFT, plus zero shift; yaw 0 or the Manhattan-yaw difference
  verify      fitness of the moved visit's vertical-surface points against the reference's
              (fraction within VERIFY_M); the best candidate must beat the unmoved visit by
              MIN_GAIN, otherwise the visit stays where its poses put it
  refine      point-to-plane ICP from the winner, kept only if it does not lower the fitness and
              stays within MAX_REFINE_M
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import open3d as o3d
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from roomscan import geometry as G

reg = o3d.pipelines.registration
MAX_SHIFT = 0.40      # m, search range per axis
BIN = 0.01            # m, profile resolution
MIN_FACE_PTS = 200    # wall-facing points per side for a side to take part
N_PEAKS = 3           # correlation peaks per axis kept as candidates
VERIFY_M = 0.03       # m, inlier distance for the verification fitness
MIN_GAIN = 0.03       # absolute fitness gain over the unmoved visit required to move it
MAX_REFINE_M = 0.05   # ICP may move the verified candidate at most this far


@dataclass
class Alignment:
    T: np.ndarray            # 4x4 world transform applied to the moving visit
    aligned: bool            # True when the visit was moved (its poses were corrected)
    method: str              # "verified+icp" | "verified" | "unmoved"
    fitness: float
    info: dict


def _profiles(UV, NUV, axis, lo, n):
    out = []
    for sign in (1, -1):
        m = sign * NUV[:, axis] > 0.85
        h = np.bincount(np.clip(((UV[m, axis] - lo) / BIN).astype(int), 0, n - 1), minlength=n).astype(float)
        out.append(ndi.gaussian_filter1d(h, 1.5) if m.sum() >= MIN_FACE_PTS else None)
    return out


def _axis_candidates(UVm, NUVm, UVr, NUVr, axis) -> list[float]:
    lo = min(UVm[:, axis].min(), UVr[:, axis].min()) - MAX_SHIFT - 0.1
    n = int((max(UVm[:, axis].max(), UVr[:, axis].max()) + MAX_SHIFT + 0.1 - lo) / BIN) + 1
    k = int(MAX_SHIFT / BIN)
    score = np.zeros(2 * k + 1)
    for hm, hr in zip(_profiles(UVm, NUVm, axis, lo, n), _profiles(UVr, NUVr, axis, lo, n)):
        if hm is not None and hr is not None:
            score += np.correlate(hr / np.linalg.norm(hr), hm / np.linalg.norm(hm), mode="full")[n - 1 - k:n + k]
    pk, _ = find_peaks(np.pad(score, 1), distance=5)          # padding lets edge maxima count
    pk = pk - 1
    pk = pk[(pk > 0) & (pk < 2 * k)]                           # a peak at the search limit is not a match
    best = pk[np.argsort(score[pk])[::-1][:N_PEAKS]] if len(pk) else []
    return [0.0] + [float((i - k) * BIN) for i in best if i != k]


def _transform(yaw, dxz, c):
    """Rotation by yaw about world +y through c, then horizontal translation dxz."""
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0.0, -yaw, 0.0]))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = c - R @ c
    T[0, 3] += dxz[0]; T[2, 3] += dxz[1]
    return T


def _fitness(Pm, tree, T):
    d, _ = tree.query(Pm @ T[:3, :3].T + T[:3, 3], distance_upper_bound=VERIFY_M)
    return float(np.isfinite(d).mean())


def align(moving: o3d.geometry.PointCloud, ref: o3d.geometry.PointCloud, room_yaw: float) -> Alignment:
    Pm, Nm = np.asarray(moving.points), np.asarray(moving.normals)
    Pr, Nr = np.asarray(ref.points), np.asarray(ref.normals)
    vm, vr = np.abs(Nm[:, 1]) < 0.3, np.abs(Nr[:, 1]) < 0.3
    if vm.sum() < 100 or vr.sum() < 100:
        return Alignment(np.eye(4), False, "unmoved", 0.0, {"reason": "too_few_vertical_points"})
    Rm = G._rot(room_yaw)
    c = Pm.mean(0)
    tree = cKDTree(Pr[vr])
    Pv = Pm[vm]
    dyaw = float(np.mod(G.manhattan_yaw(Nr) - G.manhattan_yaw(Nm) + np.pi / 4, np.pi / 2) - np.pi / 4)
    yaws = [0.0] + ([dyaw] if np.radians(0.3) < abs(dyaw) < np.radians(5) else [])
    UVr, NUVr = Pr[vr][:, [0, 2]] @ Rm.T, Nr[vr][:, [0, 2]] @ Rm.T
    f0 = _fitness(Pv, tree, np.eye(4))
    best = (f0, np.eye(4), 0.0, (0.0, 0.0))
    for yaw in yaws:
        T0 = _transform(yaw, (0.0, 0.0), c)
        Q, NQ = Pv @ T0[:3, :3].T + T0[:3, 3], Nm[vm] @ T0[:3, :3].T
        UVm, NUVm = Q[:, [0, 2]] @ Rm.T, NQ[:, [0, 2]] @ Rm.T
        for su, sv in product(_axis_candidates(UVm, NUVm, UVr, NUVr, 0), _axis_candidates(UVm, NUVm, UVr, NUVr, 1)):
            T = _transform(yaw, np.array([su, sv]) @ Rm, c)
            f = _fitness(Pv, tree, T)
            if f > best[0]:
                best = (f, T, yaw, (su, sv))
    f, T, yaw, suv = best
    info = {"fitness_unmoved": round(f0, 3), "fitness_best": round(f, 3), "dyaw_deg": round(np.degrees(yaw), 2),
            "shift_uv": [round(suv[0], 3), round(suv[1], 3)]}
    if f < f0 + MIN_GAIN:
        return Alignment(np.eye(4), False, "unmoved", f0, info)
    r = reg.registration_icp(moving, ref, 0.05, T, reg.TransformationEstimationPointToPlane())
    r = reg.registration_icp(moving, ref, 0.02, r.transformation, reg.TransformationEstimationPointToPlane())
    Ti = r.transformation
    moved = float(np.linalg.norm((Ti[:3, :3] @ c + Ti[:3, 3]) - (T[:3, :3] @ c + T[:3, 3])))
    fi = _fitness(Pv, tree, Ti)
    info.update({"icp_moved_mm": round(1000 * moved, 1), "fitness_icp": round(fi, 3)})
    if moved < MAX_REFINE_M and fi >= f:
        return Alignment(Ti, True, "verified+icp", fi, info)
    return Alignment(T, True, "verified", f, info)

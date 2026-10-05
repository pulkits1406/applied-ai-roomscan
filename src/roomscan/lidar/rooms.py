"""Room-local geometry: segmentation, camera visits per room, per-room aggregated point clouds.

Drift is close to rigid within a room and accumulates between rooms (e09), so a room is measured
from its own visits, each rigidly ICP-aligned to the longest one. A visit that fails alignment is
kept unaligned (and reported) rather than dropped: dropping visits removed the upper-wall coverage
that wall existence depends on (e18).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy import ndimage as ndi

from roomscan import geometry as G
from roomscan.lidar import fusion, register
from roomscan.stray import StrayCapture

reg = o3d.pipelines.registration
ALIGN_MIN_FITNESS = 0.15
SEG_STEP = 5
ROOM_STEP = 3
SEG_PAD_M = 1.0          # > the 0.6 m closing radius: no border erosion of rooms at the scan's edge (v0: 0.3 / 0.0)


@dataclass
class Capture:
    cap: StrayCapture
    T: np.ndarray
    seg: G.Segmentation
    room_of_frame: np.ndarray


@dataclass
class RoomCloud:
    label: int
    visits: list[tuple[int, int]]
    all: o3d.geometry.PointCloud           # every usable observation (topology, levels)
    close: o3d.geometry.PointCloud         # close-range near-normal observations (wall positions)
    alignment: list[dict] = field(default_factory=list)


def prepare(cap: StrayCapture, T: np.ndarray | None = None, seg_pad: tuple[float, float] = (SEG_PAD_M, SEG_PAD_M)) -> Capture:
    """seg_pad = (low, high) empty margin of the segmentation grid; (0.3, 0.0) reproduces the
    pre-fix behaviour (fix-loop 'before', reports/fix_loop)."""
    T = cap.T_wc() if T is None else T
    pc = G.fuse(cap, T, range(0, len(cap), SEG_STEP))
    seg = G.segment_rooms(np.asarray(pc.points), np.asarray(pc.normals), pad_m=seg_pad[0], pad_high_m=seg_pad[1])
    lab_d = ndi.grey_dilation(seg.labels, size=(7, 7))
    cij = np.clip(seg.grid.cell(T[:, [0, 2], 3]), 0, np.array(seg.grid.shape) - 1)
    inside = seg.labels[cij[:, 0], cij[:, 1]]
    return Capture(cap, T, seg, np.where(inside > 0, inside, lab_d[cij[:, 0], cij[:, 1]]))


def visits(c: Capture, label: int, gap_s=1.0, min_s=1.0) -> list[tuple[int, int]]:
    t = c.cap.timestamps
    idx = np.nonzero(c.room_of_frame == label)[0]
    if len(idx) == 0:
        return []
    spans, s = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if t[b] - t[a] > gap_s:
            spans.append((s, a))
            s = b
    spans.append((s, idx[-1]))
    return sorted([sp for sp in spans if t[sp[1]] - t[sp[0]] >= min_s], key=lambda ab: t[ab[0]])


def _crop(pc, mask, seg):
    P = np.asarray(pc.points)
    ij = np.clip(seg.grid.cell(P[:, [0, 2]]), 0, np.array(seg.grid.shape) - 1)
    return pc.select_by_index(np.nonzero(mask[ij[:, 0], ij[:, 1]])[0])


def _icp(pa, ref):
    r = reg.registration_icp(pa, ref, 0.05, np.eye(4), reg.TransformationEstimationPointToPlane())
    r = reg.registration_icp(pa, ref, 0.02, r.transformation, reg.TransformationEstimationPointToPlane())
    ok = r.fitness > ALIGN_MIN_FITNESS and r.inlier_rmse < 0.025 and np.linalg.norm(r.transformation[:3, 3]) < 0.3
    return (r.transformation if ok else np.eye(4)), {"aligned": bool(ok), "fitness": round(r.fitness, 3),
                                                     "shift_mm": round(1000 * float(np.linalg.norm(r.transformation[:3, 3])), 1)}


def room_cloud(c: Capture, label: int, spans=None, margin_m=0.5, registration: str = "icp") -> RoomCloud | None:
    """registration: "icp" (v1.0: point-to-plane ICP from the raw poses) or "coarse" (register.align:
    face-profile hypotheses verified by surface overlap, then ICP)."""
    spans = visits(c, label) if spans is None else spans
    if not spans:
        return None
    t = c.cap.timestamps
    order = sorted(spans, key=lambda ab: t[ab[1]] - t[ab[0]], reverse=True)
    rc = room_cloud_rows(c, label, [list(range(a, b + 1, ROOM_STEP)) for a, b in order], margin_m, registration)
    rc.visits = spans
    return rc


def room_cloud_rows(c: Capture, label: int, visit_rows: list[list[int]], margin_m=0.5, registration: str = "icp") -> RoomCloud:
    """Room cloud from explicit per-visit frame rows; the first visit is the registration reference."""
    near = ndi.binary_dilation(c.seg.labels == label, iterations=int(margin_m / G.GRID))
    per_visit = [(_crop(fusion.fuse(c.cap, c.T, rows, fusion.ALL), near, c.seg),
                  _crop(fusion.fuse(c.cap, c.T, rows, fusion.CLOSE), near, c.seg)) for rows in visit_rows]
    ref = per_visit[0][0]
    room_yaw = G.manhattan_yaw(np.asarray(ref.normals)) if len(ref.points) else 0.0
    all_pc, close_pc, info = o3d.geometry.PointCloud(ref), o3d.geometry.PointCloud(per_visit[0][1]), []
    for pa, pcl in per_visit[1:]:
        if registration == "coarse":
            a = register.align(pa, ref, room_yaw)
            Tm = a.T
            info.append({"aligned": a.aligned, "method": a.method, "fitness": round(a.fitness, 3), **a.info})
        elif len(pa.points) < 500:
            info.append({"aligned": False, "reason": "too_few_points"})
            continue
        else:
            Tm, rec = _icp(pa, ref)
            info.append(rec)
        all_pc += o3d.geometry.PointCloud(pa).transform(Tm)
        close_pc += o3d.geometry.PointCloud(pcl).transform(Tm)
    return RoomCloud(label, [], all_pc.voxel_down_sample(0.02), close_pc.voxel_down_sample(0.02), info)

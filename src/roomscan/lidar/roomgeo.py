"""Per-room geometry with an escalation ladder for rooms whose topology fails (e22).

A room is measured with the v1.0 rules first. Only if no polygon results does the ladder escalate,
each step addressing one failure mode found in e22 and each recorded so the plan can flag it:
  1. "v1_0"        ICP-registered visits, V6 topology rules
  2. "reregistered" visits re-registered by register.align (face-profile hypotheses verified by
                   surface overlap): fixes visits 10-25 cm apart whose wall faces were doubled
  3. "inferred"    as 2, plus room sides with no accepted wall face closed at the free-space
                   boundary; those walls are reported as inferred with a wide interval
Escalating only on failure keeps rooms that already work byte-identical: applied to every room,
re-registration lowered half-to-half repeatability (65 % -> 50 % within gate) and inference grew
correct rooms (room 7 +1.9 m2) (e22).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roomscan import geometry as G, quality
from roomscan.lidar import levels as LV, rooms as RM, walls as WL

MIN_POINTS = 2000
INFER = WL.Rules(infer_missing=True)
LADDER = (("v1_0", "icp", WL.V1_0), ("reregistered", "coarse", WL.V1_0), ("inferred", "coarse", INFER))


@dataclass
class RoomGeometry:
    label: int
    step: str                  # ladder step that produced the polygon
    registration: str
    walls: WL.Walls
    floor_y: float
    ceil_h: float | None
    n_ceil: int
    cloud: RM.RoomCloud
    rules: WL.Rules
    region: np.ndarray            # free-space cell centres (x, z) the outline was built from
    other: np.ndarray

    @property
    def Rm(self):
        return G._rot(self.walls.yaw)

    @property
    def P(self):
        return np.asarray(self.cloud.all.points)

    @property
    def N(self):
        return np.asarray(self.cloud.all.normals)


def _cells(c: RM.Capture, label: int):
    centers = c.seg.grid.centers()
    return centers[G.fill_region(c.seg.labels, label)], centers[(c.seg.labels > 0) & (c.seg.labels != label)]


def _walls(c: RM.Capture, label: int, rc: RM.RoomCloud, rules: WL.Rules):
    seg = c.seg
    P, N = np.asarray(rc.all.points), np.asarray(rc.all.normals)
    CP, CN = np.asarray(rc.close.points), np.asarray(rc.close.normals)
    region, other = _cells(c, label)
    # first pass with a provisional ceiling, then levels inside the polygon, then the final walls
    w = WL.extract(P, N, CP, CN, seg.grid.floor_y, None, region, other, rules)
    if w is None:
        return None
    floor_y, ceil_y, n_ceil = LV.levels(P, N, w.poly_uv, G._rot(w.yaw), seg.grid.floor_y)
    ceil_h = None if ceil_y is None else ceil_y - floor_y
    w = WL.extract(P, N, CP, CN, floor_y, ceil_h, region, other, rules) or w
    return w, floor_y, ceil_h, n_ceil


def measure(c: RM.Capture, label: int, ladder: bool = True) -> RoomGeometry | str:
    """RoomGeometry, or the reason no geometry could be built ("insufficient_points" | "no_polygon")."""
    clouds = {}
    for step, registration, rules in (LADDER if ladder else LADDER[:1]):
        if registration not in clouds:
            clouds[registration] = RM.room_cloud(c, label, registration=registration)
        rc = clouds[registration]
        if rc is None or len(rc.all.points) < MIN_POINTS:
            return "insufficient_points"
        r = _walls(c, label, rc, rules)
        if r is not None:
            return RoomGeometry(label, step, registration, r[0], r[1], r[2], r[3], rc, rules, *_cells(c, label))
    return "no_polygon"


def stability(g: RoomGeometry) -> quality.FaceStability:
    """Face stability of a measured room under the quality.perturbations, same rules and levels."""
    clouds = [(np.asarray(g.cloud.all.points), np.asarray(g.cloud.all.normals)),
              (np.asarray(g.cloud.close.points), np.asarray(g.cloud.close.normals))]

    def rebuild(pc, region, other):
        (P, N), (CP, CN) = pc
        w = WL.extract(P, N, CP, CN, g.floor_y, g.ceil_h, region, other, g.rules)
        return None if w is None else (w.poly_uv, w.yaw)

    return quality.face_stability(g.walls.poly_uv, g.walls.yaw, rebuild, clouds, g.region, g.other)

"""Tier-agnostic Plan assembly: room outlines with per-face uncertainty -> Plan (schema 0.1).

Every tier reduces its evidence to the same inputs: a room polygon whose edges are wall faces, a
1-sigma position uncertainty and an observed/inferred status per face, a relative scale
uncertainty, a ceiling height with its sigma, and openings. Tiers differ only in how those were
obtained and how large the sigmas are. All derived measurements (wall lengths, floor area,
perimeter, wall areas) get their intervals by first-order propagation: a finite-difference
Jacobian of each quantity with respect to each face's offset along its normal, plus the scale
term. Scale is one factor per room for every tier, so it is reported as 'shared:scale'.

Intervals are UNCALIBRATED until a calibration run against laser/tape GT replaces the sigmas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from roomscan.model import (Adjacency, CaptureInfo, Interval, Measurement, Opening, Plan, Plane, Room, Surface, Tier)

Z90 = 1.645


@dataclass
class RoomInput:
    id: str
    polygon: np.ndarray                 # (K, 2) plan coordinates; edge k runs from vertex k to k+1
    face_sigma: list[float]             # 1-sigma offset of each edge's wall face (m)
    face_status: list[str]              # "measured" | "inferred" per edge
    scale_sigma_rel: float              # 1-sigma relative scale error, common to the room
    ceiling_h: float | None
    ceiling_sigma: float
    method: str
    label: str | None = None
    placed: bool = True
    placement_sigma_m: float | None = None
    face_budget: dict = field(default_factory=dict)   # extra named terms for the record (not propagated)


@dataclass
class OpeningInput:
    id: str
    kind: str
    room_ids: list[str]
    wall_index: int                     # edge index in the first room
    width: float
    width_sigma: float
    method: str
    confidence: float
    budget: dict = field(default_factory=dict)


def meas(v, sigma, unit, method, budget=None, status="measured"):
    if v is None:
        return Measurement(value=None, unit=unit, interval=None, status="not_observed", method=method)
    return Measurement(value=float(v), unit=unit, method=method, error_budget={k: float(x) for k, x in (budget or {}).items()},
                       status=status, interval=Interval(lo=float(v - Z90 * sigma), hi=float(v + Z90 * sigma), level=0.9, calibrated=False))


def _area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _quantities(P):
    L = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    return np.concatenate([L, [abs(_area(P)), L.sum()]])


def _shift_edge(P, k, d):
    """Polygon with edge k moved outward by d along its normal (CCW polygon: outward = right of the
    edge direction); the neighbouring edges keep their directions to first order."""
    Q = P.copy()
    a, b = P[k], P[(k + 1) % len(P)]
    t = (b - a) / max(np.linalg.norm(b - a), 1e-12)
    n = np.array([t[1], -t[0]])
    Q[k] = a + d * n
    Q[(k + 1) % len(P)] = b + d * n
    return Q


def propagate(P, face_sigma, scale_rel, eps=1e-4):
    """1-sigma of [wall lengths..., area, perimeter] for a CCW polygon, and the per-term budget."""
    q0 = _quantities(P)
    J = np.stack([(_quantities(_shift_edge(P, k, eps)) - q0) / eps for k in range(len(P))], 1)   # (Q, K)
    face_var = (J ** 2) @ (np.asarray(face_sigma) ** 2)
    scale = q0 * scale_rel * np.r_[np.ones(len(P)), 2.0, 1.0]       # area scales with s^2
    return np.sqrt(face_var + scale ** 2), np.sqrt(face_var), scale, J


def _ccw(P, *per_edge):
    if _area(P) >= 0:
        return (P, *per_edge)
    # reversing the vertex order maps edge k to edge K-2-k (mod K)
    K = len(P)
    idx = [(K - 2 - k) % K for k in range(K)]
    return (P[::-1].copy(), *[[x[i] for i in idx] for x in per_edge])


def room_entities(r: RoomInput) -> tuple[Room, list[Surface]]:
    P, fs, st = _ccw(np.asarray(r.polygon, float), list(r.face_sigma), list(r.face_status))
    K = len(P)
    sig, sig_face, sig_scale, J = propagate(P, fs, r.scale_sigma_rel)
    q = _quantities(P)
    h, zt = r.ceiling_h, (r.ceiling_h if r.ceiling_h is not None else 2.4)
    inferred_any = "inferred" in st
    surfaces, wall_ids = [], []
    for k in range(K):
        p0, p1 = P[k], P[(k + 1) % K]
        L = float(q[k])
        depends = [j for j in range(K) if abs(J[k, j]) > 1e-6] + [k]
        status = "inferred" if any(st[j] == "inferred" for j in depends) else "measured"
        n2 = np.array([p1[1] - p0[1], -(p1[0] - p0[0])]) / max(L, 1e-9)
        sid = f"{r.id}_w{k}"
        wall_ids.append(sid)
        budget = {"face_positions": sig_face[k], "shared:scale": sig_scale[k]}
        hs = r.ceiling_sigma
        surfaces.append(Surface(
            id=sid, kind="wall", room_id=r.id, plane=Plane(normal=(float(n2[0]), float(n2[1]), 0.0), offset=float(-(n2 @ p0))),
            polygon=[(float(p0[0]), float(p0[1]), 0.0), (float(p1[0]), float(p1[1]), 0.0), (float(p1[0]), float(p1[1]), zt), (float(p0[0]), float(p0[1]), zt)],
            area=meas(L * zt, float(np.hypot(sig[k] * zt, L * hs)), "m2", r.method, status=status),
            length=meas(L, sig[k], "m", r.method, budget, status),
            height=meas(h, hs, "m", r.method)))
    ring = [(float(x), float(y), 0.0) for x, y in P]
    area, perim = float(q[K]), float(q[K + 1])
    a_status = "inferred" if inferred_any else "measured"
    a_budget = {"face_positions": sig_face[K], "shared:scale": sig_scale[K]}
    surfaces.append(Surface(id=f"{r.id}_floor", kind="floor", room_id=r.id, plane=Plane(normal=(0, 0, 1.0), offset=0.0), polygon=ring,
                            area=meas(area, sig[K], "m2", r.method, a_budget, a_status)))
    if h is not None:
        surfaces.append(Surface(id=f"{r.id}_ceiling", kind="ceiling", room_id=r.id, plane=Plane(normal=(0, 0, -1.0), offset=float(h)),
                                polygon=[(x, y, float(h)) for x, y, _ in ring], area=meas(area, sig[K], "m2", r.method, a_budget, a_status)))
    room = Room(id=r.id, label=r.label, polygon=[(float(x), float(y)) for x, y in P],
                floor_area=meas(area, sig[K], "m2", r.method, a_budget, a_status),
                perimeter=meas(perim, sig[K + 1], "m", r.method, {"face_positions": sig_face[K + 1], "shared:scale": sig_scale[K + 1]}, a_status),
                ceiling_height=meas(h, r.ceiling_sigma, "m", r.method, {"levels": r.ceiling_sigma} if h is not None else None),
                wall_ids=wall_ids, placement_sigma_m=r.placement_sigma_m, placed=r.placed)
    return room, surfaces


def wall_id_after_ccw(r: RoomInput, k: int) -> str:
    """Surface id of the wall that was edge k of r.polygon before orientation normalisation."""
    K = len(r.polygon)
    return f"{r.id}_w{k if _area(np.asarray(r.polygon, float)) >= 0 else (K - 2 - k) % K}"


def assemble(tier: Tier, rooms: list[RoomInput], openings: list[OpeningInput], capture: CaptureInfo, method: str,
             quality_flags: list[str], drift_correction: str | None = None) -> Plan:
    by_id = {r.id: r for r in rooms}
    R, S = [], []
    for r in rooms:
        room, surf = room_entities(r)
        R.append(room); S += surf
    O, A = [], []
    for o in openings:
        if not set(o.room_ids) <= set(by_id):
            continue
        O.append(Opening(id=o.id, kind=o.kind, wall_id=wall_id_after_ccw(by_id[o.room_ids[0]], o.wall_index), room_ids=o.room_ids,
                         width=meas(o.width, o.width_sigma, "m", o.method, o.budget), detection_confidence=o.confidence))
        if len(o.room_ids) == 2:
            A.append(Adjacency(room_a=o.room_ids[0], room_b=o.room_ids[1], via_opening_id=o.id))
        for room in R:
            if room.id in o.room_ids:
                room.opening_ids.append(o.id)
    fp = sum(x.floor_area.value for x in R)
    fp_sig = float(np.sqrt(sum(((x.floor_area.interval.hi - x.floor_area.interval.lo) / (2 * Z90)) ** 2 for x in R)))
    return Plan(capture=capture, rooms=R, surfaces=S, openings=O, adjacency=A,
                footprint_area=meas(fp, fp_sig, "m2", method, status="inferred" if any(x.floor_area.status == "inferred" for x in R) else "measured"),
                drift_correction=drift_correction, quality_flags=quality_flags)

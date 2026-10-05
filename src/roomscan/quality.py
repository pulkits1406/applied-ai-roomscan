"""Geometry-quality evidence that widens face uncertainty (all tiers; uncalibrated).

A measurement is only as good as the decision that produced the wall it belongs to. Phase 4 showed
rooms whose shape changed completely under a perturbation that should not matter (a grid padding
change moved a video room from 10.4 to 0.07 m2, e26), while their intervals stayed narrow. Here
the wall step is re-run under a fixed set of irrelevant perturbations of its input:
  yaw+ / yaw-   the room's points, normals and free-space cells rotated by +-YAW_DEG about the
                vertical through their centroid (Manhattan-yaw estimate and grid quantisation)
  half_a/half_b two disjoint halves of the points (fixed seed): density and support thresholds
and each nominal wall face is looked up in each perturbed outline (same orientation, same inward
side, overlapping extent). The face's stability term is the RMS of its offsets over the runs that
produced an outline; a face with no counterpart in such an outline counts as MISSING_OFFSET_M
(conflicting evidence about where that wall is). A run that produces no outline at all carries no
information about face positions: it is counted as topology fragility (reported, flagged) rather
than as a positional error; if no run produces an outline, every face gets MISSING_OFFSET_M.
The result is deterministic (fixed perturbations and seed).

Stability measures sensitivity, not accuracy: a room that is consistently wrong (e.g. two rooms
merged because no lintel was seen) is not widened; that failure is reported by quality flags.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from roomscan import geometry as G

YAW_DEG = 1.0
SEED = 20261005
MISSING_OFFSET_M = 0.5
MATCH_MAX_M = 0.5


@dataclass
class FaceStability:
    sigma_m: list[float]            # per nominal edge
    missing_frac: list[float]
    runs_without_outline: int
    n_runs: int
    detail: dict

    @property
    def unstable_topology(self) -> bool:
        """Faces conflict between outlines, or most runs lose the outline."""
        return self.runs_without_outline * 2 >= self.n_runs or any(m >= 0.5 for m in self.missing_frac)

    @property
    def fragile_topology(self) -> bool:
        """At least one run lost the outline (existence depends on density/orientation details)."""
        return self.runs_without_outline > 0


def _rot_y(P, c, deg):
    a = np.radians(deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return (P - c) @ R.T + c, R


def _rot_xz(X, c, deg):
    a = np.radians(deg)
    R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])     # same rotation as _rot_y on (x, z)
    return (X - c) @ R.T + c


def perturbations(clouds: list[tuple[np.ndarray, np.ndarray]], region: np.ndarray, other: np.ndarray,
                  subsample: list[bool] | None = None):
    """Yields (name, perturbed clouds, region, other, back) where back maps world (x, z) of the
    perturbed run to the nominal world frame. subsample[i] = False keeps cloud i whole in the
    half runs (e.g. camera centres)."""
    subsample = subsample or [True] * len(clouds)
    c3 = clouds[0][0].mean(0)
    c2 = c3[[0, 2]]
    for name, deg in (("yaw+", YAW_DEG), ("yaw-", -YAW_DEG)):
        pc = []
        for P, N in clouds:
            Q, R = _rot_y(P, c3, deg)
            pc.append((Q, N @ R.T))
        yield name, pc, _rot_xz(region, c2, deg), _rot_xz(other, c2, deg), (lambda X, d=deg: _rot_xz(X, c2, -d))
    rng = np.random.default_rng(SEED)
    masks = [rng.random(len(P)) < 0.5 for P, _ in clouds]
    for name, keep in (("half_a", True), ("half_b", False)):
        yield name, [(P[m == keep], N[m == keep]) if sub else (P, N) for (P, N), m, sub in zip(clouds, masks, subsample)], \
            region, other, (lambda X: X)


def _edges(poly_uv_nominal_frame):
    """(axis, coord, lo, hi, inward sign) for each edge of a near-rectilinear polygon."""
    P = poly_uv_nominal_frame
    ccw = G.polygon_area(P) > 0
    out = []
    for k in range(len(P)):
        a, b = P[k], P[(k + 1) % len(P)]
        d = b - a
        ax = 0 if abs(d[0]) < abs(d[1]) else 1                       # axis the edge is perpendicular to
        inward = np.array([-d[1], d[0]]) if ccw else np.array([d[1], -d[0]])
        lo, hi = sorted([a[1 - ax], b[1 - ax]])
        out.append((ax, 0.5 * (a[ax] + b[ax]), lo, hi, int(np.sign(inward[ax]))))
    return out


def face_offsets(nominal: list, perturbed: list) -> list[float | None]:
    out = []
    for ax, c, lo, hi, s in nominal:
        best = None
        for ax2, c2, lo2, hi2, s2 in perturbed:
            overlap = min(hi, hi2) - max(lo, lo2)
            if ax2 == ax and s2 == s and overlap > 0.3 * min(hi - lo, hi2 - lo2) and abs(c2 - c) <= MATCH_MAX_M:
                if best is None or abs(c2 - c) < abs(best):
                    best = c2 - c
        out.append(best)
    return out


def face_stability(nominal_poly_uv: np.ndarray, nominal_yaw: float,
                   rebuild: Callable[[list, np.ndarray, np.ndarray], tuple[np.ndarray, float] | None],
                   clouds: list[tuple[np.ndarray, np.ndarray]], region: np.ndarray, other: np.ndarray,
                   subsample: list[bool] | None = None) -> FaceStability:
    """rebuild(clouds, region, other) -> (poly_uv, yaw) or None, using the same rules as the nominal run."""
    Rn = G._rot(nominal_yaw)
    nominal = _edges(nominal_poly_uv)
    offs, lost, detail = [], 0, {}
    for name, pc, reg, oth, back in perturbations(clouds, region, other, subsample):
        r = rebuild(pc, reg, oth)
        if r is None:
            lost += 1
            offs.append([None] * len(nominal)); detail[name] = "no_outline"
            continue
        poly, yaw = r
        xz = back(poly @ G._rot(yaw))                                  # perturbed room frame -> nominal world
        o = face_offsets(nominal, _edges(xz @ Rn.T))
        offs.append(o)
        detail[name] = [None if x is None else round(x, 4) for x in o]
    with_outline = [o for o, (name, d) in zip(offs, detail.items()) if d != "no_outline"]
    sig, miss = [], []
    for k in range(len(nominal)):
        if not with_outline:
            sig.append(MISSING_OFFSET_M); miss.append(1.0)
            continue
        col = [o[k] for o in with_outline]
        d = np.array([MISSING_OFFSET_M if x is None else x for x in col])
        sig.append(float(np.sqrt(np.mean(d ** 2))))
        miss.append(float(np.mean([x is None for x in col])))
    return FaceStability(sig, miss, lost, len(offs), detail)

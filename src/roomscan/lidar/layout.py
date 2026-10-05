"""Property layout: where each room sits in the stitched plan.

Room *geometry* is measured room-locally (drift cancels out of wall-to-wall distances, e09/e18).
Room *placement* is the open drift question (e19). Two placements are available:
  "arkit"     each room where its own longest visit's ARKit poses put it
  "arkit_yaw" same, with every room rotated about its centroid to the common Manhattan yaw
The default follows the e19 evidence (ENGINEERING_LOG); the plane-anchored translation stitch is
not used because it did not reduce placement disagreement in e19.
"""
from __future__ import annotations

import numpy as np

# 1-sigma room placement uncertainty (m), from e19 visit-split halves: raw ARKit placement
# disagreement median 47 mm / max 89 mm between halves -> per-placement sigma ~ 47 / sqrt(2) mm.
PLACEMENT_SIGMA_M = 0.033


def common_yaw(yaws) -> float:
    a = np.mod(np.asarray(yaws), np.pi / 2) * 4
    return float(np.mod(np.arctan2(np.sin(a).mean(), np.cos(a).mean()) / 4, np.pi / 2))


def place(polys_xz: dict, yaws: dict, mode: str = "arkit_yaw") -> dict:
    """polys_xz: room -> world (x, z) polygon at its raw placement. Returns placed polygons."""
    if mode == "arkit":
        return dict(polys_xz)
    target = common_yaw(list(yaws.values()))
    out = {}
    for r, P in polys_xz.items():
        d = np.mod(target - yaws[r] + np.pi / 4, np.pi / 2) - np.pi / 4
        c, s = np.cos(-d), np.sin(-d)
        out[r] = (P - P.mean(0)) @ np.array([[c, -s], [s, c]]).T + P.mean(0)
    return out

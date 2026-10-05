"""Uncertainty propagation and orientation handling of the tier-agnostic plan assembly."""
import numpy as np

from roomscan.assemble import RoomInput, assemble, propagate, wall_id_after_ccw
from roomscan.model import CaptureInfo

RECT = np.array([[0, 0], [4, 0], [4, 3], [0, 3]], float)       # CCW, edge k from vertex k to k+1


def test_face_sigma_propagates_to_lengths_and_area():
    s = 0.01
    sig, _, _, _ = propagate(RECT, [s] * 4, 0.0)
    assert np.allclose(sig[:4], np.sqrt(2) * s, rtol=1e-3)           # a length depends on its two neighbouring faces
    assert np.isclose(sig[4], s * np.sqrt(2 * 4**2 + 2 * 3**2), rtol=1e-3)
    assert np.isclose(sig[5], s * np.sqrt(4 * 2**2), rtol=1e-3)      # each face moves the perimeter by 2x its shift


def test_scale_term_is_relative():
    sig, _, scale, _ = propagate(RECT, [0.0] * 4, 0.01)
    assert np.allclose(scale[:4], [0.04, 0.03, 0.04, 0.03]) and np.isclose(scale[4], 2 * 0.01 * 12)


def test_inferred_face_marks_dependent_walls_and_area():
    r = RoomInput("r1", RECT, [0.007, 0.007, 0.007, 0.10], ["measured"] * 3 + ["inferred"], 0.004, 2.5, 0.007, "test")
    plan = assemble("lidar", [r], [], CaptureInfo(tier="lidar"), "test", [])
    S = {x.id: x for x in plan.surfaces}
    st = [S[w].length.status for w in plan.rooms[0].wall_ids]
    assert st == ["inferred", "measured", "inferred", "inferred"]  # edge 3 itself and its neighbours 0 and 2
    assert plan.rooms[0].floor_area.status == "inferred"
    w0 = S["r1_w0"].length
    assert w0.interval.hi - w0.value > 0.15                        # 1.645 * sqrt(0.007^2 + 0.1^2)


def test_clockwise_input_keeps_face_attributes_and_wall_ids():
    cw = RECT[::-1].copy()                                       # edge k of cw = edge (2 - k) % 4 of RECT
    sig = [0.01, 0.02, 0.03, 0.04]
    r = RoomInput("r1", cw, sig, ["measured"] * 4, 0.0, None, 0.0, "test")
    plan = assemble("photo", [r], [], CaptureInfo(tier="photo"), "test", [])
    S = {x.id: x for x in plan.surfaces}
    k = 1                                                        # cw edge 1 is RECT edge 1 ([4,0]->[4,3] reversed)
    wid = wall_id_after_ccw(r, k)
    assert np.isclose(S[wid].length.value, np.linalg.norm(cw[2] - cw[1]))

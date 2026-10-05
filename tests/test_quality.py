"""Geometry-quality widening (roomscan.quality): semantics, determinism, propagation."""
import numpy as np

from roomscan import quality
from roomscan.assemble import RoomInput, assemble
from roomscan.lidar import walls as WL
from roomscan.model import CaptureInfo
from test_lidar_walls import room

RECT = np.array([[0, 0], [4, 0], [4, 3], [0, 3]], float)
CLOUD = [(np.random.default_rng(0).random((200, 3)), np.tile([1.0, 0, 0], (200, 1)))]


def _as_seen(poly, pc):
    """What a correct rebuild returns: the outline rotated with the perturbed cloud (the yaw runs
    rotate the input about its centroid; the half runs keep it in place)."""
    P0, P1 = CLOUD[0][0], pc[0][0]
    if len(P1) != len(P0):
        return poly
    c = P0.mean(0)[[0, 2]]
    a, b = P0[0, [0, 2]] - c, P1[0, [0, 2]] - c
    ang = np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return (poly - c) @ R.T + c


def _stab(rebuild):
    return quality.face_stability(RECT, 0.0, lambda pc, r, o: (lambda x: None if x is None else (_as_seen(x[0], pc), x[1]))(rebuild(pc, r, o)),
                                  CLOUD, np.zeros((0, 2)), np.zeros((0, 2)))


def test_stable_outline_gives_zero_and_is_deterministic():
    a = _stab(lambda pc, r, o: (RECT, 0.0))
    b = _stab(lambda pc, r, o: (RECT, 0.0))
    assert np.allclose(a.sigma_m, 0, atol=1e-6) and a.sigma_m == b.sigma_m and not a.unstable_topology


def test_conflicting_outline_widens_the_moved_face_only():
    moved = RECT.copy(); moved[1:3, 0] += 0.2                         # the x = 4 wall found 20 cm further out
    s = _stab(lambda pc, r, o: (moved, 0.0))
    assert s.sigma_m[1] > 0.15 and max(s.sigma_m[0], s.sigma_m[2], s.sigma_m[3]) < 1e-6


def test_missing_face_counts_as_large_offset():
    tri = np.array([[0, 0], [4, 0], [4, 3]], float)                 # perturbed outline without the x = 0 wall
    s = _stab(lambda pc, r, o: (tri, 0.0))
    assert s.sigma_m[3] == quality.MISSING_OFFSET_M and s.missing_frac[3] == 1.0 and s.unstable_topology


def test_lost_outline_is_fragility_not_position_error():
    calls = iter(range(10))
    s = _stab(lambda pc, r, o: None if next(calls) == 2 else (RECT, 0.0))   # one of four runs finds no outline
    assert np.allclose(s.sigma_m, 0, atol=1e-6) and s.fragile_topology and not s.unstable_topology


def test_clean_synthetic_room_is_stable_under_the_real_perturbations():
    P, N, region = room(np.random.default_rng(4), clutter=False)
    w = WL.extract(P, N, P, N, 0.0, 2.5, region, np.zeros((0, 2)))

    def rebuild(pc, reg, oth):
        (Q, M), = pc
        r = WL.extract(Q, M, Q, M, 0.0, 2.5, reg, oth)
        return None if r is None else (r.poly_uv, r.yaw)

    s = quality.face_stability(w.poly_uv, w.yaw, rebuild, [(P, N)], region, np.zeros((0, 2)))
    assert max(s.sigma_m) < 0.005 and s.runs_without_outline == 0


def test_stability_term_enters_the_error_budget_and_widens():
    base = RoomInput("r1", RECT, [0.007] * 4, ["measured"] * 4, 0.004, 2.5, 0.007, "t",
                     face_terms=[{"measurement": 0.007}] * 4)
    wide = RoomInput("r1", RECT, [0.007] * 4, ["measured"] * 4, 0.004, 2.5, 0.007, "t",
                     face_terms=[{"measurement": 0.007}, {"measurement": 0.007, "stability": 0.2}, {"measurement": 0.007}, {"measurement": 0.007}])
    pb, pw = (assemble("lidar", [r], [], CaptureInfo(tier="lidar"), "t", []) for r in (base, wide))
    Sb, Sw = ({s.id: s for s in p.surfaces} for p in (pb, pw))
    w0b, w0w = Sb["r1_w0"].length, Sw["r1_w0"].length                # wall 0 spans faces 3 and 1
    assert "faces:stability" in w0w.error_budget and (w0w.interval.hi - w0w.value) > 5 * (w0b.interval.hi - w0b.value)
    assert Sw["r1_w1"].length.interval.hi - Sw["r1_w1"].length.value < 0.05   # wall 1's length does not depend on face 1

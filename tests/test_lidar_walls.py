"""Wall extraction invariants on a synthetic 3.0 m x 4.0 m room (room frame aligned with world)."""
import numpy as np

from roomscan.lidar import walls as WL

W, D, H = 3.0, 4.0, 2.5


def wall_points(rng, x0, x1, z0, z1, normal, n=6000, h=H):
    t = rng.random(n)
    pts = np.stack([x0 + (x1 - x0) * t, rng.random(n) * h, z0 + (z1 - z0) * t], 1)
    pts += rng.normal(0, 0.003, pts.shape) * np.abs(np.array([normal[0], 0, normal[2]]))  # 3 mm noise along normal
    return pts, np.tile(normal, (n, 1))


def room(rng, clutter=True):
    parts = [wall_points(rng, 0, 0, 0, D, np.array([1.0, 0, 0])),        # x = 0 wall, faces +x (inward)
             wall_points(rng, W, W, 0, D, np.array([-1.0, 0, 0])),
             wall_points(rng, 0, W, 0, 0, np.array([0, 0, 1.0])),
             wall_points(rng, 0, W, D, D, np.array([0, 0, -1.0]))]
    if clutter:
        # back face of a full-height cabinet 5 cm in front of the x = W wall, facing that wall (outward)
        parts.append(wall_points(rng, W - 0.05, W - 0.05, 1.0, 2.5, np.array([1.0, 0, 0]), n=4000, h=2.45))
    P = np.concatenate([p for p, _ in parts]); N = np.concatenate([n for _, n in parts])
    gx, gz = np.meshgrid(np.arange(0.3, W - 0.3, 0.05), np.arange(0.3, D - 0.3, 0.05))
    region = np.stack([gx.ravel(), gz.ravel()], 1)
    return P, N, region


def dims(w):
    uv = w.poly_uv
    return sorted([np.ptp(uv[:, 0]), np.ptp(uv[:, 1])])


def test_rectangle_recovered_to_mm():
    rng = np.random.default_rng(0)
    P, N, region = room(rng, clutter=False)
    w = WL.extract(P, N, P, N, 0.0, H, region, np.zeros((0, 2)))
    assert w is not None and len(w.poly_uv) == 4
    assert np.allclose(dims(w), [W, D], atol=0.002)


def test_outward_facing_clutter_does_not_move_the_wall():
    rng = np.random.default_rng(1)
    P, N, region = room(rng, clutter=True)
    w = WL.extract(P, N, P, N, 0.0, H, region, np.zeros((0, 2)))
    assert w is not None and len(w.poly_uv) == 4
    assert np.allclose(dims(w), [W, D], atol=0.002)

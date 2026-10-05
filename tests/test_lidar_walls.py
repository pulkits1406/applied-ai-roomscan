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


def _band(P, N, lo, hi):
    m = (P[:, 1] >= lo) & (P[:, 1] <= hi)
    return P[m], N[m]


def test_missing_wall_is_inferred_not_dropped():
    """A room whose x = 0 wall was never observed: v1.0 rules give no polygon, inference closes it
    at the free-space boundary and labels that face 'inferred'."""
    rng = np.random.default_rng(2)
    P, N, region = room(rng, clutter=False)
    keep = ~((P[:, 0] < 0.05) & (N[:, 0] > 0.5))
    P, N = P[keep], N[keep]
    assert WL.extract(P, N, P, N, 0.0, H, region, np.zeros((0, 2))) is None
    w = WL.extract(P, N, P, N, 0.0, H, region, np.zeros((0, 2)), WL.Rules(infer_missing=True))
    assert w is not None and len(w.poly_uv) == 4 and w.source.count("inferred") == 1


def test_visits_with_disjoint_height_bands_are_reregistered():
    """Two visits of one room, one seeing only 0.2-1.5 m and the other only 1.3-2.4 m, 22 cm apart
    along z (e22 room 2): ICP cannot fix it; register.align recovers the offset."""
    import open3d as o3d
    from roomscan.lidar import register

    rng = np.random.default_rng(3)
    P, N, _ = room(rng, clutter=False)
    floor = np.stack([rng.random(20000) * W, np.zeros(20000), rng.random(20000) * D], 1)
    P, N = np.vstack([P, floor]), np.vstack([N, np.tile([0, 1.0, 0], (20000, 1))])

    def pc(P, N):
        c = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); c.normals = o3d.utility.Vector3dVector(N)
        return c

    ref = pc(*_band(P, N, 0.0, 1.5))
    Pm, Nm = _band(P, N, 1.3, 2.4)
    moving = pc(Pm + np.array([0.0, 0.0, 0.22]), Nm)
    a = register.align(moving, ref, 0.0)
    assert a.aligned
    assert np.allclose(a.T[[0, 2], 3], [0.0, -0.22], atol=0.015)

"""Format invariants of the Stray Scanner loader, checked against the supplied sample capture."""
from pathlib import Path

import numpy as np
import pytest

from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "single_room"
pytestmark = pytest.mark.skipif(not CAP.exists(), reason="sample capture not extracted")


@pytest.fixture(scope="module")
def cap():
    return StrayCapture.load(CAP)


def test_depth_is_millimetres_at_256x192(cap):
    d = cap.depth_m(100)
    assert d.shape == (192, 256)
    assert 0.2 < np.median(d) < 6.0


def test_pose_is_camera_to_world_opencv(cap):
    # Depth from two nearby frames must agree in world coordinates to sensor-noise level.
    i, j = 400, 410
    pw = cap.points_world(i, max_depth=3.0)
    T = cap.T_wc(j)
    pc = (pw - T[:3, 3]) @ T[:3, :3]
    K = cap.K_depth(j)
    pc = pc[pc[:, 2] > 0.1]
    u = np.round(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]).astype(int)
    ok = (u >= 0) & (u < 256) & (v >= 0) & (v < 192)
    r = np.abs(cap.depth_m(j)[v[ok], u[ok]] - pc[ok, 2])
    assert ok.sum() > 5000
    assert np.median(r) < 0.015


def test_world_y_is_up(cap):
    # Floor (dominant low plane) lies below the camera in world y.
    pts = np.concatenate([cap.points_world(i, stride=4) for i in range(0, len(cap), 50)])
    assert np.percentile(pts[:, 1], 5) < np.median(cap.T_wc()[:, 1, 3]) - 1.0


def test_video_missing_first_frame(cap):
    rows = [r for r, _ in cap.iter_rgb(rows={1, 2})]
    assert rows == [1, 2]

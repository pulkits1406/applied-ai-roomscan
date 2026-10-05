"""Real-file input contracts that do not depend on the supplied recordings: iPhone-style HEIC
with EXIF orientation and 35 mm focal, photos without EXIF, MOV rotation signalled in the track
header, Stray exports with a different depth resolution, and the inspection report."""
import numpy as np
import pytest
from PIL import ExifTags, Image

from roomscan import synthetic
from roomscan.inspect import inspect
from roomscan.lidar.build import build
from roomscan.photo.images import load_photo
from roomscan.video.fixtures import write_clip
from roomscan.video.frames import read


def _sensor_image():
    img = np.zeros((300, 400, 3), np.uint8)
    img[:, :100] = 255                     # white strip at the sensor's left edge
    return img


def test_heic_orientation_and_focal(tmp_path):
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    ex = Image.Exif(); ex[ExifTags.Base.Orientation] = 6
    ex.get_ifd(0x8769)[ExifTags.Base.FocalLengthIn35mmFilm] = 26
    Image.fromarray(_sensor_image()).save(tmp_path / "IMG_0001.HEIC", exif=ex.tobytes())
    p = load_photo(tmp_path / "IMG_0001.HEIC")
    assert p.rgb.shape[:2] == (400, 300)                                  # rotated once to portrait
    assert p.rgb[0, :, 0].mean() > 200                                    # sensor-left is now the top
    assert abs(p.f_px - 26 / 43.27 * 500) < 1e-6


def test_photo_without_focal_is_accepted_without_intrinsics(tmp_path):
    Image.fromarray(_sensor_image()).save(tmp_path / "a.jpg")
    p = load_photo(tmp_path / "a.jpg")
    assert p.f_px is None and p.K is None


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_mov_rotation_matches_player_view(tmp_path, rotation):
    import cv2
    frames = [_sensor_image()[:240, :320].copy() for _ in range(10)]
    path = write_clip(tmp_path / f"r{rotation}.mov", frames, rotation=rotation)
    cap = cv2.VideoCapture(str(path)); ok, player = cap.read(); cap.release()   # OpenCV auto-rotation = player view
    kf, info = read(path, fps=5, rotate="auto")
    assert info["rotation_deg"] == rotation
    ours = cv2.cvtColor(kf[0].rgb, cv2.COLOR_RGB2BGR)
    assert ours.shape == player.shape and np.abs(ours.astype(int) - player.astype(int)).mean() < 3


def test_stray_with_other_depth_resolution(tmp_path):
    cap = synthetic.write(tmp_path / "syn", fps=10, depth_wh=(192, 144))
    rep = inspect(cap)
    assert rep["depth_wh"] == (192, 144) and rep["rgb_wh"] == (1920, 1440)
    assert all(v.startswith("ok") or v.startswith("absent") or v.startswith("see") for v in rep["checks"].values()), rep["checks"]
    plan, _ = build(cap)
    S = {s.id: s for s in plan.surfaces}
    lengths = sorted(round(S[w].length.value, 2) for r in plan.rooms for w in r.wall_ids)
    assert len(plan.rooms) == 2 and lengths == [2.5, 2.5, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0]

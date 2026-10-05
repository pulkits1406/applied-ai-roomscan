"""Inspect a raw capture before running a pipeline:  uv run roomscan-inspect <path> [--json out.json]

Detects the input kind (Stray export / folder of room photo folders / single video file) and
reports every format fact the pipelines rely on, re-checking the conventions that were verified
only on the supplied recordings (e01, e05). Items marked CHECK disagree with what the pipeline
assumes and must be resolved before trusting a plan. Nothing here modifies the capture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mov", ".mp4", ".m4v"}


def _reproj_residual(cap, i, j, flip: bool) -> float | None:
    """Median depth disagreement (m) when frame i's depth is projected into frame j, reading the
    poses with OpenCV camera axes (flip=False, the verified Stray convention) or ARKit axes."""
    F = np.diag([1.0, -1.0, -1.0]) if flip else np.eye(3)
    Ti, Tj = cap.T_wc(i), cap.T_wc(j)
    d = cap.depth_m(i)[::2, ::2]
    K = cap.K_depth(i); K[:2] /= 2
    v, u = np.nonzero((cap.confidence(i)[::2, ::2] == 2) & (d > 0.2) & (d < 3.0))
    z = d[v, u]
    pc = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1) @ F.T
    pw = pc @ Ti[:3, :3].T + Ti[:3, 3]
    q = ((pw - Tj[:3, 3]) @ Tj[:3, :3]) @ F.T
    q = q[q[:, 2] > 0.1]
    Kj = cap.K_depth(j)
    W, H = cap.depth_wh
    uu = np.round(Kj[0, 0] * q[:, 0] / q[:, 2] + Kj[0, 2]).astype(int)
    vv = np.round(Kj[1, 1] * q[:, 1] / q[:, 2] + Kj[1, 2]).astype(int)
    ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
    if ok.sum() < 500:
        return None
    return float(np.median(np.abs(cap.depth_m(j)[vv[ok], uu[ok]] - q[ok, 2])))


def inspect_stray(root: Path) -> dict:
    from roomscan.stray import VIDEO_ROW_OFFSET, StrayCapture
    cap = StrayCapture.load(root)
    t = cap.timestamps
    out = {"kind": "stray", "rows": len(cap), "duration_s": round(float(t[-1] - t[0]), 1),
           "odometry_columns": list(cap.odometry.columns), "imu_rows": len(cap.imu),
           "rgb_wh": cap.rgb_wh, "depth_wh": cap.depth_wh, "checks": {}}
    d = cap.depth_m(len(cap) // 2)
    conf = np.unique(cap.confidence(len(cap) // 2))
    out["depth_median_m"] = round(float(np.median(d[d > 0])), 3)
    out["confidence_values"] = conf.tolist()
    ck = out["checks"]
    ck["depth_in_metres_after_mm_scaling"] = "ok" if 0.2 < out["depth_median_m"] < 8 else "CHECK: depth units"
    ck["confidence_levels"] = "ok" if set(conf.tolist()) <= {0, 1, 2} else "CHECK: unexpected confidence values"
    need = {"timestamp", "x", "y", "z", "qx", "qy", "qz", "qw", "fx", "fy", "cx", "cy"}
    ck["odometry_columns"] = "ok" if need <= set(cap.odometry.columns) else f"CHECK: missing {sorted(need - set(cap.odometry.columns))}"
    i = len(cap) // 3
    j = min(i + 10, len(cap) - 1)
    r_cv, r_ar = _reproj_residual(cap, i, j, False), _reproj_residual(cap, i, j, True)
    out["pose_reprojection_median_m"] = {"opencv_axes": r_cv, "arkit_axes": r_ar}
    ck["pose_convention_T_wc_opencv"] = ("ok" if r_cv is not None and r_cv < 0.02 and (r_ar is None or r_cv < r_ar)
                                         else "CHECK: poses do not match the verified convention (e01)")
    Tw = cap.T_wc()
    ck["world_y_up"] = "ok" if np.median(Tw[:, 1, 1]) < 0 else "CHECK: camera y axis is not mostly downward in world (+y up assumed)"
    vid = root / "rgb.mp4"
    if vid.exists():
        c = cv2.VideoCapture(str(vid))
        n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release()
        out["video_frames_declared"] = n
        ck["video_row_offset"] = ("ok (declared count = rows; e05: first frame does not decode)" if n in (len(cap), len(cap) - VIDEO_ROW_OFFSET)
                                  else f"CHECK: video frames {n} vs odometry rows {len(cap)} (VIDEO_ROW_OFFSET={VIDEO_ROW_OFFSET})")
    else:
        ck["video"] = "absent (LiDAR tier does not need it)"
    ck["upper_walls"] = "see roomscan-lidar flag upper_walls_not_observed"
    return out


def inspect_photos(root: Path) -> dict:
    from PIL import ExifTags, Image

    from roomscan.photo.images import EXIF_IFD, FULL_FRAME_DIAG_MM, load_photo, room_folders
    rooms = {}
    for name, paths in room_folders(root).items():
        rows = []
        for p in paths:
            ph = load_photo(p, max_side=10_000)
            im = Image.open(p) if p.suffix.lower() not in (".heic", ".heif") else None
            ex = (im or _heif(p)).getexif()
            ifd = ex.get_ifd(EXIF_IFD)
            f35 = ifd.get(ExifTags.Base.FocalLengthIn35mmFilm)
            rows.append({"file": p.name, "upright_wh": [ph.rgb.shape[1], ph.rgb.shape[0]], "orientation_tag": ex.get(ExifTags.Base.Orientation),
                         "f35_mm": f35, "focal_mm": float(ifd[ExifTags.Base.FocalLength]) if ExifTags.Base.FocalLength in ifd else None,
                         "lens": ifd.get(ExifTags.Base.LensModel), "model": ex.get(ExifTags.Base.Model),
                         "hfov_deg": None if ph.f_px is None else round(ph.fov_x_deg, 1),
                         "camera": None if not f35 else ("ultra-wide" if f35 <= 16 else "main" if f35 <= 30 else "tele")})
        n = len(rows)
        rooms[name] = {"photos": rows, "check": ("ok" if 2 <= n <= 8 else f"CHECK: {n} photos (contract 2-8)")
                       + ("" if all(r["f35_mm"] for r in rows) else "; CHECK: photo(s) without 35 mm focal")}
    return {"kind": "photo_folders", "rooms": rooms, "diag_mm": FULL_FRAME_DIAG_MM}


def _heif(p):
    import pillow_heif
    pillow_heif.register_heif_opener()
    from PIL import Image
    return Image.open(p)


def inspect_video(path: Path) -> dict:
    import av
    out = {"kind": "video", "file": path.name}
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        cc = s.codec_context
        out.update({"codec": cc.name, "wh": [cc.width, cc.height], "pix_fmt": cc.pix_fmt, "fps_avg": float(s.average_rate or 0),
                    "duration_s": None if s.duration is None else round(float(s.duration * s.time_base), 2),
                    "color_transfer": str(getattr(cc, "color_trc", None)), "container_metadata": dict(c.metadata),
                    "stream_metadata": dict(s.metadata)})
    cap = cv2.VideoCapture(str(path))
    out["rotation_meta_deg"] = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    cap.release()
    focal = {k: v for k, v in {**out["container_metadata"], **out["stream_metadata"]}.items() if "focal" in k.lower()}
    out["focal_metadata"] = focal or None
    out["checks"] = {
        "orientation": "ok (rotation metadata present)" if out["rotation_meta_deg"] else "CHECK: no rotation metadata; pass --rotate if frames are not upright",
        "bit_depth": "ok (8-bit)" if "10" not in (out["pix_fmt"] or "") else "NOTE: 10-bit (HDR) video; frames are converted to 8-bit RGB, geometry unaffected [BLOCKED: verify on device]",
        "intrinsics": "focal metadata found (not used yet)" if focal else "none in file: field of view is estimated from the frames"}
    return out


def inspect(path: str | Path) -> dict:
    p = Path(path)
    if p.is_dir() and (p / "odometry.csv").exists():
        return inspect_stray(p)
    if p.is_file() and p.suffix.lower() in VIDEO_EXT:
        return inspect_video(p)
    if p.is_dir():
        return inspect_photos(p)
    raise SystemExit(f"unrecognised capture: {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path"); ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    rep = inspect(a.path)
    txt = json.dumps(rep, indent=1, default=str)
    if a.json:
        Path(a.json).write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()

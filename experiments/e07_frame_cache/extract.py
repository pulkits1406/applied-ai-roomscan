"""Extract a keyframe cache shared by the photo/video-tier experiments.

cache/frames/<capture>/
  rgb/<row:06d>.jpg   native sensor orientation (landscape), 960x720
  frames.json         per frame: row, t, K (for 960x720), T_wc (OpenCV, ARKit world, y up),
                      rot_to_upright (cv2 rotate code or null), pitch_deg, median LiDAR depth,
                      sharpness (variance of Laplacian), angular speed (deg/s)
LiDAR depth/poses in this cache are pseudo ground truth only; RGB-only methods must not read them.
"""
import json, sys
from pathlib import Path
import numpy as np, cv2
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]
STEP = 5; W, H = 960, 720

def upright_code(R):
    up = R.T @ np.array([0, 1.0, 0]); ux, uy = up[0], up[1]
    if -uy >= abs(ux): return None
    if uy >= abs(ux): return cv2.ROTATE_180
    return cv2.ROTATE_90_COUNTERCLOCKWISE if ux > 0 else cv2.ROTATE_90_CLOCKWISE

for name in sys.argv[1:] or ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]:
    cap = StrayCapture.load(ROOT / name); Ts = cap.T_wc(); t = cap.timestamps
    out = ROOT / "cache/frames" / name; (out / "rgb").mkdir(parents=True, exist_ok=True)
    rows = set(range(STEP, len(cap), STEP))
    meta = []
    for r, bgr in cap.iter_rgb(rows):
        img = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / "rgb" / f"{r:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        K = cap.K_rgb(r); K[:2] *= W / 1920
        R = Ts[r, :3, :3]; fwd = R[:, 2]
        a, b = max(r - 3, 0), min(r + 3, len(cap) - 1)
        dR = Ts[a, :3, :3].T @ Ts[b, :3, :3]
        ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))) / max(t[b] - t[a], 1e-3)
        d = cap.depth_m(r)
        meta.append({"row": r, "t": float(t[r]), "K": K.round(4).tolist(), "T_wc": Ts[r].round(6).tolist(),
                     "rot_to_upright": upright_code(R), "pitch_deg": round(float(np.degrees(np.arcsin(fwd[1]))), 2),
                     "median_depth_m": round(float(np.median(d)), 3),
                     "sharpness": round(float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()), 1),
                     "ang_speed_dps": round(float(ang), 2)})
    (out / "frames.json").write_text(json.dumps({"capture": name, "image_wh": [W, H], "step": STEP, "frames": meta}))
    print(name, len(meta), "frames")

"""Is decoded video frame k the RGB image for depth/odometry row k?

Score = correlation between RGB edge magnitude and depth edge magnitude (both at 256x192) for
video-frame offsets -3..+3. Motion makes a misaligned pair decorrelate.
"""
import json
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e05_rgb_depth_sync/out"; OUT.mkdir(parents=True, exist_ok=True)
OFFS = list(range(-3, 4))
def edges(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0); gy = cv2.Sobel(img, cv2.CV_32F, 0, 1); m = np.hypot(gx, gy)
    return (m - m.mean()) / (m.std() + 1e-9)
res = {}
for name in ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]:
    d = ROOT / name
    n = len(list((d / "depth").glob("*.png")))
    targets = set(np.linspace(10, n - 10, 60).astype(int).tolist())
    want = {t + o for t in targets for o in OFFS}
    cap = cv2.VideoCapture(str(d / "rgb.mp4")); frames = {}; k = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if k in want: frames[k] = edges(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (256, 192), interpolation=cv2.INTER_AREA).astype(np.float32))
        k += 1
    scores = {o: [] for o in OFFS}
    for t in targets:
        de = edges(np.log(cv2.imread(str(d / "depth" / f"{t:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)))
        for o in OFFS:
            if t + o in frames: scores[o].append(float((de * frames[t + o]).mean()))
    best = [OFFS[int(np.argmax(sc))] for sc in zip(*[scores[o] for o in OFFS])]
    res[name] = {"decoded_frames": k, "odometry_rows": n,
                 "mean_corr_by_offset": {o: round(float(np.mean(v)), 4) for o, v in scores.items()},
                 "argmax_offset_counts": {o: best.count(o) for o in OFFS}}
    print(name, json.dumps(res[name]))
(OUT / "sync.json").write_text(json.dumps(res, indent=1))

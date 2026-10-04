"""Contact sheet of evenly spaced RGB frames + depth/confidence for one capture."""
import sys
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parents[2]
name = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
d = ROOT / name
cap = cv2.VideoCapture(str(d / "rgb.mp4"))
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
idx = set(np.linspace(0, N - 1, n).astype(int).tolist())
tiles, i = [], 0
while True:
    ok, f = cap.read()
    if not ok: break
    if i in idx:
        t = cv2.resize(f, (320, 240))
        dep = cv2.imread(str(d / "depth" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
        dv = cv2.applyColorMap(np.clip(dep / 6000 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        dv = cv2.resize(dv, (320, 240), interpolation=cv2.INTER_NEAREST)
        t = np.hstack([t, dv])
        cv2.putText(t, str(i), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(t)
    i += 1
cols = 4
while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
out = ROOT / "experiments/e00_data_survey/out" / f"contact_{name}.jpg"
cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80]); print(out, "decoded", i)

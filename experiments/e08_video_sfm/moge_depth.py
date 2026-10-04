"""Cache MoGe-2 metric depth (known FOV) for given keyframe rows.

Runs in its own process: torch and pycolmap ship separate OpenMP runtimes that abort when loaded together.
Usage: uv run python experiments/e08_video_sfm/moge_depth.py <capture> <rows.json>
Writes cache/e08/moge/<capture>/<row:06d>.npy (float16, 360x480 landscape, NaN outside MoGe mask).
"""
import json
import sys
import time

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel

from common import CACHE, FRAMES, load_frames

MOGE_WH = (480, 360)


def main():
    capture, rows_path = sys.argv[1], sys.argv[2]
    rows = json.load(open(rows_path))
    frames = {f["row"]: f for f in load_frames(capture)}
    out_dir = CACHE / "moge" / capture
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (out_dir / f"{r:06d}.npy").exists()]
    if not todo:
        return
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(dev).eval()
    secs = []
    for i, row in enumerate(todo):
        K = frames[row]["K"]
        bgr = cv2.imread(str(FRAMES / capture / "rgb" / f"{row:06d}.jpg"))
        img = cv2.rotate(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cv2.ROTATE_90_CLOCKWISE)
        # After a 90-degree rotation the upright image width spans the sensor's y axis.
        fov_x = float(np.degrees(2 * np.arctan(bgr.shape[0] / (2 * K[1][1]))))
        x = torch.tensor(img / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
        t0 = time.time()
        with torch.no_grad():
            o = model.infer(x, fov_x=fov_x)
        d = o["depth"].float().cpu().numpy()
        secs.append(time.time() - t0)
        d[~(o["mask"].cpu().numpy() > 0.5)] = np.nan
        d = cv2.resize(cv2.rotate(d, cv2.ROTATE_90_COUNTERCLOCKWISE), MOGE_WH, interpolation=cv2.INTER_LINEAR)
        np.save(out_dir / f"{row:06d}.npy", d.astype(np.float16))
        if i % 20 == 0:
            print(f"moge {capture} {i}/{len(todo)} {np.median(secs):.2f}s/frame", flush=True)
    (out_dir / "timing.json").write_text(json.dumps({"n": len(secs), "sec_per_frame_median": float(np.median(secs))}))


if __name__ == "__main__":
    main()

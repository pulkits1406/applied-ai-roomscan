"""Cache MoGe-2 metric depth (true FOV supplied) for every Nth keyframe (project venv, MPS).

Usage: uv run python experiments/e20_video_tracking/moge_depth.py <capture> --stride 2 [--max-minutes 18]
Same model, settings and format as e08/moge_depth.py (float16 360x480 landscape, NaN outside the MoGe
mask); rows already in cache/e08/moge are reused, new rows go to cache/e20/moge/<capture>/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, MOGE_WH, img_path, load_frames, mem_ok, moge_path, wait_for_memory  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-minutes", type=float, default=18)
    a = ap.parse_args()
    frames = load_frames(a.capture)[::a.stride]
    todo = [f for f in frames if moge_path(a.capture, f["row"]) is None]
    print(f"{len(frames)} rows, {len(todo)} to do", flush=True)
    if not todo:
        return
    print(wait_for_memory())
    out = CACHE / "moge" / a.capture
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("mps")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(dev).eval()
    secs, t_start = [], time.time()
    for i, f in enumerate(todo):
        K = f["K"]
        bgr = cv2.imread(str(img_path(a.capture, f["row"])))
        img = cv2.rotate(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cv2.ROTATE_90_CLOCKWISE)
        # After the 90-degree rotation the upright image width spans the sensor's y axis.
        fov_x = float(np.degrees(2 * np.arctan(bgr.shape[0] / (2 * K[1][1]))))
        x = torch.tensor(img / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
        t0 = time.time()
        with torch.no_grad():
            o = model.infer(x, fov_x=fov_x)
        d = o["depth"].float().cpu().numpy()
        d[~(o["mask"].cpu().numpy() > 0.5)] = np.nan
        secs.append(time.time() - t0)
        del o, x
        d = cv2.resize(cv2.rotate(d, cv2.ROTATE_90_COUNTERCLOCKWISE), MOGE_WH, interpolation=cv2.INTER_LINEAR)
        np.save(out / f"{f['row']:06d}.npy", d.astype(np.float16))
        if i % 25 == 0:
            torch.mps.empty_cache()
            ok, msg = mem_ok()
            print(f"moge {a.capture} {i}/{len(todo)} {np.median(secs):.2f}s/frame {msg}", flush=True)
            if not ok:
                wait_for_memory(max_wait_s=300)
        if time.time() - t_start > 60 * a.max_minutes:
            print("time budget reached; rerun to resume", flush=True)
            break
    tp = out / "timing.json"
    tim = json.loads(tp.read_text()) if tp.exists() else {"n": 0, "sec": 0.0}
    tim["n"] += len(secs)
    tim["sec"] += float(sum(secs))
    tim["sec_per_frame"] = tim["sec"] / max(tim["n"], 1)
    tp.write_text(json.dumps(tim))
    del model
    torch.mps.empty_cache()


if __name__ == "__main__":
    main()

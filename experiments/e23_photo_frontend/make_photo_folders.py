"""Simulated photo-tier input from a walkthrough: one folder per room, K upright JPEG stills with
the EXIF a phone writes (integer 35 mm-equivalent focal), nothing else.

Simulation choices (PSEUDO-GT setup, not a real photo capture):
- frames from the e07 cache (960x720 video stills; real photos are 12 MP, models run at ~500 px);
- photo-like filter as e12 (|pitch| < 25 deg, sharp, < 40 deg/s; median LiDAR depth > 1.6 m is a
  stand-in for "the user steps back to frame the room" and uses hidden depth only for selection);
- room = production LiDAR segmentation label of the camera position (folder name r<label>), so
  the photo plan's rooms line up with the LiDAR pseudo-GT rooms by label;
- per room, K frames chosen either greedily for viewing-direction diversity ("diverse": a user
  photographing each wall/corner, starting from the sharpest) or as an overlapping sweep ("sweep":
  consecutive photos within one visit whose viewing directions differ by >= SWEEP_STEP_DEG, i.e.
  ~50 % overlap at the 48 deg portrait field of view).
The pipeline sees only <out>/<room>/*.jpg; `_sim_meta.json` (rows, true K) sits outside the room
folders for evaluation.

    uv run python experiments/e23_photo_frontend/make_photo_folders.py [capture] [K] [out|''] [diverse|sweep]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from roomscan.lidar import rooms as RM
from roomscan.photo.images import write_photo
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]


SWEEP_STEP_DEG = 22.0


def _sweep(ms, yaw, K, visit_of):
    """Longest overlapping sweep inside one visit: walk frames in time order, keep a frame when its
    viewing direction is >= SWEEP_STEP_DEG from the last kept one."""
    best = []
    for v in sorted(set(visit_of)):
        idx = [i for i in sorted(range(len(ms)), key=lambda i: ms[i]["row"]) if visit_of[i] == v]
        sel = idx[:1]
        for i in idx[1:]:
            if abs(np.angle(np.exp(1j * (yaw[i] - yaw[sel[-1]])))) >= np.radians(SWEEP_STEP_DEG):
                sel.append(i)
            if len(sel) == K:
                break
        if len(sel) > len(best):
            best = sel
    return best


def main(capture="single_scan_with_ceiling", K="6", out=None, mode="diverse"):
    K = int(K)
    out = Path(out or ROOT / (f"cache/photo_sim/{capture}_K{K}" + ("" if mode == "diverse" else f"_{mode}")))
    frames = json.loads((ROOT / f"cache/frames/{capture}/frames.json").read_text())["frames"]
    c = RM.prepare(StrayCapture.load(ROOT / capture))
    sharp_thr = np.percentile([m["sharpness"] for m in frames], 40 if mode == "diverse" else 20)
    cand = {}
    for m in frames:
        wide = m["median_depth_m"] >= 1.6 or mode == "sweep"        # a sweep turns past near walls too
        if abs(m["pitch_deg"]) > 25 or not wide or m["sharpness"] < sharp_thr or m["ang_speed_dps"] > 40:
            continue
        lab = int(c.room_of_frame[m["row"]])
        if lab > 0:
            cand.setdefault(lab, []).append(m)
    meta = {"capture": capture, "K": K, "mode": mode, "rooms": {}}
    for lab, ms in sorted(cand.items()):
        if len(ms) < 2:
            continue
        fwd = np.array([np.array(m["T_wc"])[[0, 2], 2] for m in ms])        # camera +z (forward), horizontal
        yaw = np.arctan2(fwd[:, 1], fwd[:, 0])
        if mode == "sweep":
            spans = RM.visits(c, lab)
            visit_of = np.array([next((k for k, (a, b) in enumerate(spans) if a <= m["row"] <= b), -1) for m in ms])
            sel = _sweep(ms, yaw, K, visit_of)
            if len(sel) < 3:
                continue
        else:
            sel = [int(np.argmax([m["sharpness"] for m in ms]))]
            while len(sel) < min(K, len(ms)):
                d = np.min(np.abs(np.angle(np.exp(1j * (yaw[:, None] - yaw[None, sel])))), 1)
                d[sel] = -1
                sel.append(int(np.argmax(d)))
        room = out / f"r{lab}"
        room.mkdir(parents=True, exist_ok=True)
        rows = []
        for n, i in enumerate(sorted(sel, key=lambda i: ms[i]["row"])):
            m = ms[i]
            bgr = cv2.imread(str(ROOT / f"cache/frames/{capture}/rgb/{m['row']:06d}.jpg"))
            up = cv2.cvtColor(cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_BGR2RGB)   # every frame in these captures
            f_px = float(m["K"][0][0])                                                     # fx (landscape) == fy upright
            write_photo(room / f"IMG_{n:04d}.jpg", up, f_px)
            rows.append({"row": m["row"], "file": f"IMG_{n:04d}.jpg", "K_landscape": m["K"], "T_wc": m["T_wc"]})
        meta["rooms"][f"r{lab}"] = rows
        print(f"r{lab}: {len(rows)} photos of {len(ms)} candidates")
    (out / "_sim_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])

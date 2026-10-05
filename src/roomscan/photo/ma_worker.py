"""MapAnything multi-view inference, run in the isolated mapanything env (roomscan.envs):

    <mapanything python> -m roomscan.photo.ma_worker <job.json> <out.npz>

job.json: {"images": [paths...], "max_side": int} or {"jobs": [{"images": [...], "out": path}, ...]}
(one model load for many view sets). Images are
loaded upright with the same EXIF handling as the project (roomscan.photo.images, numpy + PIL
only). Output per view at model resolution: depth_z, mask, camera pose (cam->world, OpenCV axes,
world = first camera), predicted intrinsics, and the input size. bf16 backbone as in e16.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("HF_HOME", os.environ.get("ROOMSCAN_HF_HOME_MAPANYTHING", str(ROOT / "cache/e13/hf")))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
HF_ID = "facebook/map-anything-apache"
HF_REVISION = "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a"
RES_SET = 504


def main(job_path, out_path=None):
    import torch
    from mapanything.models import MapAnything

    job = json.loads(Path(job_path).read_text())
    jobs = job["jobs"] if "jobs" in job else [{"images": job["images"], "out": out_path}]
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    m = MapAnything.from_pretrained(HF_ID, revision=HF_REVISION)
    m.encoder.to(torch.bfloat16); m.info_sharing.to(torch.bfloat16)        # e16: <= 0.12 pp effect, -2.3 GB
    m = m.to(dev).eval()
    t_load = time.time() - t0
    for j in jobs:
        if not Path(j["out"]).exists():
            run_one(m, j["images"], j["out"], job.get("max_side", 1024), t_load)
            if dev.type == "mps":
                torch.mps.empty_cache()


def run_one(m, images, out_path, max_side, t_load):
    import torch
    from mapanything.utils.image import preprocess_inputs

    from roomscan.photo.images import load_photo

    photos = [load_photo(p, max_side) for p in images]
    t0 = time.time()
    views = []
    for ph in photos:
        v = {"img": ph.rgb}
        if ph.K is not None:
            v["intrinsics"] = torch.tensor(ph.K, dtype=torch.float32)
        views.append(v)
    pv = preprocess_inputs(views, resolution_set=RES_SET)
    with torch.no_grad():
        preds = m.infer(pv, memory_efficient_inference=False, use_amp=True, amp_dtype="bf16",
                        apply_mask=True, mask_edges=True, apply_confidence_mask=False)
    out = {k: [] for k in ("depth", "mask", "pose", "K")}
    for p in preds:
        out["depth"].append(p["depth_z"][0, ..., 0].float().cpu().numpy())
        out["mask"].append(p["mask"][0, ..., 0].cpu().numpy() > 0.5)
        out["pose"].append(p["camera_poses"][0].float().cpu().numpy())
        out["K"].append(p["intrinsics"][0].float().cpu().numpy())
    np.savez_compressed(out_path, depth=np.stack(out["depth"]), mask=np.stack(out["mask"]), pose=np.stack(out["pose"]),
                        K=np.stack(out["K"]), in_hw=np.array([ph.rgb.shape[:2] for ph in photos]),
                        sec_load=t_load, sec_infer=time.time() - t0)


if __name__ == "__main__":
    main(*sys.argv[1:])

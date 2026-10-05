"""Per-room model inference for the photo tier, cached on disk by content.

MoGe-2 (project env, in process): per photo a metric point map in the camera frame (OpenCV axes),
with the EXIF field of view when available. MapAnything (isolated env, subprocess via
roomscan.envs): joint multi-view depth + camera poses for the room's photos. Only one GPU model is
resident at a time: MoGe is released before MapAnything starts.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from roomscan.envs import python_for
from roomscan.photo.images import Photo

MOGE_ID = "Ruicheng/moge-2-vitl-normal"
MOGE_REVISION = "cb0e8bbd6b1e243589717c78e750b1ba4c093acf"


def _key(paths, tag) -> str:
    h = hashlib.sha1(tag.encode())
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


class MoGe:
    """MoGe-2 holder; the weights load on first use, so fully cached runs never touch the GPU."""

    def __init__(self):
        self.model = None

    def _load(self):
        import torch
        from moge.model.v2 import MoGeModel
        self.torch = torch
        self.dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = MoGeModel.from_pretrained(MOGE_ID, revision=MOGE_REVISION).to(self.dev).eval()

    def __call__(self, ph: Photo) -> dict:
        if self.model is None:
            self._load()
        x = self.torch.tensor(ph.rgb / 255.0, dtype=self.torch.float32, device=self.dev).permute(2, 0, 1)
        with self.torch.no_grad():
            o = self.model.infer(x, fov_x=ph.fov_x_deg)
        pts = o["points"].float().cpu().numpy()
        mask = (o["mask"].cpu().numpy() > 0.5) & np.isfinite(pts).all(-1)
        K = o["intrinsics"].float().cpu().numpy()
        return {"points": np.where(mask[..., None], pts, np.nan).astype(np.float32), "mask": mask, "K_norm": K}

    def close(self):
        if self.model is None:
            return
        self.model = None
        if self.dev.type == "mps":
            self.torch.mps.empty_cache()


def moge(photos: list[Photo], cache: Path, model: "MoGe | None" = None) -> list[dict]:
    """Cached per photo set. Pass a loaded `model` to reuse it across sets (the caller closes it);
    otherwise one is loaded and released here."""
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"moge_{_key([p.path for p in photos], MOGE_REVISION + str([p.f_px for p in photos]))}.npz"
    if f.exists():
        z = np.load(f)
        return [{"points": z["points"][i], "mask": z["mask"][i], "K_norm": z["K_norm"][i]} for i in range(len(photos))]
    m = model or MoGe()
    out = [m(p) for p in photos]
    if model is None:
        m.close()
    if len({o["points"].shape for o in out}) == 1:
        np.savez_compressed(f, points=np.stack([o["points"] for o in out]), mask=np.stack([o["mask"] for o in out]),
                            K_norm=np.stack([o["K_norm"] for o in out]))
    return out


def mapanything_many(sets: list[list[Photo]], caches: list[Path], max_side: int = 1024) -> list[dict]:
    """Several view sets in one worker process (one model load)."""
    files, jobs = [], []
    for photos, cache in zip(sets, caches):
        cache.mkdir(parents=True, exist_ok=True)
        f = cache / f"ma_{_key([p.path for p in photos], 'map-anything-apache@00f9c245')}.npz"
        files.append(f)
        if not f.exists():
            jobs.append({"images": [str(p.path) for p in photos], "out": str(f)})
    if jobs:
        job = caches[0] / "ma_batch.json"
        job.write_text(json.dumps({"jobs": jobs, "max_side": max_side}))
        r = subprocess.run([str(python_for("mapanything")), "-m", "roomscan.photo.ma_worker", str(job)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"MapAnything worker failed ({r.returncode}): {r.stderr[-2000:]}")
    out = []
    for f in files:
        z = np.load(f) if f.exists() else None
        out.append(None if z is None else {k: z[k] for k in z.files})
    return out


def mapanything(photos: list[Photo], cache: Path, max_side: int = 1024) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"ma_{_key([p.path for p in photos], 'map-anything-apache@00f9c245')}.npz"
    if not f.exists():
        job = cache / (f.stem + ".json")
        job.write_text(json.dumps({"images": [str(p.path) for p in photos], "max_side": max_side}))
        r = subprocess.run([str(python_for("mapanything")), "-m", "roomscan.photo.ma_worker", str(job), str(f)],
                           capture_output=True, text=True)
        if r.returncode != 0 or not f.exists():
            raise RuntimeError(f"MapAnything worker failed ({r.returncode}): {r.stderr[-2000:]}")
    z = np.load(f)
    return {k: z[k] for k in z.files}

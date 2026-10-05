"""ALIKED keypoints + LightGlue matches for sequential and retrieved loop pairs (cache/e20/.venv).

Usage:
  PYTORCH_ENABLE_MPS_FALLBACK=1 TORCH_HOME=cache/e20/torch cache/e20/.venv/bin/python \
      experiments/e20_video_tracking/features.py <capture> [--offsets 1 2 4] [--max-minutes 18]
Writes cache/e20/feat/<capture>/<row>.npz (full-res 960x720 keypoints), cache/e20/match/<capture>/
chunk_*.npz (one key per pair "<row0>_<row1>": (M,3) = idx0, idx1, score*1e4) and pairs.json.
Resumable: finished frames/pairs are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, img_path, load_frames, mem_ok, wait_for_memory  # noqa: E402

NUM_KPTS = 2048


def extract(capture: str, rows: list[int], dev) -> float:
    from lightglue import ALIKED
    from lightglue.utils import load_image
    d = CACHE / "feat" / capture
    d.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (d / f"{r:06d}.npz").exists()]
    if not todo:
        return 0.0
    ex = ALIKED(max_num_keypoints=NUM_KPTS, detection_threshold=0.01).eval().to(dev)
    t0 = time.time()
    with torch.no_grad():
        for k, r in enumerate(todo):
            f = ex.extract(load_image(img_path(capture, r)).to(dev), resize=None)
            np.savez(d / f"{r:06d}.npz", kpts=f["keypoints"][0].cpu().numpy().astype(np.float32),
                     scores=f["keypoint_scores"][0].cpu().numpy().astype(np.float16),
                     desc=f["descriptors"][0].cpu().numpy().astype(np.float16))
            if k % 200 == 0:
                print(f"extract {k}/{len(todo)}", flush=True)
    sec = time.time() - t0
    del ex
    torch.mps.empty_cache()
    return sec


def load_feat(capture: str, r: int) -> dict:
    z = np.load(CACHE / "feat" / capture / f"{r:06d}.npz")
    return {k: z[k] for k in z.files}


def loop_pairs(capture: str, rows: list[int], t: dict[int, float], k_words: int = 256, top: int = 3,
               query_every: int = 2, min_dt: float = 15.0) -> list[tuple[int, int]]:
    """Retrieval by a tf-idf bag of ALIKED words (k-means vocabulary built on this capture, no poses)."""
    rng = np.random.default_rng(0)
    descs = [load_feat(capture, r)["desc"].astype(np.float32) for r in rows]
    samp = np.concatenate([d[rng.choice(len(d), min(len(d), 200), replace=False)] for d in descs])
    C = samp[rng.choice(len(samp), k_words, replace=False)]
    for _ in range(12):  # Lloyd iterations; descriptors are L2-normalised so argmax dot = nearest
        a = np.argmax(samp @ C.T, 1)
        for c in range(k_words):
            m = a == c
            if m.any():
                v = samp[m].mean(0)
                C[c] = v / (np.linalg.norm(v) + 1e-9)
    H = np.zeros((len(rows), k_words), np.float32)
    for i, d in enumerate(descs):
        H[i] = np.bincount(np.argmax(d @ C.T, 1), minlength=k_words)
    idf = np.log(len(rows) / (1 + (H > 0).sum(0)))
    H = H * idf
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-9
    S = H @ H.T
    ts = np.array([t[r] for r in rows])
    pairs = set()
    for i in range(0, len(rows), query_every):
        cand = np.where(np.abs(ts - ts[i]) > min_dt)[0]
        if not len(cand):
            continue
        for j in cand[np.argsort(-S[i, cand])[:top]]:
            pairs.add((rows[min(i, j)], rows[max(i, j)]))
    return sorted(pairs)


def match(capture: str, pairs: list[tuple[int, int]], dev, max_minutes: float) -> tuple[float, int]:
    from lightglue import LightGlue
    d = CACHE / "match" / capture
    d.mkdir(parents=True, exist_ok=True)
    done = set()
    for p in sorted(d.glob("chunk_*.npz")):
        done |= set(np.load(p).files)
    todo = [p for p in pairs if f"{p[0]}_{p[1]}" not in done]
    print(f"match: {len(pairs)} pairs, {len(todo)} to do", flush=True)
    if not todo:
        return 0.0, 0
    m = LightGlue(features="aliked").eval().to(dev)
    cache: dict[int, dict] = {}

    def feat(r):
        if r not in cache:
            f = load_feat(capture, r)
            cache[r] = {"keypoints": torch.from_numpy(f["kpts"])[None].to(dev),
                        "descriptors": torch.from_numpy(f["desc"].astype(np.float32))[None].to(dev),
                        "image_size": torch.tensor([[960.0, 720.0]], device=dev)}
            if len(cache) > 24:
                cache.pop(next(iter(cache)))
        return cache[r]

    n_chunk = len(list(d.glob("chunk_*.npz")))
    buf, t0, n = {}, time.time(), 0
    with torch.no_grad():
        for k, (a, b) in enumerate(todo):
            o = m({"image0": feat(a), "image1": feat(b)})
            m0, sc = o["matches0"][0].cpu().numpy(), o["matching_scores0"][0].cpu().numpy()
            v = np.where(m0 > -1)[0]
            buf[f"{a}_{b}"] = np.stack([v, m0[v], np.round(sc[v] * 1e4)], 1).astype(np.int32)
            n += 1
            if k % 100 == 99:
                torch.mps.empty_cache()  # the MPS allocator otherwise keeps every attention-size buffer
            if len(buf) >= 500 or k == len(todo) - 1:
                np.savez_compressed(d / f"chunk_{n_chunk:04d}.npz", **buf)
                n_chunk += 1
                buf = {}
                el = time.time() - t0
                print(f"match {k + 1}/{len(todo)} {el / n:.3f}s/pair", flush=True)
                ok, msg = mem_ok()
                if not ok:
                    print("[mem] pausing:", msg, flush=True)
                    wait_for_memory(max_wait_s=300)
                if el > 60 * max_minutes:
                    print("time budget reached; rerun to resume", flush=True)
                    break
    sec = time.time() - t0
    del m
    torch.mps.empty_cache()
    return sec, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--loop-stride", type=int, default=2, help="loop retrieval among every Nth keyframe")
    ap.add_argument("--max-minutes", type=float, default=18)
    a = ap.parse_args()
    print(wait_for_memory())
    dev = torch.device("mps")
    frames = load_frames(a.capture)
    rows = [f["row"] for f in frames]
    t = {f["row"]: f["t"] for f in frames}
    sec_ext = extract(a.capture, rows, dev)
    pairs_p = CACHE / "match" / a.capture / "pairs.json"
    if pairs_p.exists():
        pj = json.loads(pairs_p.read_text())
    else:
        t0 = time.time()
        loops = loop_pairs(a.capture, rows[::a.loop_stride], t)
        seq = sorted({(rows[i], rows[i + o]) for i in range(len(rows)) for o in a.offsets if i + o < len(rows)})
        pj = {"offsets": a.offsets, "loop_stride": a.loop_stride, "seq": seq, "loop": [p for p in loops if p not in set(seq)],
              "sec_retrieval": time.time() - t0}
        pairs_p.parent.mkdir(parents=True, exist_ok=True)
        pairs_p.write_text(json.dumps(pj))
    pairs = [tuple(p) for p in pj["seq"]] + [tuple(p) for p in pj["loop"]]
    sec_m, n = match(a.capture, pairs, dev, a.max_minutes)
    tim_p = CACHE / "match" / a.capture / "timing.json"
    tim = json.loads(tim_p.read_text()) if tim_p.exists() else {"sec_extract": 0, "sec_match": 0, "n_matched": 0}
    tim["sec_extract"] += sec_ext
    tim["sec_match"] += sec_m
    tim["n_matched"] += n
    tim["n_frames"] = len(rows)
    tim["sec_retrieval"] = pj["sec_retrieval"]
    tim_p.write_text(json.dumps(tim))
    print(json.dumps(tim))


if __name__ == "__main__":
    main()

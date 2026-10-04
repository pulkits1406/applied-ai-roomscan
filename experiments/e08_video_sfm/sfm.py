"""RGB-only COLMAP reconstruction of cached keyframes, evaluated against ARKit poses (pseudo-GT).

Usage: uv run python experiments/e08_video_sfm/sfm.py <capture> --stride N [--mappers incremental global]
Features + sequential matches are computed once per (capture, stride) into cache/e08/<capture>_s<N>/;
each (mapper, intrinsics) variant maps from its own copy of that database.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time

import numpy as np
import pycolmap

from common import CACHE, OUT, FRAMES, load_frames, run_dir, traj_metrics

VOCAB = CACHE / "vocab_tree_faiss_flickr100K_words32K.bin"
VOCAB_URL = "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words32K.bin"


def cam_from_world(img) -> np.ndarray:
    p = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
    T = np.eye(4)
    T[:3, :4] = p.matrix()
    return T


def mean_K(frames) -> np.ndarray:
    return np.median(np.array([f["K"] for f in frames]), axis=0)


def build_database(capture: str, stride: int, max_features: int, overlap: int, peak: float, matching: str, tag: str) -> dict:
    d = run_dir(capture, stride, tag)
    d.mkdir(parents=True, exist_ok=True)
    db = d / "database.db"
    meta_p = d / "db_meta.json"
    if db.exists() and meta_p.exists():
        return json.loads(meta_p.read_text())
    frames = load_frames(capture)[::stride]
    names = [f"{f['row']:06d}.jpg" for f in frames]
    K = mean_K(frames)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    reader.camera_params = f"{K[0,0]},{K[1,1]},{K[0,2]},{K[1,2]}"
    ext = pycolmap.FeatureExtractionOptions()
    ext.sift.max_num_features = max_features
    ext.sift.peak_threshold = peak
    t0 = time.time()
    pycolmap.extract_features(db, FRAMES / capture / "rgb", image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=ext)
    t_ext = time.time() - t0
    if matching == "seqloop" and not VOCAB.exists():
        import urllib.request
        urllib.request.urlretrieve(VOCAB_URL, VOCAB)
    t0 = time.time()
    if matching in ("sequential", "seqloop"):
        pair = pycolmap.SequentialPairingOptions()
        pair.overlap = overlap
        pair.quadratic_overlap = True
        if matching == "seqloop":
            # Vocabulary-tree retrieval every few frames links revisits across featureless gaps.
            pair.loop_detection = True
            pair.loop_detection_period = 3
            pair.loop_detection_num_images = 40
            pair.vocab_tree_path = str(VOCAB)
        pycolmap.match_sequential(db, pairing_options=pair)
    else:
        pycolmap.match_exhaustive(db)
    t_match = time.time() - t0
    dbo = pycolmap.Database.open(db)
    meta = {"capture": capture, "stride": stride, "n_images": len(names), "max_features": max_features, "overlap": overlap,
            "peak_threshold": peak, "matching": matching, "tag": tag,
            "K": K.tolist(), "sec_extract": t_ext, "sec_match": t_match,
            "verified_pairs": dbo.num_verified_image_pairs(), "matched_pairs": dbo.num_matched_image_pairs()}
    dbo.close()
    meta_p.write_text(json.dumps(meta, indent=1))
    return meta


def variant_db(capture: str, stride: int, tag: str, intr: str):
    d = run_dir(capture, stride, tag)
    src = d / "database.db"
    if intr == "fixed":
        return src
    dst = d / f"database_{intr}.db"
    shutil.copyfile(src, dst)
    db = pycolmap.Database.open(dst)
    cam = db.read_all_cameras()[0]
    fx, fy, cx, cy = cam.params
    new = pycolmap.Camera(model="SIMPLE_RADIAL", width=cam.width, height=cam.height,
                          params=[0.5 * (fx + fy), cx, cy, 0.0], camera_id=cam.camera_id)
    new.has_prior_focal_length = True
    db.update_camera(new)
    db.close()
    return dst


def run_mapper(capture: str, stride: int, tag: str, mapper: str, intr: str, max_runtime_s: int):
    d = run_dir(capture, stride, tag)
    out = d / f"{mapper}_{intr}"
    out.mkdir(exist_ok=True)
    db = variant_db(capture, stride, tag, intr)
    img_dir = FRAMES / capture / "rgb"
    refine = intr != "fixed"
    t0 = time.time()
    if mapper == "incremental":
        o = pycolmap.IncrementalPipelineOptions()
        o.ba_refine_focal_length = refine
        o.ba_refine_extra_params = refine
        o.ba_refine_principal_point = False
        o.mapper.abs_pose_refine_focal_length = refine
        o.mapper.abs_pose_refine_extra_params = refine
        o.max_runtime_seconds = max_runtime_s
        o.random_seed = 0
        # Handheld video: small baselines between neighbouring keyframes and weakly textured walls.
        o.mapper.init_min_tri_angle = 8.0
        o.mapper.abs_pose_min_num_inliers = 20
        recs = pycolmap.incremental_mapping(db, img_dir, out, options=o)
    else:
        o = pycolmap.GlobalPipelineOptions()
        ba = o.mapper.bundle_adjustment
        ba.refine_focal_length = refine
        ba.refine_extra_params = refine
        ba.refine_principal_point = False
        o.random_seed = 0
        recs = pycolmap.global_mapping(db, img_dir, out, options=o)
    sec = time.time() - t0
    return recs, sec, out


def evaluate(capture: str, recs: dict, n_total: int) -> dict:
    by_row = {f["row"]: f for f in load_frames(capture)}
    models = sorted(recs.values(), key=lambda r: -r.num_reg_images())
    res = {"n_models": len(models), "registered_per_model": [m.num_reg_images() for m in models],
           "n_total": n_total, "models": []}
    res["registered_largest"] = models[0].num_reg_images() if models else 0
    res["registered_frac_largest"] = res["registered_largest"] / n_total
    res["registered_any"] = int(sum(res["registered_per_model"]))
    for mi, m in enumerate(models):
        if m.num_reg_images() < 10:
            continue
        items = []
        for img in m.images.values():
            if not img.has_pose:
                continue
            row = int(img.name[:6])
            items.append((by_row[row]["t"], row, np.linalg.inv(cam_from_world(img)), np.array(by_row[row]["T_wc"])))
        items.sort()
        t = np.array([x[0] for x in items])
        mt = traj_metrics(t, np.stack([x[2] for x in items]), np.stack([x[3] for x in items]))
        mt.pop("_align")
        cams = list(m.cameras.values())
        mt["camera"] = {"model": str(cams[0].model), "params": [float(p) for p in cams[0].params]}
        mt["mean_reproj_error_px"] = float(m.compute_mean_reprojection_error())
        mt["n_points3D"] = int(m.num_points3D())
        mt["t_span_s"] = [float(t[0] - min(f["t"] for f in by_row.values())), float(t[-1] - min(f["t"] for f in by_row.values()))]
        res["models"].append(mt)
    return res


def save_poses(capture: str, recs: dict, path):
    by_row = {f["row"]: f for f in load_frames(capture)}
    models = sorted(recs.values(), key=lambda r: -r.num_reg_images())
    out = []
    for mi, m in enumerate(models):
        for img in m.images.values():
            if img.has_pose:
                row = int(img.name[:6])
                out.append({"row": row, "t": by_row[row]["t"], "model": mi,
                            "T_wc": np.linalg.inv(cam_from_world(img)).round(6).tolist()})
    out.sort(key=lambda x: (x["model"], x["row"]))
    path.write_text(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--overlap", type=int, default=10)
    ap.add_argument("--mappers", nargs="+", default=["incremental", "global"])
    ap.add_argument("--intrinsics", nargs="+", default=["fixed", "refined"])
    ap.add_argument("--max-runtime", type=int, default=900)
    ap.add_argument("--peak-threshold", type=float, default=0.001)
    ap.add_argument("--matching", choices=["sequential", "seqloop", "exhaustive"], default="sequential")
    ap.add_argument("--tag", default=None, help="config name; default derived from peak/matching")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"p{a.peak_threshold:g}_{a.matching}"
    meta = build_database(a.capture, a.stride, a.max_features, a.overlap, a.peak_threshold, a.matching, tag)
    print("db", json.dumps(meta))
    summ_p = OUT / f"{a.capture}_s{a.stride}_{tag}_sfm_summary.json"
    summ = json.loads(summ_p.read_text()) if summ_p.exists() else {}
    summ["database"] = meta
    best = None
    for mapper in a.mappers:
        for intr in a.intrinsics:
            key = f"{mapper}_{intr}"
            print("==>", a.capture, key, flush=True)
            recs, sec, out_dir = run_mapper(a.capture, a.stride, tag, mapper, intr, a.max_runtime)
            ev = evaluate(a.capture, recs, meta["n_images"])
            ev["sec_mapping"] = sec
            ev["model_dir"] = str(out_dir)
            summ[key] = ev
            print(key, {k: ev[k] for k in ["n_models", "registered_per_model", "sec_mapping"]},
                  {k: round(ev["models"][0][k], 4) for k in ["scale_m_per_unit", "ate_rmse_m"]} if ev["models"] else None, flush=True)
            # Global-mapper models register more frames but their positions can be badly wrong, so the
            # per-run pose file prefers incremental_fixed; report.py writes the canonical pose files.
            score = (key == "incremental_fixed", ev["registered_largest"])
            if best is None or score > best[0]:
                best = (score, key, recs)
            summ_p.write_text(json.dumps(summ, indent=1))
    summ["best_variant"] = best[1]
    save_poses(a.capture, best[2], OUT / f"{a.capture}_s{a.stride}_{tag}_sfm_poses.json")
    summ_p.write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

"""COLMAP incremental mapping from cached ALIKED+LightGlue matches (project venv, pycolmap 4.2.1).

Usage: uv run python experiments/e20_video_tracking/colmap_lg.py <capture> [--no-loops] [--tag lg]
Same mapper settings as e08 incremental_fixed (PINHOLE, true median K, no intrinsics refinement,
init_min_tri_angle 8, abs_pose_min_num_inliers 20) so only the features/matches differ.
Writes cache/e20/colmap/<capture>_<tag>/ and out/<capture>_colmap_<tag>_{poses,eval}.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, FRAMES, OUT, evaluate_method, load_frames, median_K, save_poses  # noqa: E402


def load_matches(capture: str) -> dict[str, np.ndarray]:
    out = {}
    for p in sorted((CACHE / "match" / capture).glob("chunk_*.npz")):
        z = np.load(p)
        out.update({k: z[k] for k in z.files})
    return out


def build_db(capture: str, d: Path, use_loops: bool) -> dict:
    db_p = d / "database.db"
    meta_p = d / "db_meta.json"
    if db_p.exists() and meta_p.exists():
        return json.loads(meta_p.read_text())
    frames = load_frames(capture)
    names = [f"{f['row']:06d}.jpg" for f in frames]
    K = median_K(frames)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    reader.camera_params = f"{K[0,0]},{K[1,1]},{K[0,2]},{K[1,2]}"
    pycolmap.Database.open(db_p).close()  # creates the empty schema
    pycolmap.import_images(db_p, FRAMES / capture / "rgb", camera_mode=pycolmap.CameraMode.SINGLE,
                           image_names=names, options=reader)
    db = pycolmap.Database.open(db_p)
    ids = {im.name: im.image_id for im in db.read_all_images()}
    t0 = time.time()
    for f in frames:
        z = np.load(CACHE / "feat" / capture / f"{f['row']:06d}.npz")
        # LightGlue keypoints use integer pixel centres; COLMAP puts the first pixel centre at 0.5.
        db.write_keypoints(ids[f"{f['row']:06d}.jpg"], z["kpts"].astype(np.float32) + 0.5)
    pj = json.loads((CACHE / "match" / capture / "pairs.json").read_text())
    pairs = [tuple(p) for p in pj["seq"]] + ([tuple(p) for p in pj["loop"]] if use_loops else [])
    M = load_matches(capture)
    lines, n_written = [], 0
    for a, b in pairs:
        key = f"{a}_{b}"
        if key not in M or len(M[key]) < 15:
            continue
        na, nb = f"{a:06d}.jpg", f"{b:06d}.jpg"
        db.write_matches(ids[na], ids[nb], M[key][:, :2].astype(np.uint32))
        lines.append(f"{na} {nb}")
        n_written += 1
    db.close()
    (d / "pairs.txt").write_text("\n".join(lines) + "\n")
    opt = pycolmap.TwoViewGeometryOptions()
    opt.min_num_inliers = 15
    pycolmap.verify_matches(db_p, d / "pairs.txt", options=opt)
    db = pycolmap.Database.open(db_p)
    meta = {"capture": capture, "n_images": len(names), "use_loops": use_loops, "pairs_written": n_written,
            "pairs_seq": len(pj["seq"]), "pairs_loop": len(pj["loop"]) if use_loops else 0,
            "verified_pairs": db.num_verified_image_pairs(), "sec_db_verify": time.time() - t0, "K": K.tolist()}
    db.close()
    meta_p.write_text(json.dumps(meta, indent=1))
    return meta


def cam_from_world(img) -> np.ndarray:
    p = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
    T = np.eye(4)
    T[:3, :4] = p.matrix()
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--no-loops", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max-runtime", type=int, default=1100)
    ap.add_argument("--min-inliers", type=int, default=20, help="abs_pose_min_num_inliers (e08: 20, COLMAP default 30)")
    ap.add_argument("--mapper-tag", default="", help="suffix for mapper variants on the same database")
    a = ap.parse_args()
    db_tag = a.tag or ("lg_seq" if a.no_loops else "lg")
    tag = db_tag + a.mapper_tag
    d = CACHE / "colmap" / f"{a.capture}_{db_tag}"
    d.mkdir(parents=True, exist_ok=True)
    meta = build_db(a.capture, d, not a.no_loops)
    print("db", json.dumps(meta), flush=True)
    o = pycolmap.IncrementalPipelineOptions()
    o.ba_refine_focal_length = False
    o.ba_refine_extra_params = False
    o.ba_refine_principal_point = False
    o.mapper.abs_pose_refine_focal_length = False
    o.mapper.abs_pose_refine_extra_params = False
    o.max_runtime_seconds = a.max_runtime
    o.random_seed = 0
    o.mapper.init_min_tri_angle = 8.0
    o.mapper.abs_pose_min_num_inliers = a.min_inliers
    out_dir = d / f"incremental_fixed{a.mapper_tag}"
    out_dir.mkdir(exist_ok=True)
    t0 = time.time()
    recs = pycolmap.incremental_mapping(d / "database.db", FRAMES / a.capture / "rgb", out_dir, options=o)
    sec = time.time() - t0
    models = sorted(recs.values(), key=lambda r: -r.num_reg_images())
    poses, piece_of = {}, {}
    for mi, m in enumerate(models):
        for img in m.images.values():
            if img.has_pose:
                r = int(img.name[:6])
                if r not in poses:
                    poses[r] = np.linalg.inv(cam_from_world(img))
                    piece_of[r] = mi
    rows = [f["row"] for f in load_frames(a.capture)]
    ev = evaluate_method(a.capture, rows, poses, piece_of, metric=False,
                         extra={"method": f"colmap_{tag}", "sec_mapping": sec, "db": meta, "abs_pose_min_num_inliers": a.min_inliers,
                                "registered_per_model": [m.num_reg_images() for m in models][:20]})
    OUT.mkdir(parents=True, exist_ok=True)
    save_poses(CACHE / "poses" / f"{a.capture}_colmap_{tag}.json", poses, piece_of, method=f"colmap_{tag}")
    (OUT / f"{a.capture}_colmap_{tag}_eval.json").write_text(json.dumps(ev, indent=1))
    lg = ev.get("largest", {})
    print(json.dumps({k: ev.get(k) for k in ["n_pieces", "piece_sizes", "coverage_largest_pct", "coverage_any_pct",
                                              "longest_run_s", "longest_run_any_piece_s", "sec_mapping"]}),
          "ATE", lg.get("ate_rmse_m"), "winscale", lg.get("win_scale_minmax"))


if __name__ == "__main__":
    main()

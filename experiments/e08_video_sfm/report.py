"""Collect e08 results into out/summary.json, canonical pose files and trajectory plots.

Usage: uv run python experiments/e08_video_sfm/report.py <capture>=<cfg_dir>:<variant> [...]
e.g. single_room=single_room_s1_p0.001_seqloop:incremental_fixed
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycolmap

from common import CACHE, OUT, load_frames, umeyama
from sfm import cam_from_world, save_poses
from scale import load_models


def compact(ev):
    keep = ["n_total", "n_models", "registered_per_model", "registered_any", "registered_largest", "sec_mapping"]
    out = {k: ev[k] for k in keep if k in ev}
    out["models"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
                      if k in ["n", "t_span_s", "scale_m_per_unit", "ate_rmse_m", "gt_extent_m", "rpe_1s_rot_deg_median",
                               "rpe_1s_len_abs_err_median_pct", "window_scale_rel_min_max", "window_scale_rel_range_pct",
                               "window_scale_rel_std_pct", "camera", "mean_reproj_error_px"]} for m in ev["models"]]
    return out


def plot_traj(capture, recs, path):
    frames = {f["row"]: f for f in load_frames(capture)}
    gt = np.array([np.array(f["T_wc"])[:3, 3] for f in frames.values()])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(gt[:, 0], gt[:, 2], color="0.7", lw=1, label="ARKit (pseudo-GT)")
    for mi, rec in enumerate(recs[:10]):
        ims = sorted([im for im in rec.images.values() if im.has_pose], key=lambda im: im.name)
        if len(ims) < 10:
            continue
        c = np.stack([np.linalg.inv(cam_from_world(im))[:3, 3] for im in ims])
        g = np.stack([np.array(frames[int(im.name[:6])]["T_wc"])[:3, 3] for im in ims])
        s, R, t = umeyama(c, g)
        a = (s * (R @ c.T)).T + t
        ax.plot(a[:, 0], a[:, 2], ".", ms=2, label=f"model {mi} ({len(ims)})")
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
    ax.set_title(f"{capture}: SfM fragments, each Sim3-aligned"); ax.legend(fontsize=6, loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def main():
    summary_p = OUT / "summary.json"
    summary = json.loads(summary_p.read_text()) if summary_p.exists() else {}
    for arg in sys.argv[1:]:
        capture, rest = arg.split("=")
        cfg, variant = rest.split(":")
        sfm_files = sorted(OUT.glob(f"{capture}_s*_sfm_summary.json"))
        ent = {"chosen": f"{cfg}/{variant}", "sfm_runs": {}}
        for f in sfm_files:
            d = json.loads(f.read_text())
            name = f.name.replace(f"{capture}_", "").replace("_sfm_summary.json", "")
            ent["sfm_runs"][name] = {"database": {k: d["database"][k] for k in ["n_images", "peak_threshold", "matching", "sec_extract", "sec_match", "verified_pairs"]},
                                     **{k: compact(v) for k, v in d.items() if k not in ("database", "best_variant")}}
        sp = OUT / f"{capture}_scale.json"
        if sp.exists():
            sc = json.loads(sp.read_text())
            ent["scale"] = {"cfg": sc["cfg"], "variant": sc["variant"],
                            "models": [{k: m[k] for k in ["model", "n_registered", "n_frames_used", "s_true_m_per_unit", "moge_over_lidar_per_frame_median", "arkit_tri"]}
                                       | {src: {k: m[src][k] for k in ["pooled_err_pct", "per_frame_err_pct", "single_frame_abs_err_median_pct", "single_frame_abs_err_p90_pct"]}
                                          for src in ["moge", "lidar"]} for m in sc["models"]],
                            "all_models_normalized": sc.get("all_models_normalized"), "moge_timing": sc.get("moge_timing")}
        recs = load_models(CACHE / cfg / variant)
        save_poses(capture, {i: r for i, r in enumerate(recs)}, OUT / f"{capture}_sfm_poses.json")
        plot_traj(capture, recs, OUT / f"{capture}_trajectories.png")
        summary[capture] = ent
    summary_p.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

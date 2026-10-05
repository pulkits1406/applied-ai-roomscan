"""Method 1: joint MapAnything run on [room A photos + doorway photo(s) + room B photos].

Algorithm inputs: images + true intrinsics only (the EXIF case). ARKit/LiDAR are used only below, in
evaluate(). Per doorway pair (A, B) and seeded subset (4 photos of A, 4 of B, drawn with e13's
time-spread sampler, doorway rows excluded) three variants are run: no_door, door1, door2.
Per-view predictions are cached to cache/e17/<A>_<B>_s<si>_<variant>.npz; out/stitch_runs.jsonl is
rebuilt from the caches on every invocation (resumable).

  cache/e13/.venv/bin/python experiments/e17_photo_stitch/run_stitch.py            # infer + evaluate
  cache/e13/.venv/bin/python experiments/e17_photo_stitch/run_stitch.py --eval-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "e16_photo_scale_generalization"))
from e16lib import (ROOT, SETS, R_wc_upright, centre, infer_views, load_model, save_npz, upright_K,  # noqa: E402
                    upright_rgb, wait_for_memory)
from run_subsets import draw_subsets  # noqa: E402  (e13)

CACHE = ROOT / "cache/e17"
N_SUB, N_PER_ROOM = 3, 4
DOORS = json.loads((HERE / "out/doorway_frames.json").read_text())
SUCCESS = {"b_centre_err_median_m": 0.5, "b_yaw_err_median_deg": 15.0}


def plan():
    for d in DOORS:
        A, B = d["pair"]
        door = [p["row"] for p in d["picked_into_B (folder A)"]]
        ra = [r for r in SETS["rooms"][str(A)] if r not in door]
        rb = [r for r in SETS["rooms"][str(B)] if r not in door]
        SA = draw_subsets(ra, N_PER_ROOM, N_SUB, seed=17000 + 10 * A + B)
        SB = draw_subsets(rb, N_PER_ROOM, N_SUB, seed=18000 + 10 * A + B)
        for si, (a, b) in enumerate(zip(SA, SB)):
            for variant, dd in [("no_door", []), ("door1", door[:1]), ("door2", door[:2])]:
                yield A, B, si, variant, a, dd, b


def align_rot_then_sim(Rp, cp, Rg, cg, fixed_scale=None):
    """Sim(3) pred->world: rotation = chordal mean of Rg_i Rp_i^T (robust to near-collinear centres),
    then scale (unless fixed) and translation by least squares on the centres."""
    M = sum(g @ p.T for g, p in zip(Rg, Rp))
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    xp, xg = cp - cp.mean(0), cg - cg.mean(0)
    s = fixed_scale or float(np.sum((xp @ R.T) * xg) / np.sum(xp ** 2))
    return s, R, cg.mean(0) - s * R @ cp.mean(0)


def yaw(Rwc):
    f = Rwc[:, 2]
    return np.degrees(np.arctan2(f[0], f[2]))


def evaluate(rows, nA, nD, poses):
    Rp, cp = poses[:, :3, :3].astype(float), poses[:, :3, 3].astype(float)
    Rg, cg = np.array([R_wc_upright(r) for r in rows]), np.array([centre(r) for r in rows])
    iA, iB = np.arange(nA), np.arange(nA + nD, len(rows))
    out = {}
    # fitA / fitAfolder are the protocol evaluation; fitAll (uses B's ARKit poses) is a diagnostic upper bound
    # that separates a wrong joint reconstruction from an ill-conditioned A-only alignment.
    # fitA_s1 keeps MapAnything's own metric pose scale (scale fixed to 1), which stays well-conditioned when
    # A's centres span too little to estimate scale.
    fits = {"fitA": (iA, None), "fitA_s1": (iA, 1.0), "fitAfolder": (np.arange(nA + nD), None),
            "fitAll": (np.arange(len(rows)), None)}
    for name, (idx, fs) in fits.items():
        s, R, t = align_rot_then_sim(Rp[idx], cp[idx], Rg[idx], cg[idx], fs)
        ca = s * cp @ R.T + t
        Ra = np.einsum("ij,njk->nik", R, Rp)
        err = np.linalg.norm(ca - cg, axis=1)
        dyaw = np.abs((np.array([yaw(r) for r in Ra]) - np.array([yaw(r) for r in Rg]) + 180) % 360 - 180)
        rot = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", Ra, Rg) - 1) / 2, -1, 1)))
        out[name] = {"scale_pred_to_m": s, "A_centre_err_median_m": float(np.median(err[iA])),
                     "B_centre_err_m": err[iB].round(3).tolist(), "B_centre_err_median_m": float(np.median(err[iB])),
                     "B_centroid_err_m": float(np.linalg.norm(ca[iB].mean(0) - cg[iB].mean(0))),
                     "B_yaw_err_deg": dyaw[iB].round(1).tolist(), "B_yaw_err_median_deg": float(np.median(dyaw[iB])),
                     "B_rot_err_median_deg": float(np.median(rot[iB]))}
        if nD:
            out[name]["door_centre_err_m"] = err[nA:nA + nD].round(3).tolist()
        out[name]["success"] = bool(out[name]["B_centre_err_median_m"] < SUCCESS["b_centre_err_median_m"]
                                    and out[name]["B_yaw_err_median_deg"] < SUCCESS["b_yaw_err_median_deg"])
    out["gt_AB_centroid_dist_m"] = float(np.linalg.norm(cg[iB].mean(0) - cg[iA].mean(0)))
    out["gt_A_spread_m"] = float(np.linalg.svd(cg[iA] - cg[iA].mean(0), compute_uv=False)[0] / np.sqrt(nA))
    # Alignment-free cross-room metrics over all (a in A, b in B): relative rotation error, and the angle
    # between predicted and true direction a->b expressed in camera a's frame.
    rr, dd = [], []
    for i in iA:
        for j in iB:
            Rrp, Rrg = Rp[i].T @ Rp[j], Rg[i].T @ Rg[j]
            rr.append(np.degrees(np.arccos(np.clip((np.trace(Rrp.T @ Rrg) - 1) / 2, -1, 1))))
            vp, vg = Rp[i].T @ (cp[j] - cp[i]), Rg[i].T @ (cg[j] - cg[i])
            dd.append(np.degrees(np.arccos(np.clip(vp @ vg / (np.linalg.norm(vp) * np.linalg.norm(vg) + 1e-12), -1, 1))))
    out["xroom_rel_rot_err_median_deg"] = float(np.median(rr))
    out["xroom_dir_err_median_deg"] = float(np.median(dd))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--eval-only", action="store_true"); a = ap.parse_args()
    model = None
    recs = []
    for A, B, si, variant, ra, dd, rb in plan():
        rows = ra + dd + rb
        npz = CACHE / f"{A}_{B}_s{si}_{variant}.npz"
        if not npz.exists():
            if a.eval_only:
                continue
            mem = wait_for_memory()
            if model is None:
                model = load_model("mps")
            res = infer_views(model, [upright_rgb(r) for r in rows], [upright_K(r) for r in rows])
            save_npz(npz, rows, res, nA=len(ra), nD=len(dd))
            print(A, B, si, variant, f"{res['sec']:.1f}s", mem, flush=True)
        z = np.load(npz)
        rec = {"pair": [A, B], "subset": si, "variant": variant, "rows": rows, "nA": len(ra), "nD": len(dd),
               "sec": float(z["sec"]), **evaluate(rows, len(ra), len(dd), z["poses"])}
        recs.append(rec)
        f = rec["fitA"]
        g, h = rec["fitAll"], rec["fitA_s1"]
        print(A, B, si, variant, f"fitA B {f['B_centre_err_median_m']:.2f} m {f['B_yaw_err_median_deg']:.0f} deg s={f['scale_pred_to_m']:.2f}",
              f"| s1 B {h['B_centre_err_median_m']:.2f} m {h['B_yaw_err_median_deg']:.0f} deg | all B {g['B_centre_err_median_m']:.2f} m {g['B_yaw_err_median_deg']:.0f} deg",
              f"| xrot {rec['xroom_rel_rot_err_median_deg']:.0f} dir {rec['xroom_dir_err_median_deg']:.0f} | Aspread {rec['gt_A_spread_m']:.2f}",
              f"(A-B {rec['gt_AB_centroid_dist_m']:.2f} m) ok={f['success']}/{h['success']}", flush=True)
    with open(HERE / "out/stitch_runs.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    med = lambda v: round(float(np.median(v)), 2)
    summ = {"success_rule": SUCCESS, "per_pair": {}}
    for d in DOORS:
        A, B = d["pair"]; summ["per_pair"][f"{A}-{B}"] = {}
        for variant in ["no_door", "door1", "door2"]:
            g = [r for r in recs if r["pair"] == [A, B] and r["variant"] == variant]
            if not g:
                continue
            summ["per_pair"][f"{A}-{B}"][variant] = {
                "n": len(g), "gt_AB_dist_m": med([r["gt_AB_centroid_dist_m"] for r in g]),
                **{f"{fit}_{k}": med([r[fit][m] for r in g]) for fit in ["fitA", "fitA_s1", "fitAll"]
                   for k, m in [("B_err_m", "B_centre_err_median_m"), ("B_yaw_deg", "B_yaw_err_median_deg")]},
                "fitA_A_resid_m": med([r["fitA"]["A_centre_err_median_m"] for r in g]),
                "fitA_scale": [round(r["fitA"]["scale_pred_to_m"], 2) for r in g],
                "xroom_rel_rot_deg": med([r["xroom_rel_rot_err_median_deg"] for r in g]),
                "xroom_dir_deg": med([r["xroom_dir_err_median_deg"] for r in g]),
                "n_success_fitA": sum(r["fitA"]["success"] for r in g), "n_success_fitA_s1": sum(r["fitA_s1"]["success"] for r in g),
                "n_success_fitAll": sum(r["fitAll"]["success"] for r in g)}
    summ["total_gpu_sec"] = round(sum(r["sec"] for r in recs), 1)
    (HERE / "out/stitch_summary.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

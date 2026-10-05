"""Check that casting MapAnything weights to bf16 (to cut memory ~2 GB) leaves results unchanged.

Re-runs already-cached fp32-weight (bf16 autocast) subsets with bf16 weights and compares the metrics.
  cache/e13/.venv/bin/python experiments/e16_photo_scale_generalization/check_bf16_weights.py
"""
import json
import subprocess

import numpy as np
import torch

from e16lib import ROOT, evaluate, infer_views, load_model, upright_K, upright_rgb, wait_for_memory

model = load_model("mps", bf16_backbone=True)
fp = subprocess.run(["footprint", str(__import__("os").getpid())], capture_output=True, text=True).stdout.splitlines()[1]
out = {"footprint_after_load": fp, "runs": []}
for name in ["r1_K2_s0", "r3_K2_s0", "r3_K2_s1", "r3_K2_s2", "r3_K2_s3"]:
    wait_for_memory()
    z = np.load(ROOT / f"cache/e16/runs/{name}.npz"); rows = [int(r) for r in z["rows"]]
    ref = evaluate(rows, z["depth_land"], z["K_pred"], z["poses"], float(z["sec"]))
    res = infer_views(model, [upright_rgb(r) for r in rows], [upright_K(r) for r in rows])
    from e16lib import to_land
    dl = np.stack([to_land(d, m) for d, m in zip(res["depth_up"], res["mask_up"])]).astype(np.float16)
    new = evaluate(rows, dl, np.stack(res["K"]), np.stack(res["poses"]), res["sec"])
    keys = ["scale_err_pct", "focal_err_pct", "scale_err_pct_focalcorr", "baseline_ratio_pred_over_gt", "absrel_aligned"]
    r = {k: [round(ref[k], 3), round(new[k], 3)] for k in keys}
    out["runs"].append({"name": name, **r}); print(name, r, flush=True)
fp2 = subprocess.run(["footprint", str(__import__("os").getpid())], capture_output=True, text=True).stdout.splitlines()[1]
out["footprint_after_runs"] = fp2
(ROOT / "experiments/e16_photo_scale_generalization/out/check_bf16_weights.json").write_text(json.dumps(out, indent=1))
print(fp, fp2)

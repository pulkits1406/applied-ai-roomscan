"""Second monocular metric model: are per-room scale biases independent of MoGe-2's? [PSEUDO-GT]

Depth Anything V2 Metric-Indoor (Small, Apache-2.0, trained on Hypersim) via transformers.
Same 359 photos and LiDAR comparison as run_moge.py. Reports per-room bias of each model, their
correlation, and a K=4 room-level ensemble (geometric mean of the two models' scale estimates).
"""
import json
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
from transformers import pipeline
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e12_photo_scale/out"
F = json.loads((OUT / "moge_per_frame.json").read_text())
cap = StrayCapture.load(ROOT / "single_scan_with_ceiling")
pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", device="mps")
rows = []
for r_s, v in F.items():
    r = int(r_s); bgr = cv2.imread(str(ROOT / f"cache/frames/single_scan_with_ceiling/rgb/{r:06d}.jpg"))
    up = Image.fromarray(cv2.cvtColor(cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_BGR2RGB))
    with torch.no_grad(): d = pipe(up)["predicted_depth"].squeeze().float().cpu().numpy()
    d = cv2.resize(d, (up.size[0], up.size[1]))
    dl = cv2.resize(cv2.rotate(d, cv2.ROTATE_90_COUNTERCLOCKWISE), (256, 192), interpolation=cv2.INTER_AREA)
    lid = cap.depth_m(r); m = (cap.confidence(r) == 2) & (lid < 6) & (dl > 0)
    rows.append({"row": r, "room": v["room"], "dav2_err": float(np.median(dl[m] / lid[m])) - 1, "moge_err": v["scale_err"]})
(OUT / "dav2_per_frame.json").write_text(json.dumps(rows))
rng = np.random.default_rng(0); rooms = {}
for x in rows: rooms.setdefault(x["room"], []).append(x)
res = {"per_photo": {k: {"median_signed_pct": round(100 * float(np.median([x[k] for x in rows])), 2), "median_abs_pct": round(100 * float(np.median([abs(x[k]) for x in rows])), 2)} for k in ("moge_err", "dav2_err")},
       "per_photo_error_corr": round(float(np.corrcoef([x["moge_err"] for x in rows], [x["dav2_err"] for x in rows])[0, 1]), 3),
       "per_room_bias_pct": {room: {"moge": round(100 * float(np.median([x["moge_err"] for x in xs])), 1), "dav2": round(100 * float(np.median([x["dav2_err"] for x in xs])), 1)} for room, xs in rooms.items()}}
b = np.array([[v["moge"], v["dav2"]] for v in res["per_room_bias_pct"].values()]); res["per_room_bias_corr"] = round(float(np.corrcoef(b.T)[0, 1]), 3)
for K in (2, 4, 8):
    out = {"moge": [], "dav2": [], "ensemble": []}
    for room, xs in rooms.items():
        for _ in range(200):
            sub = [xs[i] for i in rng.choice(len(xs), min(K, len(xs)), replace=False)]
            m_, d_ = np.median([s["moge_err"] for s in sub]), np.median([s["dav2_err"] for s in sub])
            out["moge"].append(m_); out["dav2"].append(d_); out["ensemble"].append(np.sqrt((1 + m_) * (1 + d_)) - 1)
    res[f"K{K}"] = {k: {"median_abs_pct": round(100 * float(np.median(np.abs(v))), 2), "p90_abs_pct": round(100 * float(np.percentile(np.abs(v), 90)), 2), "within_8pct": round(float((np.abs(np.array(v)) <= .08).mean()), 3)} for k, v in out.items()}
(OUT / "ensemble_dav2.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))

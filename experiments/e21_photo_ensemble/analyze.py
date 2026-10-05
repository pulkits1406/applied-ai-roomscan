"""Photo-tier scale: two structurally different estimators, and their disagreement as interval width.

Data: e16 runs.jsonl (300 runs; rooms 1,3,4,6,7,8; K = 2..8; MoGe-2 and MapAnything on identical
subsets) [PSEUDO-GT; ~3 % floor]. Everything learned (MapAnything constant c, interval parameters)
is fitted leave-one-room-out (LORO): the room being scored never contributes to its own parameters.
Estimators (size ratio pred/true; 1 = exact):
  moge        MoGe-2, FOV given
  ma_fcorr    MapAnything, per-view depth x f_true/f_pred (EXIF focal)
  ma_const    MapAnything raw x c, c = LORO median of 1/raw
  ens         geometric mean of moge and ma_const
Intervals (90 % nominal) on the log ratio:
  fixed       +-w, w = LORO 90th percentile of |log err| of the estimator
  adaptive    +-max(f, k * |log(moge / ma_const)|), (f, k) chosen LORO for 90 % coverage with minimum mean width
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
R = [json.loads(l) for l in (ROOT / "experiments/e16_photo_scale_generalization/out/runs.jsonl").read_text().splitlines()]
R = [r for r in R if r.get("variant") == "images+K"]
rooms = sorted({r["room"] for r in R})

def est(r, c):
    m, f, raw = r["moge_ratio"], r["ratio_focalcorr"], r["ratio_pred_over_lidar"]
    a = raw * c
    return {"moge": m, "ma_fcorr": f, "ma_const": a, "ens": np.sqrt(m * a)}

rows = []
for room in rooms:
    train = [r for r in R if r["room"] != room]
    c = float(np.median([1 / r["ratio_pred_over_lidar"] for r in train]))
    for r in R:
        if r["room"] == room:
            rows.append({"room": room, "K": r["K"], "c": c, **est(r, c)})

def log_err(x): return np.abs(np.log(x))

def fit_adaptive(train):
    d = np.array([abs(np.log(t["moge"] / t["ma_const"])) for t in train]); e = np.array([log_err(t["ens"]) for t in train])
    best = None
    for f in np.linspace(0.01, 0.25, 49):
        for k in np.linspace(0, 1.5, 31):
            w = np.maximum(f, k * d)
            if (e <= w).mean() >= 0.9 and (best is None or w.mean() < best[0]):
                best = (w.mean(), f, k)
    return best[1], best[2]

res = {"label": "PSEUDO-GT; LORO-calibrated; 6 rooms only", "n_runs": len(rows), "estimators": {}, "per_room": {}, "intervals": {}}
for name in ("moge", "ma_fcorr", "ma_const", "ens"):
    e = np.array([log_err(t[name]) for t in rows])
    res["estimators"][name] = {"median_abs_pct": round(100 * float(np.median(np.expm1(e))), 2), "p90_abs_pct": round(100 * float(np.percentile(np.expm1(e), 90)), 2),
                               "within_8pct": round(float((np.expm1(e) <= 0.08).mean()), 3), "catastrophic_gt25pct": int((np.expm1(e) > 0.25).sum())}
for room in rooms:
    rr = [t for t in rows if t["room"] == room]
    res["per_room"][room] = {n: {"bias_pct": round(100 * (float(np.exp(np.median(np.log([t[n] for t in rr])))) - 1), 1),
                                 "within_8pct": round(float(np.mean([abs(t[n] - 1) <= 0.08 for t in rr])), 2)} for n in ("moge", "ma_const", "ens")}
    res["per_room"][room]["c_loro"] = round(rr[0]["c"], 3)
# correlation of per-room biases between the two model families (DAv2 vs MoGe was 0.75)
bm = [res["per_room"][r]["moge"]["bias_pct"] for r in rooms]; ba = [res["per_room"][r]["ma_const"]["bias_pct"] for r in rooms]
res["per_room_bias_corr_moge_vs_ma_const"] = round(float(np.corrcoef(bm, ba)[0, 1]), 3)
# intervals, LORO
cov = {"fixed_ens": [], "adaptive_ens": [], "fixed_moge": []}; wid = {k: [] for k in cov}; per_room_cov = {}
for room in rooms:
    tr = [t for t in rows if t["room"] != room]; te = [t for t in rows if t["room"] == room]
    w_fe = np.percentile([log_err(t["ens"]) for t in tr], 90); w_fm = np.percentile([log_err(t["moge"]) for t in tr], 90)
    f, k = fit_adaptive(tr)
    pc = {}
    for t in te:
        d = abs(np.log(t["moge"] / t["ma_const"])); wa = max(f, k * d)
        for key, w, x in (("fixed_ens", w_fe, t["ens"]), ("adaptive_ens", wa, t["ens"]), ("fixed_moge", w_fm, t["moge"])):
            ok = log_err(x) <= w; cov[key].append(ok); wid[key].append(np.expm1(w)); pc.setdefault(key, []).append(ok)
    per_room_cov[room] = {key: round(float(np.mean(v)), 2) for key, v in pc.items()}
    per_room_cov[room]["adaptive_params"] = [round(f, 3), round(k, 2)]
res["intervals"] = {key: {"coverage": round(float(np.mean(cov[key])), 3), "mean_halfwidth_pct": round(100 * float(np.mean(wid[key])), 1)} for key in cov}
res["intervals"]["per_room_coverage"] = per_room_cov
(HERE / "out").mkdir(exist_ok=True); (HERE / "out/summary.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))

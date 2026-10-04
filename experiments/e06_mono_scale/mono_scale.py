"""Single-image metric scale error of MoGe-2 against LiDAR depth (pseudo ground truth).

Per frame: rotate the RGB upright using the pose's gravity direction, predict metric depth,
rotate back, compare with ARKit LiDAR depth on confidence-2 pixels.
- scale ratio s = median(lidar / pred): the per-image metric scale error the photo tier inherits.
- AbsRel after per-frame median scaling: shape quality independent of scale.
Variants: FOV unknown (model estimates intrinsics) vs FOV given (as from EXIF).
"""
import json, time
from pathlib import Path
import numpy as np, cv2, torch
from moge.model.v2 import MoGeModel
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e06_mono_scale/out"; OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(dev).eval()

def upright_k(T):
    up = T[:3, :3].T @ np.array([0, 1.0, 0])  # world up in camera (OpenCV) coords; image up is -y
    ux, uy = up[0], up[1]
    if -uy >= abs(ux): return None
    if uy >= abs(ux): return cv2.ROTATE_180
    return cv2.ROTATE_90_COUNTERCLOCKWISE if ux > 0 else cv2.ROTATE_90_CLOCKWISE
INV = {None: None, cv2.ROTATE_180: cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE: cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_CLOCKWISE: cv2.ROTATE_90_COUNTERCLOCKWISE}

rows_out = []
for name in ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]:
    cap = StrayCapture.load(ROOT / name)
    rows = set(np.linspace(5, len(cap) - 5, 25).astype(int).tolist())
    for r, bgr in cap.iter_rgb(rows):
        T = cap.T_wc(r); k = upright_k(T)
        img = cv2.cvtColor(cv2.resize(bgr, (960, 720), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        if k is not None: img = cv2.rotate(img, k)
        K = cap.K_rgb(r); fov_x_land = np.degrees(2 * np.arctan(1920 / (2 * K[0, 0])))
        fov_y_land = np.degrees(2 * np.arctan(1440 / (2 * K[1, 1])))
        fov_x = fov_x_land if k in (None, cv2.ROTATE_180) else fov_y_land
        x = torch.tensor(img / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
        lid = cap.depth_m(r); conf = cap.confidence(r)
        rec = {"capture": name, "row": r, "rot": str(k)}
        for variant, kw in [("fov_unknown", {}), ("fov_given", {"fov_x": float(fov_x)})]:
            t0 = time.time()
            with torch.no_grad(): o = model.infer(x, **kw)
            dt = time.time() - t0
            d = o["depth"].float().cpu().numpy(); m = o["mask"].cpu().numpy() > 0.5
            d[~m] = np.nan
            if k is not None: d = cv2.rotate(d, INV[k])
            d = cv2.resize(d, (256, 192), interpolation=cv2.INTER_AREA)
            v = (conf == 2) & np.isfinite(d) & (d > 0) & (lid < 6.0)
            ratio = lid[v] / d[v]; s = float(np.median(ratio))
            absrel = float(np.median(np.abs(s * d[v] - lid[v]) / lid[v]))
            rec[variant] = {"scale": s, "absrel_after_scale": absrel, "sec": dt}
            if "intrinsics" in o and variant == "fov_unknown":
                Kp = o["intrinsics"].cpu().numpy(); W = img.shape[1]
                rec["pred_fov_x_deg"] = float(np.degrees(2 * np.arctan(0.5 / Kp[0, 0]))); rec["true_fov_x_deg"] = float(fov_x)
        rows_out.append(rec)
        print(name, r, rec["rot"], {v: round(rec[v]["scale"], 3) for v in ["fov_unknown", "fov_given"]}, round(rec.get("pred_fov_x_deg", 0), 1), round(fov_x, 1))
(OUT / "per_frame.json").write_text(json.dumps(rows_out, indent=1))
summ = {}
for variant in ["fov_unknown", "fov_given"]:
    s = np.array([r[variant]["scale"] for r in rows_out]); a = np.array([r[variant]["absrel_after_scale"] for r in rows_out])
    le = np.abs(np.log(s))
    summ[variant] = {"n": len(s), "scale_median": round(float(np.median(s)), 4), "scale_p10_p90": np.percentile(s, [10, 90]).round(4).tolist(),
                     "abs_scale_err_median_pct": round(100 * float(np.median(np.abs(s - 1))), 2), "abs_scale_err_p90_pct": round(100 * float(np.percentile(np.abs(s - 1), 90)), 2),
                     "absrel_after_scale_median": round(float(np.median(a)), 4), "sec_per_frame_median": round(float(np.median([r[variant]["sec"] for r in rows_out])), 2)}
    for name in ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]:
        ss = np.array([r[variant]["scale"] for r in rows_out if r["capture"] == name])
        summ[variant][f"scale_median_{name}"] = round(float(np.median(ss)), 4)
print(json.dumps(summ, indent=1)); (OUT / "summary.json").write_text(json.dumps(summ, indent=1))

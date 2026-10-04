"""Door-height scale cue for the photo tier.

1. Grounding DINO (tiny, Apache-2.0) detects "a door." in upright photos.
2. For each box: fit the wall plane on a ring of pixels just outside the box (jambs/wall), intersect
   rays through the box top-centre and bottom-centre with that plane, height = 3-D distance.
   Works for open doors (the opening's interior is not used). Rotation-invariant: no gravity needed.
3. Same measurement with LiDAR depth (pseudo-GT geometry) -> true door height and the
   detector/geometry error, separately from the MoGe scale error.
Scale estimate from one door: s = H_DOOR_PRIOR / h_moge; resulting size error = (H_DOOR_PRIOR / h_moge) * (1 + mono_err) - 1.
"""
import json
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e12_photo_scale/out"; CACHE = ROOT / "cache/e12"
CAPN = "single_scan_with_ceiling"; cap = StrayCapture.load(ROOT / CAPN)
F = json.loads((OUT / "moge_per_frame.json").read_text())
meta = {m["row"]: m for m in json.loads((ROOT / f"cache/frames/{CAPN}/frames.json").read_text())["frames"]}
dev = "mps"; mid = "IDEA-Research/grounding-dino-tiny"
proc = AutoProcessor.from_pretrained(mid); det = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(dev).eval()
rng = np.random.default_rng(0)

def upright_depth_and_K(d_land, K960):
    """Landscape 256x192 depth -> upright 192x256 + matching intrinsics."""
    up = cv2.rotate(d_land, cv2.ROTATE_90_CLOCKWISE); s = 256 / 960
    return up, (K960[1, 1] * s, K960[0, 0] * s, (720 - 1 - K960[1, 2]) * s, K960[0, 2] * s)

def door_height(depth_up, K, box_up, img_wh):
    fx, fy, cx, cy = K; H, W = depth_up.shape
    sx, sy = W / img_wh[0], H / img_wh[1]
    x0, y0, x1, y1 = box_up[0] * sx, box_up[1] * sy, box_up[2] * sx, box_up[3] * sy
    bw = x1 - x0
    vv, uu = np.mgrid[0:H, 0:W]
    ring = (uu > x0 - 0.35 * bw) & (uu < x1 + 0.35 * bw) & (vv > y0) & (vv < y1) & ~((uu > x0 + 0.05 * bw) & (uu < x1 - 0.05 * bw))
    ring &= np.isfinite(depth_up) & (depth_up > 0)
    if ring.sum() < 40: return None
    z = depth_up[ring]; P = np.stack([(uu[ring] - cx) * z / fx, (vv[ring] - cy) * z / fy, z], 1)
    best = None
    for _ in range(200):                                  # RANSAC wall plane
        s = P[rng.choice(len(P), 3, replace=False)]; n = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n /= nn; d = -n @ s[0]; inl = np.abs(P @ n + d) < 0.03
        if best is None or inl.sum() > best[0]: best = (inl.sum(), n, d)
    if best is None or best[0] < 0.4 * len(P): return None
    _, n, d = best; uc = 0.5 * (x0 + x1)
    pts = []
    for v in (y0, y1):
        ray = np.array([(uc - cx) / fx, (v - cy) / fy, 1.0]); den = n @ ray
        if abs(den) < 1e-6: return None
        pts.append(ray * (-d / den))
    if min(p[2] for p in pts) <= 0: return None
    return float(np.linalg.norm(pts[0] - pts[1]))

rows = []
for r_s, v in F.items():
    r = int(r_s); bgr = cv2.imread(str(ROOT / f"cache/frames/{CAPN}/rgb/{r:06d}.jpg"))
    up = Image.fromarray(cv2.cvtColor(cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_BGR2RGB))
    inp = proc(images=up, text="a door.", return_tensors="pt").to(dev)
    with torch.no_grad(): o = det(**inp)
    rr = proc.post_process_grounded_object_detection(o, inp.input_ids, threshold=0.35, text_threshold=0.3, target_sizes=[up.size[::-1]])[0]
    K960 = np.array(meta[r]["K"])
    dm, Km = upright_depth_and_K(np.load(CACHE / f"{r:06d}.npy").astype(np.float32), K960)
    dl = cap.depth_m(r).copy(); dl[cap.confidence(r) < 2] = np.nan
    dlu, Kl = upright_depth_and_K(dl, K960)
    for box, sc in zip(rr["boxes"].cpu().numpy(), rr["scores"].cpu().numpy()):
        x0, y0, x1, y1 = box; W, H = up.size
        if (y1 - y0) < 0.35 * H or (x1 - x0) > 0.8 * W: continue          # door must be tall in frame and not the whole view
        if y0 < 2 or y1 > H - 3: continue                                  # top and bottom must be inside the frame
        hm = door_height(dm, Km, box, up.size); hl = door_height(dlu, Kl, box, up.size)
        rows.append({"row": r, "room": v["room"], "score": float(sc), "box": box.round(1).tolist(), "h_moge": hm, "h_lidar": hl, "mono_err": v["scale_err"]})
(OUT / "door_cue_per_detection.json").write_text(json.dumps(rows, indent=1))
ok = [x for x in rows if x["h_moge"] and x["h_lidar"]]
hl = np.array([x["h_lidar"] for x in ok]); hm = np.array([x["h_moge"] for x in ok])
res = {"photos": len(F), "detections_kept": len(rows), "measurable": len(ok), "photos_with_door": len({x["row"] for x in ok}),
       "lidar_door_height_m": {"median": round(float(np.median(hl)), 3), "p10_p90": np.percentile(hl, [10, 90]).round(3).tolist()},
       "moge_vs_lidar_height_err_pct_median_abs": round(100 * float(np.median(np.abs(hm / hl - 1))), 2)}
for H in [2.03, 2.1, float(np.median(hl))]:
    e = (H / hm) * (1 + np.array([x["mono_err"] for x in ok])) - 1
    res[f"door_estimator_prior_{H:.3f}"] = {"median_signed_pct": round(100 * float(np.median(e)), 2), "median_abs_pct": round(100 * float(np.median(np.abs(e))), 2),
                                           "p90_abs_pct": round(100 * float(np.percentile(np.abs(e), 90)), 2), "within_8pct": round(float((np.abs(e) <= .08).mean()), 3)}
(OUT / "door_cue.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))

"""Survey Stray Scanner captures: formats, units, timing, frame alignment."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, cv2

ROOT = Path(__file__).resolve().parents[2]
CAPS = ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]

def survey(name):
    d = ROOT / name
    out = {"capture": name}
    odo = pd.read_csv(d / "odometry.csv", skipinitialspace=True)
    odo.columns = [c.strip() for c in odo.columns]
    imu = pd.read_csv(d / "imu.csv", skipinitialspace=True)
    imu.columns = [c.strip() for c in imu.columns]
    t = odo["timestamp"].to_numpy()
    dt = np.diff(t)
    out["odo_rows"] = len(odo)
    out["odo_duration_s"] = float(t[-1] - t[0])
    out["odo_dt_median"] = float(np.median(dt))
    out["odo_dt_hist"] = {f"{k:.4f}": int(v) for k, v in zip(*np.unique(np.round(dt, 3), return_counts=True))}
    fr = odo["frame"].to_numpy()
    out["frame_ids_contiguous"] = bool(np.all(np.diff(fr) == 1))
    out["frame_first_last"] = [int(fr[0]), int(fr[-1])]
    q = odo[["qx", "qy", "qz", "qw"]].to_numpy()
    out["quat_norm_range"] = [float(np.linalg.norm(q, axis=1).min()), float(np.linalg.norm(q, axis=1).max())]
    xyz = odo[["x", "y", "z"]].to_numpy()
    out["pos_min"] = xyz.min(0).round(3).tolist(); out["pos_max"] = xyz.max(0).round(3).tolist()
    out["path_length_m"] = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
    out["fx_range"] = [float(odo.fx.min()), float(odo.fx.max())]
    out["cx_range"] = [float(odo.cx.min()), float(odo.cx.max())]
    out["distortion_cols_all_nan"] = bool(odo["distortion_center_x"].isna().all())
    # IMU
    ti = imu["timestamp"].to_numpy()
    out["imu_rows"] = len(imu); out["imu_dt_median"] = float(np.median(np.diff(ti)))
    acc = imu[["a_x", "a_y", "a_z"]].to_numpy()
    out["imu_acc_norm_median"] = float(np.median(np.linalg.norm(acc, axis=1)))
    out["imu_starts_after_odo_s"] = float(ti[0] - t[0]); out["imu_ends_after_odo_s"] = float(ti[-1] - t[-1])
    # depth / confidence
    dfiles = sorted((d / "depth").glob("*.png")); cfiles = sorted((d / "confidence").glob("*.png"))
    out["n_depth"] = len(dfiles); out["n_conf"] = len(cfiles)
    sample = [dfiles[i] for i in np.linspace(0, len(dfiles) - 1, 8).astype(int)]
    ds, cs = [], []
    for f in sample:
        dimg = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        cimg = cv2.imread(str(d / "confidence" / f.name), cv2.IMREAD_UNCHANGED)
        ds.append(dimg); cs.append(cimg)
    out["depth_dtype_shape"] = [str(ds[0].dtype), list(ds[0].shape)]
    allD = np.stack(ds); allC = np.stack(cs)
    out["depth_raw_min_max"] = [int(allD.min()), int(allD.max())]
    out["depth_raw_pcts"] = np.percentile(allD[allD > 0], [1, 50, 99]).round(1).tolist()
    out["depth_zero_frac"] = float((allD == 0).mean())
    out["conf_dtype_shape"] = [str(cs[0].dtype), list(cs[0].shape)]
    out["conf_values"] = {int(k): float(v) for k, v in zip(*np.unique(allC, return_counts=True))}
    out["conf_values"] = {k: round(v / allC.size, 3) for k, v in out["conf_values"].items()}
    # depth by confidence level
    out["depth_median_by_conf_mm"] = {int(c): float(np.median(allD[allC == c])) for c in np.unique(allC)}
    # video
    cap = cv2.VideoCapture(str(d / "rgb.mp4"))
    out["video_fps"] = cap.get(cv2.CAP_PROP_FPS)
    out["video_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out["video_wh"] = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    ok, fr0 = cap.read(); out["decoded_first_frame_shape"] = list(fr0.shape) if ok else None
    cap.release()
    return out

if __name__ == "__main__":
    res = [survey(c) for c in (sys.argv[1:] or CAPS)]
    outdir = ROOT / "experiments/e00_data_survey/out"; outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "survey.json").write_text(json.dumps(res, indent=1))
    for r in res:
        print(json.dumps(r, indent=None))

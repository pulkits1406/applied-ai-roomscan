"""Collect per-method evals into out/summary.json + out/table.md, evaluate the e08 SIFT baseline with the
same metrics, and draw top-down trajectory plots (project venv).

Usage: uv run python experiments/e20_video_tracking/report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, CAPTURES, OUT, ROOT, apply_sim3, evaluate_method, load_frames, load_poses  # noqa: E402

E08 = ROOT / "experiments/e08_video_sfm/out"


def e08_baseline(capture: str):
    p = E08 / f"{capture}_sfm_poses.json"
    if not p.exists():
        return None
    items = json.loads(p.read_text())
    poses = {it["row"]: np.array(it["T_wc"]) for it in items}
    piece_of = {it["row"]: it["model"] for it in items}
    rows = [f["row"] for f in load_frames(capture)]
    ev = evaluate_method(capture, rows, poses, piece_of, metric=False, extra={"method": "colmap_sift_e08"})
    (OUT / f"{capture}_colmap_sift_e08_eval.json").write_text(json.dumps(ev, indent=1))
    return ev, poses, piece_of


def row_of(ev: dict) -> str:
    lg = ev.get("largest", {})
    f = lambda x, d=1: "–" if x is None else f"{x:.{d}f}"  # noqa: E731
    lr = lg.get("longrange_dist_err_pct", {})
    lrn = lg.get("longrange_dist_err_pct_native_scale", {})
    dr = lg.get("drift_fit30", {})
    ms = lg.get("metric_scale_err_pct")
    return (f"| {ev['method']} | {ev['n_input']} | {ev['n_pieces']} | {f(ev.get('coverage_largest_pct'), 0)} | "
            f"{f(ev.get('longest_run_s'))} | {f(lg.get('ate_rmse_m'), 2)} | {f(lg.get('win_ate_median_m'), 3)} | "
            f"{f(lg.get('win_scale_spread_pct'))} | {f(dr.get('err_end_m'), 2)} | {f(lr.get('median_abs'))}/{f(lr.get('p90_abs'))} | "
            f"{f(ms)} | {f(lrn.get('median_abs'))} |")


HEADER = ("| method | frames | pieces | % in largest | longest run s | ATE m (Sim3) | 20s-win ATE med m | "
          "win scale spread % | drift end m (fit 30 s) | long-range dist err med/p90 % | metric scale err % | native LR err med % |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|---|")


def plot(capture: str, entries: list[tuple[str, dict, dict, dict | None]]):
    by_row = {f["row"]: f for f in load_frames(capture)}
    n = len(entries)
    fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    g = np.array([np.array(f["T_wc"])[:3, 3] for f in by_row.values()])
    for ax, (name, ev, poses, piece_of) in zip(axs[0], entries):
        ax.plot(g[:, 0], g[:, 2], color="0.75", lw=2, label="ARKit")
        if "largest" in ev:
            al = ev["largest"]["_align"]
            s, R, t = al["s"], np.array(al["R"]), np.array(al["t"])
            if piece_of:  # largest piece = the piece containing the most rows
                cnt = {}
                for r in poses:
                    cnt[piece_of[r]] = cnt.get(piece_of[r], 0) + 1
                pid = max(cnt, key=cnt.get)
                big = sorted(r for r in poses if piece_of[r] == pid)
            else:
                big = sorted(poses)
            c = apply_sim3(s, R, t, np.array([poses[r][:3, 3] for r in big]))
            tt = np.array([by_row[r]["t"] for r in big])
            sc = ax.scatter(c[:, 0], c[:, 2], c=tt - tt[0], s=2, cmap="viridis")
        lg = ev.get("largest", {})
        ax.set_title(f"{name}\n{ev.get('n_pieces')} pieces, {ev.get('coverage_largest_pct', 0):.0f}% in largest, "
                     f"ATE {lg.get('ate_rmse_m', float('nan')):.2f} m", fontsize=8)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
    fig.suptitle(f"{capture}: top-down x-z, largest piece Sim3-aligned to ARKit (colour = time)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"{capture}_trajectories.png", dpi=110)
    plt.close(fig)


def main():
    summary, lines = {}, []
    for cap in CAPTURES:
        entries = []
        b = e08_baseline(cap)
        if b:
            entries.append(("colmap_sift_e08", b[0], b[1], b[2]))
        for p in sorted(OUT.glob(f"{cap}_*_eval.json")):
            name = p.name[len(cap) + 1:-len("_eval.json")]
            if name == "colmap_sift_e08":
                continue
            ev = json.loads(p.read_text())
            pp = CACHE / "poses" / f"{cap}_{name}.json"
            if not pp.exists():
                continue
            poses, piece_of, _ = load_poses(pp)
            entries.append((name, ev, poses, piece_of if len(set(piece_of.values())) > 1 else None))
        if not entries:
            continue
        summary[cap] = {}
        lines.append(f"\n**{cap}**\n\n" + HEADER)
        for name, ev, _, _ in entries:
            ev = dict(ev)
            ev.setdefault("method", name)
            lg = {k: v for k, v in ev.get("largest", {}).items() if k not in ("_align",)}
            summary[cap][name] = {**{k: v for k, v in ev.items() if k != "largest"}, "largest": lg}
            lines.append(row_of(ev))
        plot(cap, entries)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

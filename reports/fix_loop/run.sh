#!/usr/bin/env bash
# Regenerate the fix-loop before and after runs from scratch (CPU only, ~2 min).
#   reports/fix_loop/run.sh
# 1. render the exact synthetic two-room capture (deterministic)
# 2. score BEFORE (segmentation padding 0.3 m / 0 m, the pre-fix behaviour) and AFTER (1 m / 1 m)
#    with the production evaluator in one site, each capture run through roomscan-lidar
# 3. no-regression check on the supplied whole-apartment capture (if present)
# Outputs: runs/fix_loop/ (ignored) and reports/fix_loop/results/ (tracked copies).
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p runs/fix_loop reports/fix_loop/results
[ -f runs/fix_loop/synthetic_capture/odometry.csv ] || uv run python -c "from roomscan import synthetic; synthetic.write('runs/fix_loop/synthetic_capture')"
uv run roomscan-bench reports/fix_loop/site --run runs/fix_loop --execute --force
cp runs/fix_loop/eval/fix_loop_synthetic.json runs/fix_loop/eval/fix_loop_synthetic.md runs/fix_loop/eval/fix_loop_synthetic_walls.png reports/fix_loop/results/
cp runs/fix_loop/before_fix/plan.png reports/fix_loop/results/before_plan.png
cp runs/fix_loop/after_fix/plan.png reports/fix_loop/results/after_plan.png
if [ -d single_scan_with_ceiling ]; then
  uv run roomscan-lidar single_scan_with_ceiling runs/fix_loop/real_before --seg-pad 0.3 0.0 > /dev/null
  uv run roomscan-lidar single_scan_with_ceiling runs/fix_loop/real_after > /dev/null
  uv run python -m roomscan.bench.compare runs/fix_loop/real_before/plan.json runs/fix_loop/real_after/plan.json \
    > reports/fix_loop/results/real_capture_before_vs_after.json
else
  echo "single_scan_with_ceiling not present: real-data no-regression check skipped" | tee reports/fix_loop/results/real_capture_before_vs_after.json
fi
git diff 5b25c37 8570bb4 -- src/roomscan/geometry.py src/roomscan/lidar/rooms.py > reports/fix_loop/fix.diff || true
echo "done: reports/fix_loop/results/"

#!/usr/bin/env bash
# Reproduction audit from a clean clone: nothing from the development tree except the committed
# repository, the sample zips and (optionally) already-downloaded weights.
#
#   scripts/clean_clone_audit.sh <work_dir> [--weights-from <dir with an HF cache holding MapAnything>]
#
# Steps (each timed into <work_dir>/audit.log): git clone -> uv sync -> pytest -> isolated
# MapAnything env (fresh, under <work_dir>/envs) -> weights check (download if missing) -> copy the
# sample zips -> scripts/prepare_intermediates.sh (unzip + CPU intermediates) -> one command per tier
# with the network disabled (HF_HUB_OFFLINE=1) -> evaluator -> schema validation of every plan.
# Without --weights-from, MapAnything (4.9 GB) is downloaded into the clone.
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$1"; shift
WEIGHTS=""
[ "${1:-}" = "--weights-from" ] && WEIGHTS="$2"
mkdir -p "$WORK"
LOG="$WORK/audit.log"; : > "$LOG"
step() { local name="$1"; shift; local t0=$SECONDS; echo "== $name" | tee -a "$LOG"
         if "$@" >> "$LOG" 2>&1; then echo "   ok ($((SECONDS - t0)) s)" | tee -a "$LOG"; else echo "   FAILED ($((SECONDS - t0)) s)" | tee -a "$LOG"; exit 1; fi; }

R="$WORK/repo"
step "git clone (committed state only)" git clone -q "$SRC" "$R"
cd "$R"
export ROOMSCAN_ENVS="$WORK/envs"
[ -n "$WEIGHTS" ] && export ROOMSCAN_HF_HOME_MAPANYTHING="$WEIGHTS"
step "uv sync" uv sync
step "pytest" uv run pytest -q
step "isolated MapAnything env (fresh)" scripts/setup_envs.sh mapanything
step "weights (fetch missing, then offline check)" bash -c "uv run python scripts/fetch_weights.py ${WEIGHTS:+--hf-home $WEIGHTS} && uv run python scripts/fetch_weights.py --check ${WEIGHTS:+--hf-home $WEIGHTS}"
step "sample zips" bash -c "cp '$SRC'/single_room.zip '$SRC'/single_scan_floor_only.zip '$SRC'/single_scan_with_ceiling.zip ."
step "intermediates from raw (CPU)" scripts/prepare_intermediates.sh
export HF_HUB_OFFLINE=1
step "LiDAR tier, offline" uv run roomscan-lidar single_scan_with_ceiling runs/audit/lidar
step "photo folders (simulated from the sample walkthrough)" uv run python experiments/e23_photo_frontend/make_photo_folders.py single_scan_with_ceiling 8 "" sweep
step "photo tier, offline" uv run roomscan-photo cache/photo_sim/single_scan_with_ceiling_K8_sweep runs/audit/photo
step "video tier, offline" uv run roomscan-video single_scan_with_ceiling/rgb.mp4 runs/audit/video
step "evaluator (synthetic fix-loop site, before/after)" reports/fix_loop/run.sh
step "schema validation of every plan" uv run python -c "
import glob, json, jsonschema
from roomscan.model import Plan
schema = json.load(open('schema/plan.schema.json'))
files = sorted(glob.glob('runs/**/plan.json', recursive=True))
for f in files:
    d = json.load(open(f)); jsonschema.validate(d, schema); Plan.model_validate(d)
print(len(files), 'plans valid against schema/plan.schema.json and the model')
assert files"
echo "audit complete: $LOG" | tee -a "$LOG"

#!/usr/bin/env bash
# Regenerate the gitignored intermediates that experiments read, in dependency order, skipping
# every step whose outputs already exist. See experiments/DEPENDENCIES.md for the full graph.
#
#   scripts/prepare_intermediates.sh [--dry-run] [--gpu]
#
#   --dry-run  print what would run / be skipped; execute nothing
#   --gpu      also run the MPS model steps (MoGe-2 on the e12 photo set, MoGe-2 + ALIKED/LightGlue
#              for e20). Needs the weights (scripts/fetch_weights.py) and, for e20 features, the video
#              env (scripts/setup_envs.sh video). One GPU model at a time; each waits for free memory.
#
# CPU steps: raw zips -> repo-root capture dirs, e02 fuse (*.ply), e07 frame cache, e09 pose graphs,
# e10 segment_v2 (*_v2b_h*_labels.npy), e12 photo_sets (tracked; normally already present).
# The experiment scripts also rewrite small tracked summaries (json/png) as a side effect; those are
# restored to their pre-run content so this script changes only ignored files.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
DRY=0
GPU=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --gpu) GPU=1 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
  esac
done

CAPS=(single_room single_scan_floor_only single_scan_with_ceiling)
# zip -> top-level folder inside it (verified with `unzip -l`) -> repo-root name the code expects.
# A function, not an associative array: macOS /bin/bash is 3.2.
zip_hash() {
  case "$1" in
    single_room) echo c00a170fe1 ;;
    single_scan_floor_only) echo 1a8384c3f6 ;;
    single_scan_with_ceiling) echo c7d28f72c6 ;;
  esac
}

PY="${ROOMSCAN_PY_PROJECT:-$REPO/.venv/bin/python}"
resolve_video_py() {
  local c
  for c in "${ROOMSCAN_PY_VIDEO:-}" "${ROOMSCAN_ENVS:-$REPO/.envs}/video/bin/python" "$REPO/cache/e20/.venv/bin/python"; do
    [[ -n "$c" && -x "$c" ]] && { echo "$c"; return; }
  done
  echo ""
}

# Small JSON reads work before `uv sync` (fresh clone, --dry-run) via the system python3.
jpy() { if [[ -x "$PY" ]]; then "$PY" "$@"; else python3 "$@"; fi; }

say() { printf '%-8s %-26s %s\n' "$1" "$2" "$3"; }

mem_wait() {
  local free
  for _ in $(seq 1 60); do
    free=$(vm_stat 2>/dev/null | awk '/Pages free/ {gsub(/\./,"",$3); print $3}')
    [[ -z "$free" || "$free" -ge 50000 ]] && return 0
    echo "  free pages $free < 50000; waiting 30 s" >&2
    sleep 30
  done
  echo "  memory still short after 30 min; aborting" >&2
  exit 1
}

# run STEP "tracked files to preserve (space-separated, may be empty)" cmd...
run() {
  local step="$1" keep="$2"; shift 2
  if [[ $DRY -eq 1 ]]; then say "[run]" "$step" "$*"; return 0; fi
  say "[run]" "$step" "$*"
  local tmp; tmp="$(mktemp -d "${TMPDIR:-/tmp}/roomscan-keep.XXXXXX")"
  local f kept=()
  for f in $keep; do
    if [[ -f "$f" ]] && git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
      mkdir -p "$tmp/$(dirname "$f")"; cp -p "$f" "$tmp/$f"; kept+=("$f")
    fi
  done
  local t0=$SECONDS
  "$@"
  for f in "${kept[@]+"${kept[@]}"}"; do cp -p "$tmp/$f" "$f"; done
  say "[done]" "$step" "$((SECONDS - t0)) s; restored ${#kept[@]} tracked file(s) (pre-run copies in $tmp)"
}

need_py() {
  [[ -x "$PY" ]] || { echo "project interpreter $PY not found: run 'uv sync' first" >&2; exit 1; }
}

# ---------------------------------------------------------------- 0. raw captures
for cap in "${CAPS[@]}"; do
  h="$(zip_hash "$cap")"
  if [[ -d "$cap" ]]; then say "[skip]" "raw $cap" "$cap/ exists"; continue; fi
  [[ -f "$cap.zip" ]] || { echo "missing $cap/ and $cap.zip: raw data must be supplied" >&2; exit 1; }
  top="$(unzip -Z1 "$cap.zip" | cut -d/ -f1 | sort -u)"
  [[ "$top" == "$h" ]] || { echo "$cap.zip top-level is '$top', expected '$h'" >&2; exit 1; }
  [[ -e "$h" ]] && { echo "$h/ already exists; refusing to extract over it" >&2; exit 1; }
  if [[ $DRY -eq 1 ]]; then say "[run]" "raw $cap" "unzip -q $cap.zip && mv $h $cap"; continue; fi
  say "[run]" "raw $cap" "unzip -q $cap.zip && mv $h $cap"
  unzip -q "$cap.zip" && mv "$h" "$cap"
done

# ---------------------------------------------------------------- 1. e02 fused clouds
missing=()
for cap in "${CAPS[@]}"; do
  [[ -s "experiments/e02_fusion/out/${cap}_vox2cm.ply" ]] || missing+=("$cap")
done
if [[ ${#missing[@]} -eq 0 ]]; then
  say "[skip]" "e02 fuse" "all 3 *_vox2cm.ply present"
else
  [[ $DRY -eq 1 ]] || need_py
  keep="experiments/e02_fusion/out/fusion_summary.json"
  for cap in "${missing[@]}"; do keep+=" experiments/e02_fusion/out/${cap}_topdown.png"; done
  run "e02 fuse" "$keep" "$PY" experiments/e02_fusion/fuse.py "${missing[@]}"
fi

# ---------------------------------------------------------------- 2. e07 frame cache
missing=()
for cap in "${CAPS[@]}"; do
  fj="cache/frames/$cap/frames.json"
  # frames.json is written last, after every JPEG, so its presence marks a finished capture.
  if [[ -s "$fj" ]]; then
    n_json=$(jpy -c "import json,sys;print(len(json.load(open(sys.argv[1]))['frames']))" "$fj" 2>/dev/null || echo "?")
    n_jpg=$( { find "cache/frames/$cap/rgb" -name '*.jpg' 2>/dev/null || true; } | wc -l | tr -d ' ')
    [[ "$n_json" == "$n_jpg" ]] || echo "  WARNING: $fj lists $n_json frames but rgb/ has $n_jpg jpgs" >&2
  else
    missing+=("$cap")
  fi
done
if [[ ${#missing[@]} -eq 0 ]]; then
  say "[skip]" "e07 frame cache" "frames.json present for all 3"
else
  [[ $DRY -eq 1 ]] || need_py
  run "e07 frame cache" "" "$PY" experiments/e07_frame_cache/extract.py "${missing[@]}"
fi

# ---------------------------------------------------------------- 3. e09 pose graphs (CPU, ~2 min)
for v in "" "_v2"; do
  missing=()
  for cap in "${CAPS[@]}"; do [[ -s "cache/e09/${cap}_T_wc_corrected${v}.npy" ]] || missing+=("$cap"); done
  script="posegraph${v}.py"
  if [[ ${#missing[@]} -eq 0 ]]; then
    say "[skip]" "e09 $script" "cache/e09/*_T_wc_corrected${v}.npy present"
  else
    [[ $DRY -eq 1 ]] || need_py
    keep=""
    for cap in "${missing[@]}"; do keep+=" experiments/e09_drift_posegraph/out/${cap}_posegraph${v}.json"; done
    run "e09 $script" "$keep" "$PY" "experiments/e09_drift_posegraph/$script" "${missing[@]}"
  fi
done

# ---------------------------------------------------------------- 4. e10 segment_v2 (with_ceiling)
# Only with_ceiling labels are consumed (e09 per_room_rigid, e12 photo_sets, e17); the tracked
# segmentation_v2b.json describes that capture alone, so the other captures are not regenerated.
SEG=experiments/e10_room_seg/out/single_scan_with_ceiling
if [[ -s "${SEG}_v2b_h1.9_labels.npy" && -s "${SEG}_v2b_h2.05_labels.npy" && -s "${SEG}_v2b_h1.6_labels.npy" ]]; then
  say "[skip]" "e10 segment_v2" "single_scan_with_ceiling_v2b_h{1.9,2.05,1.6}_labels.npy present"
else
  [[ $DRY -eq 1 ]] || { need_py; mkdir -p experiments/e10_room_seg/out; }
  run "e10 segment_v2" "experiments/e10_room_seg/out/segmentation_v2b.json ${SEG}_v2_grid.json ${SEG}_rooms_v2b.png" \
    "$PY" experiments/e10_room_seg/segment_v2.py single_scan_with_ceiling
fi

# ---------------------------------------------------------------- 5. e12 photo sets (tracked)
if [[ -s experiments/e12_photo_scale/out/photo_sets.json ]]; then
  say "[skip]" "e12 photo_sets" "out/photo_sets.json present (tracked)"
else
  [[ $DRY -eq 1 ]] || need_py
  run "e12 photo_sets" "" "$PY" experiments/e12_photo_scale/photo_sets.py
fi

# ---------------------------------------------------------------- GPU steps
if [[ $GPU -eq 0 ]]; then
  say "[off]" "GPU steps" "e12 run_moge, e20 moge_depth, e20 features (pass --gpu)"
  exit 0
fi

# 6. e12 run_moge: recomputes every row (not resumable), so it runs only when rows are missing.
n_want=$(jpy -c "import json;d=json.load(open('experiments/e12_photo_scale/out/photo_sets.json'));print(sum(len(v) for v in d['rooms'].values()))" 2>/dev/null || echo 999999)
n_have=$( { find cache/e12 -maxdepth 1 -name '*.npy' 2>/dev/null || true; } | wc -l | tr -d ' ')
if [[ "$n_have" -ge "$n_want" ]]; then
  say "[skip]" "e12 run_moge" "cache/e12 has $n_have/$n_want rows"
else
  [[ $DRY -eq 1 ]] || { need_py; mem_wait; }
  run "e12 run_moge (~10 min)" "experiments/e12_photo_scale/out/moge_per_frame.json" "$PY" experiments/e12_photo_scale/run_moge.py
fi

# 7. e20 MoGe depth, with_ceiling stride 2 (resumable; rows cached by e08 are reused)
if [[ -s cache/e20/moge/single_scan_with_ceiling/timing.json ]]; then
  say "[skip]" "e20 moge_depth" "cache/e20/moge/single_scan_with_ceiling/timing.json present"
else
  [[ $DRY -eq 1 ]] || { need_py; mem_wait; }
  run "e20 moge_depth (~25 min)" "" "$PY" experiments/e20_video_tracking/moge_depth.py single_scan_with_ceiling --stride 2
fi

# 8. e20 ALIKED + LightGlue features/matches (video env, resumable)
VPY="$(resolve_video_py)"
for cap in single_room single_scan_with_ceiling; do
  if [[ -s "cache/e20/match/$cap/pairs.json" ]]; then
    say "[skip]" "e20 features $cap" "cache/e20/match/$cap/pairs.json present"
    continue
  fi
  if [[ -z "$VPY" ]]; then
    [[ $DRY -eq 1 ]] || { echo "no video env: run scripts/setup_envs.sh video" >&2; exit 1; }
    VPY="<video env: run scripts/setup_envs.sh video>"
  fi
  [[ $DRY -eq 1 ]] || mem_wait
  run "e20 features $cap" "" env PYTORCH_ENABLE_MPS_FALLBACK=1 TORCH_HOME="${ROOMSCAN_TORCH_HOME_VIDEO:-$REPO/cache/e20/torch}" \
    "$VPY" experiments/e20_video_tracking/features.py "$cap"
done

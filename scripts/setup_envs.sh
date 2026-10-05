#!/usr/bin/env bash
# Build the isolated Python envs that cannot share the project venv.
#
#   scripts/setup_envs.sh [mapanything|video|all] [--force]
#
# Each env lands at ${ROOMSCAN_ENVS:-<repo>/.envs}/<name>, interpreter <that>/bin/python.
# Pins: envs/<name>/requirements.txt (full freeze, macOS arm64) + envs/<name>/python-version.
# Idempotent: an env whose stamp matches the current requirements hash and whose import smoke
# test passes is left untouched. --force rebuilds the package set (uv pip sync) regardless.
#
# Interpreter resolution used by production code (first hit wins):
#   1. $ROOMSCAN_PY_MAPANYTHING / $ROOMSCAN_PY_VIDEO
#   2. ${ROOMSCAN_ENVS:-<repo>/.envs}/<name>/bin/python      (built by this script)
#   3. legacy cache/e13/.venv/bin/python / cache/e20/.venv/bin/python
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVS_DIR="${ROOMSCAN_ENVS:-$REPO/.envs}"
TARGET="all"
FORCE=0
for a in "$@"; do
  case "$a" in
    mapanything|video|all) TARGET="$a" ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
  esac
done

command -v uv >/dev/null || { echo "uv not found (https://docs.astral.sh/uv/)" >&2; exit 1; }

# The freezes are platform-specific (no CUDA wheels listed). `uv pip sync` installs exactly the
# listed set, which is only complete on macOS arm64; elsewhere resolve from the same pins.
if [[ "$(uname -s)-$(uname -m)" == "Darwin-arm64" ]]; then EXACT=1; else EXACT=0; fi

smoke_mapanything='
import cv2, torch, mapanything
from mapanything.models import MapAnything  # import only; no weights are loaded
assert cv2.__version__.startswith("4.10."), cv2.__version__
print("mapanything ok: torch", torch.__version__, "cv2", cv2.__version__,
      "mps", torch.backends.mps.is_available())
'
smoke_video='
import lightglue, kornia, torch, cv2
from lightglue import ALIKED, LightGlue  # import only; no checkpoints are fetched
print("video ok: torch", torch.__version__, "kornia", kornia.__version__, "cv2", cv2.__version__,
      "mps", torch.backends.mps.is_available())
'

mem_wait() {
  # Heavy installs unpack multi-GB wheels; wait while the machine is short of free pages.
  local free
  for _ in $(seq 1 20); do
    free=$(vm_stat 2>/dev/null | awk '/Pages free/ {gsub(/\./,"",$3); print $3}')
    [[ -z "$free" || "$free" -ge 50000 ]] && return 0
    echo "  free pages $free < 50000; waiting 30 s" >&2
    sleep 30
  done
}

build() {
  local name="$1" smoke="$2"
  local spec="$REPO/envs/$name"
  local venv="$ENVS_DIR/$name"
  local py="$venv/bin/python"
  local pyver; pyver="$(tr -d '[:space:]' < "$spec/python-version")"
  local want; want="$( (cat "$spec/requirements.txt" "$spec/python-version"; echo "exact=$EXACT") | shasum -a 256 | cut -c1-16)"
  local stamp="$venv/.roomscan-stamp"

  echo "== $name -> $venv"
  if [[ $FORCE -eq 0 && -x "$py" && -f "$stamp" && "$(cat "$stamp")" == "$want" ]]; then
    if "$py" -c "$smoke"; then echo "  up to date (stamp $want); skipped"; return 0; fi
    echo "  stamp matches but smoke test failed; reinstalling"
  fi

  local t0=$SECONDS
  if [[ ! -x "$py" ]] || [[ "$("$py" -c 'import platform;print(platform.python_version())')" != "$pyver" ]]; then
    uv venv --allow-existing --python "$pyver" "$venv"
  fi
  mem_wait
  if [[ $EXACT -eq 1 ]]; then
    VIRTUAL_ENV="$venv" uv pip sync --python "$py" "$spec/requirements.txt"
  else
    echo "  non-macOS-arm64 host: resolving from pins (uv pip install -r), not an exact sync" >&2
    VIRTUAL_ENV="$venv" uv pip install --python "$py" -r "$spec/requirements.txt"
  fi
  # Checked before the project is added: its declared deps are deliberately absent here.
  VIRTUAL_ENV="$venv" uv pip check --python "$py" || echo "  WARNING: uv pip check reported issues" >&2
  # Project code (experiment helpers, production worker modules) without its heavy deps.
  VIRTUAL_ENV="$venv" uv pip install --python "$py" --no-deps -e "$REPO"
  "$py" -c "$smoke"
  echo "$want" > "$stamp"
  echo "  built in $((SECONDS - t0)) s; size $(du -sh "$venv" | cut -f1)"
}

mkdir -p "$ENVS_DIR"
case "$TARGET" in
  mapanything) build mapanything "$smoke_mapanything" ;;
  video) build video "$smoke_video" ;;
  all) build mapanything "$smoke_mapanything"; build video "$smoke_video" ;;
esac

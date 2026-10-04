#!/bin/zsh
# One invocation per room (~15-20 min each on M4 MPS); resumable via out/runs.jsonl.
cd "$(dirname "$0")"
PY=../../cache/e13/.venv/bin/python
for room in 1 3 4 6 7 8; do
  $PY run_subsets.py --rooms $room --ks 1 2 3 4 6 8 2>&1 | grep -v -i -E "warning|torch hub|Using cache"
done

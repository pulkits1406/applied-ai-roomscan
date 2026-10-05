# Reproduction audit (clean clone, 2026-10-06)

Command: `scripts/clean_clone_audit.sh <empty dir> --weights-from <existing MapAnything HF cache>`,
run on the development Mac (Apple M4, 16 GB, macOS) against commit `fa0a864` (code identical to
`da88a97`, which produced `reports/data/`). The clone contains only committed files: no `.venv`,
`.envs`, `cache/`, `runs/` or intermediates from the development tree.

| Step | Result | Time |
|---|---|---|
| git clone (committed state only) | ok | 0 s |
| `uv sync` (warm uv package cache) | ok | 2 s |
| `uv run pytest -q` | ok — 35 passed, 5 skipped (4 need the sample captures, 1 opt-in GPU) | 97 s |
| `scripts/setup_envs.sh mapanything` (fresh env) | ok | 32 s |
| `scripts/fetch_weights.py` then `--check` | ok — downloaded ALIKED, LightGlue, COLMAP vocab tree (~60 MB, experiment-only); 8/8 present offline | 135 s |
| copy sample zips | ok | 1 s |
| `scripts/prepare_intermediates.sh` (unzip + CPU intermediates from raw) | ok | 158 s |
| LiDAR tier, `HF_HUB_OFFLINE=1` | ok | 37 s |
| simulated photo folders | ok | 6 s |
| photo tier, offline | ok | 142 s |
| video tier, offline | ok | 493 s |
| fix-loop before/after via the evaluator | ok | 95 s |
| schema validation of every produced plan (`jsonschema` + model) | ok — 7 plans | 0 s |

**Total ≈ 20 min** after the network-bound steps. **Outputs reproduce the development tree:**
`roomscan.bench.compare` against `runs/report/` gives structure identical and max numeric
difference **0.0** for the video and photo plans, **9.6e-12 m** for the LiDAR plan; fix-loop wall
predictions identical; real-capture no-regression difference 0.36 mm, as in `reports/fix_loop/`.

## What this audit does not prove

- **Not a separate machine.** Same Mac, same OS, same uv package cache. A different OS/GPU (or macOS
  release) may change MPS kernels; provenance in every plan makes such differences detectable.
- **Large weights were not re-downloaded.** MoGe-2 (1.3 GB) came from the machine-wide Hugging Face
  cache and MapAnything (4.9 GB) from `--weights-from`. On a fresh machine `fetch_weights.py` downloads
  both (≈ 6.2 GB; network-bound); its HF download path was exercised only with small files.
- **Warm package cache.** On a cold machine, `uv sync` and the env build add roughly 1–2 GB of
  wheel downloads.
- **README "< 15 min to running on a fresh capture":** the install-to-first-LiDAR-plan steps took
  ≈ 4 min here with warm caches; with cold caches the time is dominated by downloads and is not measured.
- **Real iPhone files** were not available; the tiers ran on the supplied sample captures and on
  synthetic files.
- Side effect: `prepare_intermediates.sh` leaves a backup of the tracked files it restores in the
  system temp directory (`roomscan-keep.*`).

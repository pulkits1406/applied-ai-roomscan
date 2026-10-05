# Reproduction: fresh clone to evaluated plans

This guide takes a fresh clone to evaluated plans. It was measured on an Apple M4 (16 GB RAM, macOS 26, uv 0.12.23) on 2026-10-05.
The isolated env pins are full freezes for **macOS arm64**. On other platforms `setup_envs.sh`
resolves from the same pins instead of syncing exactly.

## 0. Prerequisites

- `uv` (installs Python 3.13 and 3.12.13 itself).
- `git`, `unzip`.
- Network access for steps 1–3. Everything after step 3 runs offline.
- Disk space: ≈ 1.6 GB project venv, 2.3 GB isolated envs, 7 GB weights, 0.9 GB raw zips, and
  0.35 GB for the CPU intermediates. The GPU intermediates add ≈ 1.5 GB (e20 features and MoGe).

## 1. Project env

```
uv sync                      # .venv, Python 3.13, groups dev + ml (torch, MoGe-2, transformers, pycolmap)
uv run pytest -q
```

## 2. Isolated envs (MapAnything, video)

```
scripts/setup_envs.sh all            # or: mapanything | video ; --force rebuilds
```

| env | built at | pins | smoke test | build time* | size |
|---|---|---|---|---|---|
| mapanything | `.envs/mapanything` | `envs/mapanything/requirements.txt` (Py 3.12.13; mapanything from git `3d10cf7a…`; opencv-headless 4.10.0.84; torch 2.14.1) | `from mapanything.models import MapAnything`; cv2 4.10 | 39 s | 1.5 GB |
| video | `.envs/video` | `envs/video/requirements.txt` (Py 3.12.13; lightglue git `eb42fee2…`; kornia 0.8.3; torch 2.14.1; opencv 5.0.0.93) | `import lightglue, kornia, torch`; MPS available | 33 s | 845 MB |

\*Both times were measured with a warm uv cache, so they cover install and link only. On a cold
cache, add the download time for about 1 GB of wheels, mostly torch. uv clones files from its
cache on APFS, so the real extra disk use is lower than `du` reports. Running the script again
when the stamp is current takes 2.8 s: it checks the stamp and runs the smoke tests.

`setup_envs.sh` works like this:
- It installs the pins with `uv pip sync`, so each env matches the freeze exactly.
- It runs `uv pip check`.
- It adds the project with `uv pip install --no-deps -e .`.
- It writes a stamp, a hash of the pins. If the stamp matches and the smoke test passes, the env
  is skipped.

Envs land in `${ROOMSCAN_ENVS:-<repo>/.envs}`. For the 2026-10-05 build, `uv pip freeze` of each
new env matched the legacy env (`cache/e13/.venv`, `cache/e20/.venv`) except for the editable and
git entries.

**Interpreter resolution** (production code; the first hit wins):

| env | 1. env var | 2. built env | 3. legacy |
|---|---|---|---|
| MapAnything | `$ROOMSCAN_PY_MAPANYTHING` | `${ROOMSCAN_ENVS:-.envs}/mapanything/bin/python` | `cache/e13/.venv/bin/python` |
| video | `$ROOMSCAN_PY_VIDEO` | `${ROOMSCAN_ENVS:-.envs}/video/bin/python` | `cache/e20/.venv/bin/python` |

Experiment docstrings still name the legacy paths. Any interpreter in the same row works.

## 3. Weights

```
uv run python scripts/fetch_weights.py                 # fetch every default model that is missing
uv run python scripts/fetch_weights.py --check         # offline presence table, exit 1 if anything is missing
uv run python scripts/fetch_weights.py --check --verify   # + sha256 of every file (~7 GB read, ~3 s on M4)
```

The manifest is `envs/weights.yaml`. For each model it records the source, the pinned HF commit or
the URL plus sha256, the license, the size, the env, the consumer, and where the files land:

| model | source / pin | lands in | used by |
|---|---|---|---|
| MoGe-2 ViT-L normal | HF `Ruicheng/moge-2-vitl-normal@cb0e8bbd` (MIT) | `$HF_HOME` (default `~/.cache/huggingface`) | production photo + video |
| MapAnything | HF `facebook/map-anything-apache@00f9c245` (Apache-2.0, 4.9 GB) | `cache/e13/hf` (`$ROOMSCAN_HF_HOME_MAPANYTHING`) | production photo; e13/e16/e17/e20 |
| DINOv2 hub code | GitHub `facebookresearch/dinov2@7764ea0f` as `facebookresearch_dinov2_main` | `$TORCH_HOME/hub` (default `~/.cache/torch`) | MapAnything encoder construction (code only) |
| ALIKED n16 | GitHub URL + sha256 (BSD-3) | `cache/e20/torch/hub/checkpoints` (`$ROOMSCAN_TORCH_HOME_VIDEO`) | production video; e20 |
| LightGlue (ALIKED) | GitHub release `v0.1_arxiv` + sha256 (Apache-2.0) | same | production video; e20 |
| Grounding DINO tiny | HF `IDEA-Research/grounding-dino-tiny@a2bb814d` | `$HF_HOME` | experiment e12 only |
| Depth Anything V2 Metric Indoor Small | HF `depth-anything/…-Small-hf@8078d68a` (licence not declared on the card) | `$HF_HOME` | experiment e12 only |
| COLMAP vocab tree | GitHub release 3.11.1 + sha256 | `cache/e08/` | experiment e08 only |

**How locations are resolved.** A model counts as present if it is found in any of these
locations, checked in order:
1. `--hf-home` / `--torch-home` (both repeatable).
2. The manifest location.
3. `$HF_HOME` / `$TORCH_HOME`.
4. The user default.

Missing files go to the first `--hf-home` / `--torch-home` given, or else to the manifest location.

The existing 4.6 GB MapAnything cache already satisfies its entry, because the manifest location
is `cache/e13/hf`. If the weights live somewhere else, pass `--hf-home DIR`. The table then prints
the `HF_HOME=` value that consumers need.

**What `--check` does.**
- It sets `HF_HUB_OFFLINE=1` and resolves HF files with `try_to_load_from_cache`.
- It hashes the small URL checkpoints.
- It never imports torch and never loads a model.
- It also flags a `refs/main` that does not match the pin. Downloads by commit sha write
  `refs/main` when it is absent, so that offline loads by repo id still resolve.

`--check` output, 2026-10-05:

```
model                                 | env         | used by                                                   | on disk  | location             | status
moge2                                 | project     | production photo + production video                       | 1.2 GB   | ~/.cache/huggingface | ok
mapanything                           | mapanything | production photo (MoGe x MapAnything scale ensemble, e21) | 4.6 GB   | cache/e13/hf         | ok
dinov2-hub-code                       | mapanything | production photo + experiments (MapAnything encoder ...)  | 3.9 MB   | ~/.cache/torch       | ok (ref not recorded by torch.hub)
aliked-n16                            | video       | production video                                          | 2.6 MB   | cache/e20/torch      | ok (sha256 ok)
lightglue-aliked                      | video       | production video                                          | 45.4 MB  | cache/e20/torch      | ok (sha256 ok)
grounding-dino-tiny                   | project     | experiment only (e12 door_cue.py)                         | 658.3 MB | ~/.cache/huggingface | ok
depth-anything-v2-metric-indoor-small | project     | experiment only (e12 ensemble_dav2.py)                    | 94.6 MB  | ~/.cache/huggingface | ok
colmap-vocab-tree                     | project     | experiment only (e08 sfm.py --matching seqloop, ...)      | 9.0 MB   | cache/e08            | ok (sha256 ok)
8/8 present (offline check)
```

The cached DINOv2 hub code matches a fresh download of commit `7764ea0f` file for file. The HF
pins equalled upstream `main` on 2026-10-05. MapAnything downloads took ≈ 18 min.

## 4. Offline operation

Once step 3 reports everything present, run with:

```
export HF_HUB_OFFLINE=1          # no Hub HEAD requests; from_pretrained resolves refs/main locally
export TRANSFORMERS_OFFLINE=1    # same for transformers pipelines (Grounding DINO, DAv2)
```

For MapAnything, also keep `HF_HOME=cache/e13/hf`. The e13 and e20 scripts set this themselves
with `setdefault`. Keep `TORCH_HOME` pointing at the directory that holds
`hub/facebookresearch_dinov2_main`, which is `~/.cache/torch` by default. uniception tries
`torch.hub.load` online first and falls back to that cache. For the video env, set
`TORCH_HOME=cache/e20/torch`; otherwise lightglue downloads its checkpoints again.

## 5. Raw data and intermediates

Put `single_room.zip`, `single_scan_floor_only.zip` and `single_scan_with_ceiling.zip` in the repo
root, then:

```
scripts/prepare_intermediates.sh --dry-run        # what would run / be skipped
scripts/prepare_intermediates.sh                  # CPU steps
scripts/prepare_intermediates.sh --gpu            # + e12 run_moge (~10 min), e20 moge_depth (~49 min), e20 features (~22 min)
```

The steps run in this order:
1. Unzip each capture and rename its hash folder (`c00a170fe1` → `single_room`, `1a8384c3f6` →
   `single_scan_floor_only`, `c7d28f72c6` → `single_scan_with_ceiling`). The folder name is
   checked with `unzip -Z1`, and the script refuses to extract over an existing directory.
2. e02 fuse.
3. e07 frame cache.
4. e09 pose graphs.
5. e10 `segment_v2.py single_scan_with_ceiling`.
6. e12 photo_sets.

Every step is skipped when its outputs exist. The experiment scripts rewrite small tracked
summaries as a side effect, so the script restores those files after each step. For the full
graph, see `experiments/DEPENDENCIES.md`.

## 6. Production runs and evaluator

```
uv run roomscan-lidar single_scan_with_ceiling runs/v1/single_scan_with_ceiling_lidar
uv run roomscan-bench benchmark/pseudo_gt_stray_apartment --run runs/pseudo_v0      # PSEUDO-GT demo
uv run python experiments/e15_harness_on_samples/make_pseudo_site.py               # regenerates runs/pseudo_v0
```

The photo and video tiers call their isolated interpreters through the resolution order in step
2. See `README.md` for their entry points.

## Known conflicts

- **pycolmap and torch cannot share a process.** Verified in the project venv on 2026-10-05
  (pycolmap 4.2.1, torch 2.14.1). The abort happens with either import order, while each module
  imports fine on its own:

  ```
  $ .venv/bin/python -c "import pycolmap, torch"      # exit 134 (SIGABRT); same for "import torch, pycolmap"
  OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
  OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. ...
  *** SIGABRT (@0x1817c45e8) received by PID 4540 (TID 0x1ee24a1c0) stack trace: ***
  ```

  Keep COLMAP work and MoGe/torch work in separate processes, as e08 `scale.py` → `moge_depth.py`
  already does. Do not set `KMP_DUPLICATE_LIB_OK`.
- **mapanything pins `opencv-python-headless==4.10.0.84`.** The project needs opencv 5, so
  MapAnything has its own env.
- **cv2 and PyAV both bundle libavdevice.** This prints objc duplicate-class warnings, which are
  harmless. MoGe also prints an MPS autocast warning, which is harmless too.
- **Memory.** Run one GPU model at a time. MapAnything peaks at ≈ 6.4 GB while loading, and MoGe
  ViT-L uses 1–2 GB. Pause heavy steps when `vm_stat` shows fewer than 50 000 free pages.
  `setup_envs.sh` and `prepare_intermediates.sh --gpu` wait for free memory automatically.

# e26 — is the pipeline repeatable, and does uncertainty respond to weak geometry?

## 1. The phase-4 "video nondeterminism" was a code change, not nondeterminism [FACT]
Two uncached runs of the same clip gave window-1 areas 10.4 vs 0.07 m². Stage by stage
(`compare_model_runs.py`, `out/model_runs.json`): decoded keyframes identical; **MoGe-2 and
MapAnything outputs bitwise identical in all 19 view sets** (18 windows + the FOV pass) across two
processes with different model-loading patterns. Re-running the room step on run 1's cached
outputs reproduces run 1 exactly with the old segmentation padding (10.38 / 3.658 / 11.917 m²) and
run 2 exactly with the new padding (0.069 / 4.32 / 11.917 m²): the padding fix landed while run
1's process was alive. Cause of the misattribution: plans did not record the code version → every
plan now carries `provenance` (git commit, dirty flag, library versions, parameters).

## 2. Determinism by stage [MEASURED]
| stage | result |
|---|---|
| video decode + keyframe selection (timestamps) | identical |
| MoGe-2 (MPS, fp16 autocast), separate processes | bitwise identical (19/19 sets; opt-in test `ROOMSCAN_GPU_TESTS=1`) |
| MapAnything (MPS, bf16), separate processes | bitwise identical (19/19 sets) |
| LiDAR v1.1, `with_ceiling`, two runs | structure identical, max numeric difference 2.3e-11 m (Open3D multithreaded reductions) |
| room step / stability (CPU, fixed seed) | identical (tests) |
Remaining nondeterminism: floating-point summation order (≤ 1e-10 m). Not tested: other
machines/OS versions/torch versions (MPS kernels may differ across macOS releases) — the
provenance record makes such comparisons explicit. Regression: `tests/test_repeatability.py`
(`roomscan.bench.compare.materially_identical`, tolerance 1e-6 m).

## 3. The real instability is geometric sensitivity
The same model outputs give 10.4 or 0.07 m² depending on a parameter (grid padding) that should not
matter. That sensitivity is now measured for every room (`src/roomscan/quality.py`): the wall step
is re-run with the input rotated ±1° and on two disjoint halves of the points (fixed seed); each
face's RMS offset becomes a named `faces:stability` error-budget term; a face missing from a
perturbed outline counts as 0.5 m; a run that finds no outline at all is reported as
`topology_fragile` (no positional information). Photo/video add `faces:registration` = std of
camera heights above the reconstructed floor. Rooms whose floor-area interval exceeds ±50 % are
flagged `geometry_unreliable`. Intervals remain `calibrated: false`.

| capture (pseudo-GT site, `out/eval_with_quality.json`) | interval coverage before → after | effect |
|---|---|---|
| LiDAR `with_ceiling` (reference) | — | rooms 1–4, 6–7 unchanged (stability < 2 mm); r3 one face +11 mm; r8 `topology_fragile` (half-density run lost the outline) |
| LiDAR `floor_only` | 0/5 → 2/5 | widened where its merged room is unstable |
| photo sweep / diverse | 2/12, 2/7 → unchanged | intervals ±10 cm → ±0.4–1.3 m; errors (e.g. 5.7 m vs 3.0 m) are larger still |
| video | 4/6 (size-matched) | v0 0.07 ± 0.65 m², flagged unstable + unreliable |
**Interpretation.** The mechanism widens sensitive geometry and leaves stable geometry alone; it
does not (and should not) cover rooms that are consistently wrong — e.g. a photo set taken from a
tiny room that reconstructs the larger space it looks into. Those are reported by flags, not
hidden by width. No coverage was manufactured: widths come from measured sensitivity.
Runtime cost: LiDAR `with_ceiling` 36 → 33–40 s (4 extra wall extractions per room).

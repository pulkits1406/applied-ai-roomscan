# Commit-hash mapping (history rewrite, 2026-10-06)

Before publication, the repository history was rewritten once: co-author trailers were removed from
commit messages and the author/committer identity was set to `Pulkit Sharma <pulkitsharma14062002@gmail.com>` (GitHub: `pulkits1406`) on every commit.
Commit order, author and committer **dates**, messages otherwise, and every file tree are unchanged
(the final tree is byte-identical). Only commit hashes changed; this table maps them, oldest first,
so references made before the rewrite remain traceable.

| old | new | subject |
|---|---|---|
| `3cc8ee5` | `57ee184` | Initial Commit |
| `75058b9` | `10ff758` | Add Stray Scanner loader with verified pose/frame conventions and tests |
| `ca93505` | `2e38d00` | Data exploration experiments e00-e06: format, poses, fusion, drift, heights, sync, mono scale |
| `4876b54` | `9d8c024` | Engineering log and compliance matrix from exploration phase |
| `a6cf4f6` | `2ef9072` | Add ml dependency group, ignore run outputs, README |
| `8612871` | `30f7219` | Add output schema, benchmark format and evaluator with gates |
| `1442489` | `7547ebb` | LiDAR pipeline v0 and harness run on sample captures (pseudo-GT) |
| `f4250f0` | `28eaa74` | Frame cache, drift ablations, room segmentation experiments |
| `a7a5f0b` | `5b016c3` | Photo/video tier metric-scale experiments |
| `c533a0c` | `ace0274` | Phase 2 log, protocol hypotheses, physical benchmark procedure |
| `e3b1c6d` | `5f887b2` | Schema: calibration state, shared error terms, quality flags; evaluator calibration residuals |
| `acce776` | `c7828b9` | e18: LiDAR repeatability variants (oriented normals, close-range wall positions) |
| `1f9040f` | `812579e` | LiDAR core v1: staged modules, CLI, render; synthetic wall tests |
| `504df41` | `d6a6389` | e19: drift ablation, plane-anchored stitch negative result |
| `74dc40c` | `fffd064` | e16/e17/e21: MapAnything generalisation, photo stitching, two-family ensemble |
| `286d765` | `48eb4a0` | e20: continuous video tracking candidates and failure anatomy |
| `1b1e53e` | `3b4dd57` | Phase 3 log, protocol hypotheses, compliance and physical procedure updates |
| `8328541` | `0420873` | Add HANDOVER.md: project state snapshot at session compaction |
| `244f7e5` | `63470ba` | Reproducibility: isolated-env recipes, weight manifest/fetch, intermediates DAG |
| `5b25c37` | `e8aa951` | LiDAR v1.1: e22 small-room diagnosis, per-room escalation ladder, shared plan assembly |
| `8570bb4` | `bc9af03` | Photo and video frontends on the shared back end; evaluator --execute; synthetic smoke test |
| `8f98342` | `9eee57e` | e24 three-tier evaluation via --execute; evaluator identity fixes; compliance update |
| `2b53124` | `c0b00f6` | Phase 4 handover: update HANDOVER.md, log close, protocol hypotheses from e23/e25 |
| `46871d4` | `aea0bc9` | e26: determinism characterised, provenance in every plan, geometry-quality uncertainty |
| `49a1a4a` | `bc890f3` | Real-file input contracts: HEIC, MOV rotation, Stray sizes from data, roomscan-inspect |
| `4fbed2b` | `f73eca0` | Development-site evaluation and tuning-leakage guard |
| `48418fe` | `86f19a9` | Development capture protocol, e27 chain evaluation, benchmark purpose policy, log and compliance |
| `4435f2a` | `279abb6` | HANDOVER: phase 5 state, development capture ready, next steps from the dev capture |
| `86b0f01` | `2b79123` | Submission hardening: damage rule mechanism, fix-loop reproduction switch, schema-file validation, path scrub |
| `9d3d6ec` | `b73137b` | Report tooling: machine-generated benchmark report, fix-loop site and runner, clean-clone audit script |
| `da88a97` | `94d2762` | Reports and evaluator record repo-relative paths only (no machine paths in tracked outputs) |
| `fa0a864` | `ecb6e41` | clean_clone_audit.sh: executable bit |
| `8dda8c2` | `4952bde` | Submission package: audited compliance matrix, capture protocol, device matrix, generated reports, fix-loop bundle, reproduction audit, manifest |

"""Which evaluation results may be used for what (tuning-leakage guard).

  tune / calibrate      development (dev), failure-mode (fail), synthetic and pseudo-GT data only;
                        never bm, rep, inc, walk (the benchmark must stay untouched by tuning)
  benchmark_report      bm, rep, inc, walk only (dev data is not independent evidence)
Calibration additionally needs laser/tape GT. Every use must name its sources explicitly; there
is no default set that silently includes a site.
"""
from __future__ import annotations

from roomscan.bench.gt import BENCHMARK_PURPOSES

ALLOWED = {"tune": {"dev", "fail", "synthetic", "pseudo"}, "calibrate": {"dev", "fail"}, "benchmark_report": set(BENCHMARK_PURPOSES)}


class LeakageError(ValueError):
    pass


def check(reports: list[dict], use: str) -> None:
    """reports: evaluator outputs (eval/<site>.json). Raises LeakageError if any may not be used."""
    for r in reports:
        if r.get("purpose") not in ALLOWED[use]:
            raise LeakageError(f"site {r.get('site')} (purpose {r.get('purpose')}) may not be used to {use}")
        if use == "calibrate" and r.get("gt_kind") not in ("laser", "tape"):
            raise LeakageError(f"site {r.get('site')}: calibration needs laser/tape GT, not {r.get('gt_kind')}")


def residual_rows(reports: list[dict], use: str) -> list[dict]:
    check(reports, use)
    return [row for r in reports for row in r.get("calibration_residuals", [])]

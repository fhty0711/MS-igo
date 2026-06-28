#!/usr/bin/env python3
"""Generate a lightweight report for the signalized-intersection benchmark."""

from __future__ import annotations

import csv
import html
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMPARE_DIR = ROOT / "figures" / "signalized_intersection_comparison"
REPORT_DIR = ROOT / "reports" / "signalized_intersection_report"
ASSET_DIR = REPORT_DIR / "assets"

SUMMARY_CSV = COMPARE_DIR / "summary.csv"
FIGURES = {
    "trajectories": COMPARE_DIR / "overview_trajectories.png",
    "metrics": COMPARE_DIR / "overview_metrics.png",
    "outcomes": COMPARE_DIR / "overview_outcomes.png",
    "manifest": COMPARE_DIR / "manifest.json",
}


def _load_summary() -> list[dict[str, str]]:
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _short_scenario(name: str) -> str:
    return name.replace("signalized_intersection_", "").replace("signalized_intersection", "critical")


def _short_cost(name: str) -> str:
    if name == "signalized_intersection":
        return "full"
    return name.replace("signalized_intersection_", "")


def _fmt(row: dict[str, str], key: str, digits: int = 2) -> str:
    value = row.get(key, "")
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _scenario_intent(scenario_name: str) -> str:
    if scenario_name.endswith("easy_pass"):
        return "easy_pass"
    if scenario_name.endswith("must_stop"):
        return "must_stop"
    return "dilemma"


def _enriched_row(row: dict[str, str]) -> dict[str, str]:
    """Return row with paper-level fields, preserving old summary CSV support."""
    enriched = dict(row)
    scenario = enriched.get("scenario", "")
    intent = enriched.get("scenario_intent") or _scenario_intent(scenario)
    red_legal = _as_bool(enriched.get("red_legal", ""))
    no_blocking = _as_bool(enriched.get("no_blocking", ""))
    cleared = _as_bool(enriched.get("cleared_intersection", ""))
    stopped = _as_bool(enriched.get("stopped_before_line", ""))
    try:
        min_clearance = float(enriched.get("min_clearance", "nan"))
    except Exception:
        min_clearance = float("-inf")

    if "task_success" not in enriched or enriched["task_success"] == "":
        if intent == "must_stop":
            task_success = red_legal and no_blocking and stopped and not cleared
        elif intent == "dilemma":
            task_success = red_legal and no_blocking and (cleared or (stopped and not cleared))
        else:
            task_success = red_legal and no_blocking and cleared
        enriched["task_success"] = str(task_success)

    if "safety_success" not in enriched or enriched["safety_success"] == "":
        enriched["safety_success"] = str(red_legal and no_blocking and min_clearance >= 0.0)

    if "paper_claim" not in enriched or enriched["paper_claim"] == "":
        task_success = _as_bool(enriched["task_success"])
        if not red_legal or not no_blocking or min_clearance < 0.0:
            paper_claim = "unsafe_or_blocked"
        elif task_success and cleared:
            paper_claim = "safe_pass"
        elif task_success and stopped and not cleared:
            paper_claim = "safe_stop"
        else:
            paper_claim = "undecided"
        enriched["paper_claim"] = paper_claim

    if "failure_reason" not in enriched or enriched["failure_reason"] == "":
        task_success = _as_bool(enriched["task_success"])
        if not red_legal:
            failure_reason = "red_illegal"
        elif not no_blocking:
            failure_reason = "blocked_intersection"
        elif min_clearance < 0.0:
            failure_reason = "cross_traffic_conflict"
        elif not task_success:
            failure_reason = "task_unresolved"
        else:
            failure_reason = "none"
        enriched["failure_reason"] = failure_reason

    if "scheme_a_success" not in enriched or enriched["scheme_a_success"] == "":
        paper_claim = enriched.get("paper_claim", "")
        enriched["scheme_a_success"] = str(
            _as_bool(enriched["task_success"])
            and _as_bool(enriched["safety_success"])
            and paper_claim in {"safe_pass", "safe_stop"}
        )

    enriched["scenario_intent"] = intent
    return enriched


def _full_profile_supports_claim(rows: list[dict[str, str]]) -> bool:
    full_rows = [
        _enriched_row(row)
        for row in rows
        if row.get("cost_profile", "") == "signalized_intersection"
    ]
    return bool(full_rows) and all(_as_bool(row.get("scheme_a_success", "")) for row in full_rows)


def _full_profile_takeaway(rows: list[dict[str, str]]) -> str:
    full_rows = [
        _enriched_row(row)
        for row in rows
        if row.get("cost_profile", "") == "signalized_intersection"
    ]
    if not full_rows:
        return "Full-profile rows are not present in the supplied summary CSV."
    claims = ", ".join(
        f"{_short_scenario(row.get('scenario', ''))}: {row.get('paper_claim', 'undecided')}"
        for row in full_rows
    )
    all_task = all(_as_bool(row.get("task_success", "")) for row in full_rows)
    all_safety = all(_as_bool(row.get("safety_success", "")) for row in full_rows)
    task_status = "satisfies" if all_task else "does not satisfy"
    safety_status = "satisfies" if all_safety else "does not satisfy"
    return (
        f"Full profile {task_status} the intended stop/pass task behavior across "
        f"the supplied scenarios, and {safety_status} the strict safety-success "
        f"criterion ({claims})."
    )


def _profile_aggregate_rows(rows: list[dict[str, str]]) -> list[list[str]]:
    by_cost: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        enriched = _enriched_row(row)
        by_cost.setdefault(enriched.get("cost_profile", ""), []).append(enriched)

    result = []
    for cost in sorted(by_cost, key=lambda name: (name != "signalized_intersection", name)):
        cost_rows = by_cost[cost]
        n = len(cost_rows)
        task_rate = sum(_as_bool(row.get("task_success", "")) for row in cost_rows) / max(n, 1)
        safety_rate = sum(_as_bool(row.get("safety_success", "")) for row in cost_rows) / max(n, 1)
        scheme_rate = sum(_as_bool(row.get("scheme_a_success", "")) for row in cost_rows) / max(n, 1)
        min_clearance = min(float(row.get("min_clearance", "nan")) for row in cost_rows)
        worst_risk = max(float(row.get("risk_quantile", "nan")) for row in cost_rows)
        failures = sorted(
            {
                row.get("failure_reason", "unknown")
                for row in cost_rows
                if row.get("failure_reason", "none") != "none"
            }
        )
        result.append(
            [
                _short_cost(cost),
                str(n),
                f"{task_rate:.2f}",
                f"{safety_rate:.2f}",
                f"{scheme_rate:.2f}",
                f"{min_clearance:.2f}",
                f"{worst_risk:.2f}",
                ", ".join(failures) if failures else "none",
            ]
        )
    return result


def _result_rows(rows: list[dict[str, str]]) -> list[list[str]]:
    result = []
    for row in rows:
        row = _enriched_row(row)
        result.append(
            [
                _short_scenario(row.get("scenario", "")),
                _short_cost(row.get("cost_profile", "")),
                row.get("scenario_intent", ""),
                row.get("mode", ""),
                _fmt(row, "final_x", 1),
                _fmt(row, "final_v", 1),
                _fmt(row, "min_clearance", 1),
                _fmt(row, "risk_quantile", 1),
                row.get("task_success", ""),
                row.get("safety_success", ""),
                row.get("scheme_a_success", ""),
                row.get("paper_claim", ""),
                row.get("failure_reason", ""),
                row.get("red_legal", ""),
                row.get("no_blocking", ""),
                row.get("cleared_intersection", ""),
                row.get("stopped_before_line", ""),
            ]
        )
    return result


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _build_markdown(rows: list[dict[str, str]], generated: str | None = None) -> str:
    generated = generated or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    headers = [
        "scenario",
        "cost",
        "intent",
        "mode",
        "final x",
        "final v",
        "min clearance",
        "risk q90",
        "task success",
        "safety success",
        "scheme A success",
        "paper claim",
        "failure reason",
        "red legal",
        "no blocking",
        "cleared",
        "stopped",
    ]
    table = _markdown_table(headers, _result_rows(rows))
    aggregate_table = _markdown_table(
        [
            "cost",
            "runs",
            "task rate",
            "safety rate",
            "scheme A rate",
            "min clearance",
            "worst risk q90",
            "failure reason",
        ],
        _profile_aggregate_rows(rows),
    )
    takeaway = _full_profile_takeaway(rows)
    supports_claim = _full_profile_supports_claim(rows)
    claim_sentence = (
        "The result supports the paper-level success claim for Scheme A: MG-IGO "
        "can rank black-box prioritized chance/STL costs with non-smooth temporal "
        "rules and multi-modal uncertainty while resolving the stop/pass dilemma safely."
        if supports_claim
        else "The result does not yet support the paper-level success claim for "
        "Scheme A; it should be read as a benchmark/reporting run with explicit "
        "failure diagnostics until every full-profile scenario has scheme A success."
    )
    return f"""# Signalized Intersection Benchmark Report

Generated: {generated}

## Purpose

This benchmark targets a signalized intersection dilemma where ego must choose
stop/pass under prioritized STL-style rules and probabilistic cross traffic.
The full profile preserves the black-box, non-smooth, multi-modal, probabilistic
structure: temporal min/max rules, exact priority layers, and per-sample
cross-traffic violations aggregated by a chance constraint.

## Profiles

- `signalized_intersection`: full prioritized chance/STL profile.
- `signalized_intersection_no_chance`: removes the priority-2 probabilistic
  cross-traffic chance layer.
- `signalized_intersection_single_mode`: replaces the multi-modal cross-traffic
  distribution with one deterministic yellow-rush sample.
- `signalized_intersection_soft_dilemma`: keeps chance risk but weakens the
  stop/pass dilemma from tunable to soft.

The optimization profile uses 40 deterministic stratified behavior samples in
the chance layer and the evaluation metrics use 80 samples over the same
obey/yellow-rush/red-run behavior family. The chance layer keeps each
cross-traffic rollout as a per-sample `g(x, xi, ctx) <= 0` violation and then
uses the 0.9 quantile for `alpha=0.1`.

## Scenarios

- `signalized_intersection_easy_pass`: long yellow and faster approach.
- `signalized_intersection_must_stop`: short yellow and slower approach.
- `signalized_intersection_critical`: nominal dilemma timing.

## Summary

{table}

## Profile aggregates

{aggregate_table}

## Interpretation / Paper-level takeaway

{takeaway}

This is a Scheme A single-ego stochastic benchmark. Ego is the only optimizing
agent; cross traffic is an exogenous probabilistic behavior model with
obey/yellow-rush/red-run style samples. {claim_sentence} It does not claim
active multi-agent RNE behavior in this experiment.

## Figures

- `assets/overview_trajectories.png`
- `assets/overview_metrics.png`
- `assets/overview_outcomes.png`
- `assets/manifest.json`

## Reproduction

```bash
cd /mnt/d/claude_workspace1/igo
JAX_PLATFORMS=cuda /root/.venvs/tcmgigo-jaxgpu/bin/python compare_signalized_intersection_profiles.py --force
python generate_signalized_intersection_report.py
```
"""


def _build_html(rows: list[dict[str, str]], generated: str | None = None) -> str:
    generated = generated or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    headers = [
        "scenario",
        "cost",
        "intent",
        "mode",
        "final x",
        "final v",
        "min clearance",
        "risk q90",
        "task success",
        "safety success",
        "scheme A success",
        "paper claim",
        "failure reason",
        "red legal",
        "no blocking",
        "cleared",
        "stopped",
    ]
    table = _html_table(headers, _result_rows(rows))
    aggregate_table = _html_table(
        [
            "cost",
            "runs",
            "task rate",
            "safety rate",
            "scheme A rate",
            "min clearance",
            "worst risk q90",
            "failure reason",
        ],
        _profile_aggregate_rows(rows),
    )
    takeaway = _full_profile_takeaway(rows)
    supports_claim = _full_profile_supports_claim(rows)
    claim_sentence = (
        "The result supports the paper-level success claim for Scheme A: MG-IGO "
        "can rank black-box prioritized chance/STL costs with non-smooth temporal "
        "rules and multi-modal uncertainty while resolving the stop/pass dilemma safely."
        if supports_claim
        else "The result does not yet support the paper-level success claim for "
        "Scheme A; it should be read as a benchmark/reporting run with explicit "
        "failure diagnostics until every full-profile scenario has scheme A success."
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Signalized Intersection Benchmark Report</title>
  <style>
    body {{ margin: 0; background: #0a1120; color: #edf3ff; font-family: Arial, sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 34px 28px 54px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin-top: 32px; border-bottom: 1px solid #33445f; padding-bottom: 8px; }}
    p, li {{ color: #d9e3f4; line-height: 1.65; }}
    code {{ color: #9ff3bd; }}
    table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 13px; }}
    th, td {{ border: 1px solid #40506b; padding: 8px 7px; text-align: center; }}
    th {{ background: #1b2b46; color: white; }}
    tr:nth-child(even) td {{ background: #132037; }}
    tr:nth-child(odd) td {{ background: #10192b; }}
    img {{ width: 100%; border: 1px solid #30425f; border-radius: 6px; background: #0a1120; margin: 10px 0 20px; }}
    .muted {{ color: #9fb0c8; font-size: 13px; }}
  </style>
</head>
<body>
<main>
  <h1>Signalized Intersection Benchmark Report</h1>
  <p class="muted">Generated: {html.escape(generated)}</p>

  <h2>Purpose</h2>
  <p>This benchmark targets a signalized intersection dilemma where ego must
  choose stop/pass under prioritized STL-style rules and probabilistic cross
  traffic. The full profile preserves the black-box, non-smooth, multi-modal,
  probabilistic structure: temporal min/max rules, exact priority layers, and
  per-sample cross-traffic violations aggregated by a chance constraint.</p>

  <h2>Profiles</h2>
  <ul>
    <li><code>signalized_intersection</code>: full prioritized chance/STL profile.</li>
    <li><code>signalized_intersection_no_chance</code>: removes probabilistic chance risk.</li>
    <li><code>signalized_intersection_single_mode</code>: replaces multi-modal traffic with one deterministic sample.</li>
    <li><code>signalized_intersection_soft_dilemma</code>: weakens the stop/pass dilemma from tunable to soft.</li>
  </ul>
  <p>The optimization profile uses 40 deterministic stratified behavior samples
  in the chance layer and the evaluation metrics use 80 samples over the same
  obey/yellow-rush/red-run behavior family. The chance layer keeps each
  cross-traffic rollout as a per-sample <code>g(x, xi, ctx) &lt;= 0</code>
  violation and then uses the 0.9 quantile for <code>alpha=0.1</code>.</p>

  <h2>Scenarios</h2>
  <ul>
    <li><code>signalized_intersection_easy_pass</code>: long yellow and faster approach.</li>
    <li><code>signalized_intersection_must_stop</code>: short yellow and slower approach.</li>
    <li><code>signalized_intersection_critical</code>: nominal dilemma timing.</li>
  </ul>

  <h2>Summary</h2>
  {table}

  <h2>Profile aggregates</h2>
  {aggregate_table}

  <h2>Interpretation / Paper-level takeaway</h2>
  <p>{html.escape(takeaway)}</p>
  <p>This is a Scheme A single-ego stochastic benchmark. Ego is the only
  optimizing agent; cross traffic is an exogenous probabilistic behavior model
  with obey/yellow-rush/red-run style samples. {html.escape(claim_sentence)}
  It does not claim active multi-agent RNE behavior in this experiment.</p>

  <h2>Figures</h2>
  <h3>Trajectory Overview</h3>
  <img src="assets/overview_trajectories.png" alt="trajectory overview">
  <h3>Metrics Overview</h3>
  <img src="assets/overview_metrics.png" alt="metrics overview">
  <h3>Outcome Overview</h3>
  <img src="assets/overview_outcomes.png" alt="outcome overview">

  <h2>Reproduction</h2>
  <pre><code>cd /mnt/d/claude_workspace1/igo
JAX_PLATFORMS=cuda /root/.venvs/tcmgigo-jaxgpu/bin/python compare_signalized_intersection_profiles.py --force
python generate_signalized_intersection_report.py</code></pre>
</main>
</body>
</html>
"""


def _clean_report_dir() -> None:
    if REPORT_DIR.exists():
        shutil.rmtree(REPORT_DIR)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)


def _copy_assets() -> None:
    shutil.copy2(SUMMARY_CSV, ASSET_DIR / SUMMARY_CSV.name)
    for path in FIGURES.values():
        shutil.copy2(path, ASSET_DIR / path.name)


def main() -> None:
    required = [SUMMARY_CSV, *FIGURES.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required report inputs:\n" + "\n".join(missing))

    rows = _load_summary()
    _clean_report_dir()
    _copy_assets()
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_path = REPORT_DIR / "signalized_intersection_report.md"
    html_path = REPORT_DIR / "signalized_intersection_report.html"
    md_path.write_text(_build_markdown(rows, generated=generated), encoding="utf-8")
    html_path.write_text(_build_html(rows, generated=generated), encoding="utf-8")
    print(f"[save] {md_path}")
    print(f"[save] {html_path}")


if __name__ == "__main__":
    main()

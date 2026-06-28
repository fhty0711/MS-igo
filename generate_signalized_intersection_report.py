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


def _result_rows(rows: list[dict[str, str]]) -> list[list[str]]:
    result = []
    for row in rows:
        result.append(
            [
                _short_scenario(row.get("scenario", "")),
                _short_cost(row.get("cost_profile", "")),
                row.get("mode", ""),
                _fmt(row, "final_x", 1),
                _fmt(row, "final_v", 1),
                _fmt(row, "min_clearance", 1),
                _fmt(row, "risk_quantile", 1),
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
        "mode",
        "final x",
        "final v",
        "min clearance",
        "risk q90",
        "red legal",
        "no blocking",
        "cleared",
        "stopped",
    ]
    table = _markdown_table(headers, _result_rows(rows))
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

## Scenarios

- `signalized_intersection_easy_pass`: long yellow and faster approach.
- `signalized_intersection_must_stop`: short yellow and slower approach.
- `signalized_intersection_critical`: nominal dilemma timing.

## Summary

{table}

## Figures

- `assets/overview_trajectories.png`
- `assets/overview_metrics.png`
- `assets/overview_outcomes.png`

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
        "mode",
        "final x",
        "final v",
        "min clearance",
        "risk q90",
        "red legal",
        "no blocking",
        "cleared",
        "stopped",
    ]
    table = _html_table(headers, _result_rows(rows))
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

  <h2>Summary</h2>
  {table}

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

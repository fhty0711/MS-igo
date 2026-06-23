#!/usr/bin/env python3
"""Batch compare wrapper and baseline cost profiles on borrow-overtake cases.

The script keeps the planner and solver unchanged. It only switches the cost
profile while running the same registered scenarios, then exports:

- per-run trajectory panels,
- one trajectory overview figure,
- one metrics overview figure,
- a CSV summary for quick discussion with collaborators.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

import jax
from jax import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backends import get_backend
from config import RNG_SEED
from scenarios import get_scenario
from viz_utils import AGENT_COLORS, HIST_CLR, _draw_rect, _draw_road


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "figures", "cost_profile_comparison")
CACHE_DIR = os.path.join(OUT_DIR, "cache")

DEFAULT_SCENARIOS = (
    "borrow_overtake_safe",
    "borrow_overtake_blocked",
    "borrow_overtake_critical",
)
DEFAULT_COSTS = (
    "borrow_overtake",
    "borrow_overtake_baseline",
    "borrow_overtake_matched",
)


@dataclass
class RunRecord:
    """Closed-loop result and derived metrics for one scenario/cost pair."""

    scenario_name: str
    cost_profile: str
    scenario: object
    histories: dict[str, np.ndarray]
    metrics: dict[str, float | str]


def _run_closed_loop(scenario_name: str, cost_profile: str) -> RunRecord:
    """Run one scenario/cost pair and return full state histories."""
    cache_path = _cache_path(scenario_name, cost_profile)
    scenario = get_scenario(scenario_name)
    if os.path.exists(cache_path):
        with np.load(cache_path) as data:
            histories = {
                agent.name: data[agent.name]
                for agent in scenario.agents
            }
        metrics = _compute_metrics(scenario, histories)
        metrics["scenario"] = scenario_name
        metrics["cost_profile"] = cost_profile
        print(f"[cache] scenario={scenario_name} cost={cost_profile}", flush=True)
        return RunRecord(scenario_name, cost_profile, scenario, histories, metrics)

    backend_cls = get_backend(scenario.backend)
    backend = backend_cls(scenario, cost_profile)
    n_steps = scenario.n_mpc_steps or 25
    key = random.PRNGKey(RNG_SEED)

    print(f"[run] scenario={scenario_name} cost={cost_profile} steps={n_steps}", flush=True)
    for step in range(n_steps):
        key, subkey = random.split(key)
        result = backend.step(subkey, step)
        if step % 5 == 0 or step == n_steps - 1:
            print("      " + backend.progress_line(step, result).strip(), flush=True)

    histories = {}
    for agent in scenario.agents:
        pre_step = list(backend.state_history_by_agent[agent.name])
        pre_step.append(backend.current_states[agent.state_index].copy())
        histories[agent.name] = np.asarray(pre_step, dtype=float)

    metrics = _compute_metrics(scenario, histories)
    metrics["scenario"] = scenario_name
    metrics["cost_profile"] = cost_profile
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache_path, **histories)
    return RunRecord(scenario_name, cost_profile, scenario, histories, metrics)


def _cache_path(scenario_name: str, cost_profile: str) -> str:
    safe_cost = cost_profile.replace("/", "_").replace("\\", "_")
    return os.path.join(CACHE_DIR, f"{scenario_name}__{safe_cost}.npz")


def _compute_metrics(scenario, histories: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute borrow-overtake metrics from recorded closed-loop states."""
    ego_name, slow_name, oncoming_name = scenario.agent_names[:3]
    ego = histories[ego_name]
    slow = histories[slow_name]
    oncoming = histories[oncoming_name]
    lane_mid = 0.5 * (min(scenario.road.lane_centers) + max(scenario.road.lane_centers))
    half_width = 0.5 * scenario.vehicle_geometry.width
    pass_clearance = 24.0
    return_tol = 0.6
    centerline_margin = 0.10

    lead = ego[:, 0] - slow[:, 0]
    oncoming_gap = oncoming[:, 0] - ego[:, 0]
    closing_speed = np.maximum(ego[:, 2] + oncoming[:, 2], 1e-6)
    oncoming_ttc = oncoming_gap / closing_speed
    centerline_crossing = ego[:, 1] + half_width - lane_mid
    footprint_in_oncoming = centerline_crossing > centerline_margin
    in_oncoming_fraction = np.clip(
        (ego[:, 1] - lane_mid) / (0.5 * scenario.road.lane_width),
        0.0,
        1.0,
    )
    if np.any(footprint_in_oncoming):
        min_gap_while_borrowing = float(np.min(oncoming_gap[footprint_in_oncoming]))
        min_ttc_while_borrowing = float(np.min(oncoming_ttc[footprint_in_oncoming]))
    else:
        min_gap_while_borrowing = float("inf")
        min_ttc_while_borrowing = float("inf")

    final = ego[-1]
    pass_success = bool(lead[-1] >= pass_clearance and abs(final[1] - scenario.target_y) <= return_tol)
    returned_to_lane = bool(abs(final[1] - scenario.target_y) <= return_tol)
    borrowed_lane = bool(np.any(footprint_in_oncoming))
    conflict_free = bool(
        (not borrowed_lane)
        or (min_gap_while_borrowing >= 28.0)
        or (min_ttc_while_borrowing >= 1.5)
    )
    return {
        "final_x": float(final[0]),
        "final_y": float(final[1]),
        "final_v": float(final[2]),
        "final_lead": float(lead[-1]),
        "final_oncoming_gap": float(oncoming_gap[-1]),
        "max_y": float(np.max(ego[:, 1])),
        "min_y": float(np.min(ego[:, 1])),
        "max_centerline_crossing": float(np.max(centerline_crossing)),
        "max_oncoming_lane_fraction": float(np.max(in_oncoming_fraction)),
        "min_oncoming_gap": float(np.min(oncoming_gap)),
        "min_gap_while_borrowing": min_gap_while_borrowing,
        "min_ttc_while_borrowing": min_ttc_while_borrowing,
        "mean_abs_y_error": float(np.mean(np.abs(ego[:, 1] - scenario.target_y))),
        "borrowed_lane": borrowed_lane,
        "returned_to_lane": returned_to_lane,
        "pass_success": pass_success,
        "conflict_free_while_borrowing": conflict_free,
    }


def _draw_history_panel(ax, record: RunRecord, title: str):
    """Draw one final closed-loop trajectory panel for an overview grid."""
    scenario = record.scenario
    histories = record.histories
    ego_name = scenario.agent_names[0]
    ego = histories[ego_name]
    geometry = scenario.vehicle_geometry

    # Keep the view focused on the ego maneuver. The oncoming vehicle may pass
    # far behind the ego by the final frame, which otherwise flattens the road.
    x_min = float(np.min(ego[:, 0])) - 12.0
    x_max = float(np.max(ego[:, 0])) + 18.0
    _draw_road(ax, x_min, x_max, scenario.road)

    for idx, agent in enumerate(scenario.agents):
        hist = histories[agent.name]
        color, dark = AGENT_COLORS[idx % len(AGENT_COLORS)]
        lw = 2.4 if agent.name == ego_name else 1.8
        alpha = 0.95 if agent.name == ego_name else 0.72
        ax.plot(hist[:, 0], hist[:, 1], color=HIST_CLR if agent.name == ego_name else dark,
                lw=lw, alpha=alpha, zorder=3)
        ax.scatter(hist[::5, 0], hist[::5, 1], s=10, color=color, alpha=0.9, zorder=4)
        _draw_rect(
            ax,
            float(hist[-1, 0]),
            float(hist[-1, 1]),
            float(hist[-1, 3]),
            geometry.length,
            geometry.width,
            color,
            alpha=0.95,
            zorder=6,
            edge_lw=1.0,
        )

    metrics = record.metrics
    subtitle = (
        f"y={metrics['final_y']:.2f}m, v={metrics['final_v']:.1f}m/s, "
        f"lead={metrics['final_lead']:.1f}m"
    )
    ax.set_title(f"{title}\n{subtitle}", color="white", fontsize=9, pad=5)


def _plot_trajectory_overview(records: list[RunRecord], scenarios: tuple[str, ...], costs: tuple[str, ...]):
    """Save a scenario-by-cost trajectory overview image."""
    by_key = {(r.scenario_name, r.cost_profile): r for r in records}
    fig, axes = plt.subplots(
        len(scenarios),
        len(costs),
        figsize=(7.2 * len(costs), 3.1 * len(scenarios)),
        squeeze=False,
    )
    fig.patch.set_facecolor("#0a1120")
    fig.subplots_adjust(left=0.025, right=0.99, top=0.92, bottom=0.04, wspace=0.06, hspace=0.34)
    fig.suptitle("Borrow-Lane Overtaking: Wrapper vs Baseline Cost", color="white", fontsize=15)

    for row, scenario_name in enumerate(scenarios):
        for col, cost in enumerate(costs):
            record = by_key[(scenario_name, cost)]
            label = f"{scenario_name.replace('borrow_overtake_', '')} / {cost.replace('borrow_overtake', 'wrapper')}"
            if cost.endswith("_baseline"):
                label = f"{scenario_name.replace('borrow_overtake_', '')} / baseline"
            _draw_history_panel(axes[row, col], record, label)

    path = os.path.join(OUT_DIR, "overview_trajectories.png")
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[save] {path}")


def _plot_metrics_overview(records: list[RunRecord], scenarios: tuple[str, ...], costs: tuple[str, ...]):
    """Save y/lead/oncoming-gap time-series comparison."""
    by_key = {(r.scenario_name, r.cost_profile): r for r in records}
    metric_specs = (
        ("ego lateral y", lambda r: r.histories[r.scenario.agent_names[0]][:, 1], "m"),
        ("relative lead", lambda r: r.histories[r.scenario.agent_names[0]][:, 0] - r.histories[r.scenario.agent_names[1]][:, 0], "m"),
        ("oncoming gap", lambda r: r.histories[r.scenario.agent_names[2]][:, 0] - r.histories[r.scenario.agent_names[0]][:, 0], "m"),
    )
    palette = ("#40d984", "#ffb454", "#7aa2ff", "#f06595")
    colors = {cost: palette[idx % len(palette)] for idx, cost in enumerate(costs)}

    fig, axes = plt.subplots(
        len(scenarios),
        len(metric_specs),
        figsize=(14.5, 3.0 * len(scenarios)),
        squeeze=False,
    )
    fig.patch.set_facecolor("#0a1120")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.91, bottom=0.06, wspace=0.22, hspace=0.38)
    fig.suptitle("Cost Profile Metrics", color="white", fontsize=15)

    for row, scenario_name in enumerate(scenarios):
        for col, (metric_name, metric_fn, unit) in enumerate(metric_specs):
            ax = axes[row, col]
            ax.set_facecolor("#10192b")
            for cost in costs:
                record = by_key[(scenario_name, cost)]
                values = metric_fn(record)
                label = _short_cost_label(cost)
                ax.plot(np.arange(len(values)), values, lw=2.0, color=colors[cost], label=label)
            if metric_name == "ego lateral y":
                scenario = by_key[(scenario_name, costs[0])].scenario
                for y in scenario.road.lane_centers:
                    ax.axhline(y, color="white", lw=0.9, ls="--", alpha=0.45)
            if metric_name == "relative lead":
                ax.axhline(24.0, color="white", lw=0.9, ls=":", alpha=0.6)
            if metric_name == "oncoming gap":
                ax.axhline(0.0, color="#ff6b6b", lw=0.9, ls=":", alpha=0.7)
            ax.grid(color="white", alpha=0.12, lw=0.6)
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#40506b")
            ax.set_title(f"{scenario_name.replace('borrow_overtake_', '')}: {metric_name}", color="white", fontsize=9)
            ax.set_xlabel("MPC step", color="white", fontsize=8)
            ax.set_ylabel(unit, color="white", fontsize=8)
            if row == 0 and col == len(metric_specs) - 1:
                ax.legend(facecolor="#0a1120", edgecolor="#40506b", labelcolor="white", fontsize=8)

    path = os.path.join(OUT_DIR, "overview_metrics.png")
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[save] {path}")


def _plot_safety_summary(records: list[RunRecord], scenarios: tuple[str, ...], costs: tuple[str, ...]):
    """Save a compact table for objective safety/task validation."""
    by_key = {(r.scenario_name, r.cost_profile): r for r in records}
    rows = []
    for scenario_name in scenarios:
        for cost in costs:
            record = by_key[(scenario_name, cost)]
            metrics = record.metrics
            rows.append([
                scenario_name.replace("borrow_overtake_", ""),
                _short_cost_label(cost),
                "yes" if metrics["borrowed_lane"] else "no",
                "yes" if metrics["pass_success"] else "no",
                "yes" if metrics["returned_to_lane"] else "no",
                "yes" if metrics["conflict_free_while_borrowing"] else "no",
                f"{metrics['min_gap_while_borrowing']:.1f}",
                f"{metrics['min_ttc_while_borrowing']:.2f}",
            ])

    fig, ax = plt.subplots(figsize=(13.5, 3.5))
    fig.patch.set_facecolor("#0a1120")
    ax.set_facecolor("#0a1120")
    ax.axis("off")
    columns = [
        "scenario",
        "cost",
        "borrowed",
        "pass+return",
        "returned",
        "conflict-free\nwhile borrowing",
        "min gap\nwhile borrowing (m)",
        "min TTC\nwhile borrowing (s)",
    ]
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#40506b")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#1b2b46")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#10192b" if row % 2 else "#132037")
            cell.get_text().set_color("white")
            if col in (3, 5):
                text = cell.get_text().get_text()
                if text == "yes":
                    cell.set_facecolor("#174f37")
                elif text == "no":
                    cell.set_facecolor("#63313a")
    ax.set_title(
        "Objective Validation Summary  (return tol=0.6m, gap>=28m or TTC>=1.5s)",
        color="white",
        fontsize=14,
        pad=14,
    )
    path = os.path.join(OUT_DIR, "overview_safety_table.png")
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[save] {path}")


def _write_summary_csv(records: list[RunRecord]):
    """Write one-row-per-run summary metrics."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "summary.csv")
    fieldnames = list(records[0].metrics.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.metrics)
    print(f"[save] {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare wrapper and baseline borrow-overtake costs.")
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--costs", nargs="+", default=list(DEFAULT_COSTS))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached closed-loop histories and rerun all simulations.",
    )
    return parser.parse_args()


def _short_cost_label(cost: str) -> str:
    if cost == "borrow_overtake":
        return "wrapper"
    if cost == "borrow_overtake_baseline":
        return "baseline"
    if cost == "borrow_overtake_matched":
        return "matched"
    return cost.replace("borrow_overtake_", "")


def main():
    args = parse_args()
    scenarios = tuple(args.scenarios)
    costs = tuple(args.costs)
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.force and os.path.isdir(CACHE_DIR):
        for name in os.listdir(CACHE_DIR):
            if name.endswith(".npz"):
                os.remove(os.path.join(CACHE_DIR, name))

    print("Initializing JAX...", flush=True)
    _ = jax.devices()

    records = [
        _run_closed_loop(scenario_name, cost)
        for scenario_name in scenarios
        for cost in costs
    ]
    _write_summary_csv(records)
    _plot_trajectory_overview(records, scenarios, costs)
    _plot_metrics_overview(records, scenarios, costs)
    _plot_safety_summary(records, scenarios, costs)
    print("[done] cost-profile comparison complete")


if __name__ == "__main__":
    main()

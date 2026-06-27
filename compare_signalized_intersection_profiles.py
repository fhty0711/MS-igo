#!/usr/bin/env python3
"""Batch compare signalized-intersection cost profiles.

The runner keeps the same scenario/planner/backend path as the main scenario
script, then exports cached histories, a CSV summary, and compact overview
figures for the signalized-intersection benchmark matrix.
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
from costs import signalized_intersection as si
from scenarios import get_scenario
from viz_utils import render_agents_panel


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "figures", "signalized_intersection_comparison")
CACHE_DIR = os.path.join(OUT_DIR, "cache")

DEFAULT_SCENARIOS = (
    "signalized_intersection_easy_pass",
    "signalized_intersection_must_stop",
    "signalized_intersection_critical",
)
DEFAULT_COSTS = (
    "signalized_intersection",
    "signalized_intersection_no_chance",
    "signalized_intersection_single_mode",
    "signalized_intersection_soft_dilemma",
)


@dataclass
class RunRecord:
    """Closed-loop result and derived metrics for one scenario/cost pair."""

    scenario_name: str
    cost_profile: str
    scenario: object
    histories: dict[str, np.ndarray]
    metrics: dict[str, float | str | bool]


def _cache_path(scenario_name: str, cost_profile: str) -> str:
    safe_cost = cost_profile.replace("/", "_").replace("\\", "_")
    return os.path.join(CACHE_DIR, f"{scenario_name}__{safe_cost}.npz")


def _run_closed_loop(scenario_name: str, cost_profile: str) -> RunRecord:
    """Run one signalized-intersection scenario/cost pair."""
    scenario = get_scenario(scenario_name)
    cache_path = _cache_path(scenario_name, cost_profile)
    if os.path.exists(cache_path):
        with np.load(cache_path) as data:
            histories = {agent.name: data[agent.name] for agent in scenario.agents}
        metrics = _compute_metrics(scenario, histories)
        metrics["scenario"] = scenario_name
        metrics["cost_profile"] = cost_profile
        print(f"[cache] scenario={scenario_name} cost={cost_profile}", flush=True)
        return RunRecord(scenario_name, cost_profile, scenario, histories, metrics)

    backend_cls = get_backend(scenario.backend)
    backend = backend_cls(scenario, cost_profile)
    n_steps = scenario.n_mpc_steps or 30
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


def _compute_metrics(scenario, histories: dict[str, np.ndarray]) -> dict[str, float | str | bool]:
    """Compute signalized-intersection summary metrics from ego history."""
    ego = np.asarray(histories["ego"], dtype=float)
    visual = si.estimate_visual_metrics(ego, n_samples=80)
    final = ego[-1]
    stopped_before = bool(np.any((ego[:, 0] <= si.STOP_LINE_X) & (ego[:, 2] <= 1.0)))
    cleared = bool(np.max(ego[:, 0]) >= si.INTERSECTION_EXIT_X)
    return {
        "mode": str(visual["mode"]),
        "final_x": float(final[0]),
        "final_y": float(final[1]),
        "final_v": float(final[2]),
        "min_clearance": float(visual["min_clearance"]),
        "risk_quantile": float(visual["risk_quantile"]),
        "red_legal": bool(visual["red_legal"]),
        "no_blocking": bool(visual["no_blocking"]),
        "cleared_intersection": cleared,
        "stopped_before_line": stopped_before,
    }


def _draw_history_panel(ax, record: RunRecord, title: str):
    scenario = record.scenario
    histories = record.histories
    ego = histories["ego"]
    states_by_agent = {"ego": ego[-1]}
    render_agents_panel(
        ax,
        scenario,
        states_by_agent=states_by_agent,
        trajectories_by_agent={"ego": [ego]},
        history_by_agent={"ego": ego},
        focus_agent="ego",
        x_win=78.0,
        title=title,
        show_step=2,
    )


def _plot_trajectory_overview(records: list[RunRecord], scenarios: tuple[str, ...], costs: tuple[str, ...]):
    by_key = {(r.scenario_name, r.cost_profile): r for r in records}
    fig, axes = plt.subplots(
        len(scenarios),
        len(costs),
        figsize=(5.8 * len(costs), 3.4 * len(scenarios)),
        squeeze=False,
    )
    fig.patch.set_facecolor("#0a1120")
    fig.subplots_adjust(left=0.02, right=0.99, top=0.93, bottom=0.04, wspace=0.05, hspace=0.32)
    fig.suptitle("Signalized Intersection: Profile Ablations", color="white", fontsize=15)
    for row, scenario_name in enumerate(scenarios):
        for col, cost in enumerate(costs):
            record = by_key[(scenario_name, cost)]
            metrics = record.metrics
            title = (
                f"{_short_scenario_label(scenario_name)} / {_short_cost_label(cost)}\n"
                f"{metrics['mode']}  clear={metrics['min_clearance']:.1f}m  "
                f"risk={metrics['risk_quantile']:.1f}"
            )
            _draw_history_panel(axes[row, col], record, title)
    path = os.path.join(OUT_DIR, "overview_trajectories.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[save] {path}")


def _plot_metrics_overview(records: list[RunRecord], scenarios: tuple[str, ...], costs: tuple[str, ...]):
    by_key = {(r.scenario_name, r.cost_profile): r for r in records}
    metric_names = ("min_clearance", "risk_quantile", "final_x", "final_v")
    palette = ("#40d984", "#ffb454", "#7aa2ff", "#f06595")
    colors = {cost: palette[idx % len(palette)] for idx, cost in enumerate(costs)}

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.0), squeeze=False)
    fig.patch.set_facecolor("#0a1120")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.08, wspace=0.22, hspace=0.30)
    fig.suptitle("Signalized Intersection Metrics", color="white", fontsize=15)
    x = np.arange(len(scenarios))
    width = 0.18
    for ax, metric in zip(axes.ravel(), metric_names):
        ax.set_facecolor("#10192b")
        for idx, cost in enumerate(costs):
            values = [float(by_key[(scenario_name, cost)].metrics[metric]) for scenario_name in scenarios]
            offset = (idx - 1.5) * width
            ax.bar(x + offset, values, width=width, color=colors[cost], label=_short_cost_label(cost))
        ax.set_xticks(x)
        ax.set_xticklabels([_short_scenario_label(name) for name in scenarios], color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=8)
        ax.grid(axis="y", color="white", alpha=0.12, lw=0.6)
        for spine in ax.spines.values():
            spine.set_color("#40506b")
        ax.set_title(metric, color="white", fontsize=10)
    axes[0, 1].legend(facecolor="#0a1120", edgecolor="#40506b", labelcolor="white", fontsize=8)
    path = os.path.join(OUT_DIR, "overview_metrics.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[save] {path}")


def _write_summary_csv(records: list[RunRecord]):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "summary.csv")
    fieldnames = ["scenario", "cost_profile"] + list(_compute_metrics(records[0].scenario, records[0].histories).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.metrics)
    print(f"[save] {path}")


def _short_scenario_label(name: str) -> str:
    return name.replace("signalized_intersection_", "").replace("signalized_intersection", "critical")


def _short_cost_label(cost: str) -> str:
    if cost == "signalized_intersection":
        return "full"
    return cost.replace("signalized_intersection_", "").replace("_", " ")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare signalized-intersection cost profiles.")
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--costs", nargs="+", default=list(DEFAULT_COSTS))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached closed-loop histories and rerun all simulations.",
    )
    return parser.parse_args()


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
    print("[done] signalized-intersection comparison complete")


if __name__ == "__main__":
    main()

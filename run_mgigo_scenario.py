#!/usr/bin/env python3
"""
run_mgigo_scenario.py - MGIGO scenario runner.

Usage:
    uv run python run_mgigo_scenario.py
    uv run python run_mgigo_scenario.py highway_merge highway_merge

Outputs:
    figures/<scenario output prefix>_snapshot.png
    figures/<scenario output prefix>.mp4  (falls back to .gif without ffmpeg)

Current backend:
    The runner accepts scenario/cost profiles independently. The planner now
    consumes ScenarioSpec/DecisionSpec/BlockSpec, and the default backend is
    the generic scenario runner/renderer.
"""

import argparse
import os
import warnings

import jax
from jax import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from backends import get_backend
from config import N_MPC_STEPS, RNG_SEED, SNAP_FRAMES
from costs import COST_PROFILES, get_cost_functions
from scenarios import SCENARIOS, get_scenario

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))


def _output_prefix_for(scenario, cost_profile):
    """Choose output filename prefix for scenario/cost combinations."""
    if cost_profile == scenario.cost_profile:
        return scenario.output_prefix
    safe_cost = cost_profile.replace("/", "_").replace("\\", "_")
    return f"{scenario.output_prefix}_{safe_cost}"


def _validate_registered_pair(scenario, cost_profile):
    """Validate that the selected scenario, cost, and backend can run together."""
    cost_functions = get_cost_functions(cost_profile)
    if len(cost_functions) != scenario.n_agents:
        raise ValueError(
            f"Cost profile {cost_profile!r} provides {len(cost_functions)} "
            f"agent costs, but scenario {scenario.name!r} has {scenario.n_agents} agents."
        )
    get_backend(scenario.backend)


def main(scenario_name="highway_merge", cost_profile=None):
    """Run one closed-loop MGIGO scenario and save snapshot/video outputs."""
    scenario = get_scenario(scenario_name)
    cost_profile = cost_profile or scenario.cost_profile
    try:
        _validate_registered_pair(scenario, cost_profile)
    except ValueError as exc:
        raise ValueError(
            f"Scenario {scenario.name!r} requests cost profile {cost_profile!r}, "
            "but the scenario/cost/backend registration is incomplete. Register "
            "matching scenario, cost, and backend entries before running."
        ) from exc
    backend_cls = get_backend(scenario.backend)
    backend = backend_cls(scenario, cost_profile)
    output_prefix = _output_prefix_for(scenario, cost_profile)
    n_mpc_steps = scenario.n_mpc_steps or N_MPC_STEPS
    snap_frames = scenario.snap_frames or tuple(SNAP_FRAMES)

    print("初始化 JAX...")
    _ = jax.devices()
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    key = random.PRNGKey(RNG_SEED)

    print(f"场景：{scenario.name} - {scenario.description}")
    print(f"代价：{cost_profile}")
    print(f"后端：{scenario.backend}")
    print(backend.describe_start(n_mpc_steps))

    for step in range(n_mpc_steps):
        key, subkey = random.split(key)
        result = backend.step(subkey, step)

        if step % 5 == 0 or step == n_mpc_steps - 1:
            print(backend.progress_line(step, result))

    print(backend.final_summary())

    print("\n生成 3 列对比图...")
    snap_idxs = [min(f, n_mpc_steps - 1) for f in snap_frames]
    snap_labels = scenario.snap_labels
    fig, axes = plt.subplots(1, 3, figsize=(16, 2.8), gridspec_kw={"wspace": 0.04})
    fig.patch.set_facecolor("#0a1120")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.22, wspace=0.04)

    for col, (idx, label) in enumerate(zip(snap_idxs, snap_labels)):
        backend.render_panel(axes[col], idx, title=label, x_win=44.0)

    handles = backend.legend_handles()
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        fontsize=7.5,
        facecolor="#0a1120",
        edgecolor="#445566",
        labelcolor="white",
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.12),
    )
    snap_path = os.path.join(ROOT, "figures", f"{output_prefix}_snapshot.png")
    fig.savefig(snap_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  已保存 -> {snap_path}")
    plt.close(fig)

    print("\n生成动画...")
    fig_a, ax_a = plt.subplots(figsize=(12, 3.2))
    fig_a.patch.set_facecolor("#0a1120")
    fig_a.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def _animate(frame):
        ax_a.cla()
        idx = min(frame, n_mpc_steps - 1)
        backend.render_panel(
            ax_a,
            idx,
            title=backend.animation_title(idx),
            x_win=44.0,
            show_step=2,
        )
        return (ax_a,)

    anim = FuncAnimation(fig_a, _animate, frames=n_mpc_steps, interval=250, blit=False)
    mp4_path = os.path.join(ROOT, "figures", f"{output_prefix}.mp4")
    gif_path = os.path.join(ROOT, "figures", f"{output_prefix}.gif")
    try:
        writer = FFMpegWriter(
            fps=5,
            bitrate=2200,
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
        )
        anim.save(mp4_path, writer=writer)
        print(f"  已保存 -> {mp4_path}")
    except Exception as exc:
        print(f"  未检测到 ffmpeg（{exc}），改存 GIF...")
        anim.save(gif_path, writer=PillowWriter(fps=5))
        print(f"  已保存 -> {gif_path}")
    plt.close(fig_a)
    print("\n完成！")


def parse_args():
    """Parse CLI arguments for scenario and cost profile selection."""
    parser = argparse.ArgumentParser(
        description="Run MGIGO with independently selectable scenario and cost profile."
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default="highway_merge",
        choices=sorted(SCENARIOS),
        help="Scenario name.",
    )
    parser.add_argument(
        "cost_profile",
        nargs="?",
        default=None,
        choices=sorted(COST_PROFILES),
        help="Cost profile name. Defaults to the scenario's recommended cost.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.scenario, args.cost_profile)

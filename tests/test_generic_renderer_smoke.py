"""Smoke tests for the generic non-signalized scenario renderer."""

from io import BytesIO
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _render_case(name: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import numpy as np

    from scenarios import get_scenario
    from viz_utils import render_agents_panel

    sys.modules.pop("viz_signalized", None)
    scenario = get_scenario(name)
    states = {
        agent.name: scenario.initial_states[agent.state_index]
        for agent in scenario.agents
    }
    histories = {
        agent.name: [scenario.initial_states[agent.state_index]]
        for agent in scenario.agents
    }

    fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
    try:
        render_agents_panel(
            ax,
            scenario,
            states_by_agent=states,
            trajectories_by_agent={agent.name: [] for agent in scenario.agents},
            history_by_agent=histories,
            focus_agent=scenario.agent_names[0],
            x_win=58.0,
            title=name,
        )
        if not ax.patches:
            raise AssertionError(f"{name} should draw road/vehicle patches")
        if not ax.lines:
            raise AssertionError(f"{name} should draw lane boundary lines")
        gids = {patch.get_gid() for patch in ax.patches if patch.get_gid()}
        forbidden_gids = {
            "traffic_signal",
            "conflict_box",
            "cross_vehicle_sample",
            "cross_vehicle_critical",
        }
        leaked_gids = gids & forbidden_gids
        if leaked_gids:
            raise AssertionError(f"{name} leaked signalized layers: {leaked_gids}")
        if "viz_signalized" in sys.modules:
            raise AssertionError(f"{name} should not import signalized renderer")

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    finally:
        plt.close(fig)

    buf.seek(0)
    pixels = mpimg.imread(buf)
    if pixels.size == 0:
        raise AssertionError(f"{name} render produced an empty image")
    if float(np.std(pixels)) <= 0.001:
        raise AssertionError(f"{name} render looks blank")


def test_generic_renderer_handles_highway_merge():
    _render_case("highway_merge")


def test_generic_renderer_handles_borrow_overtake_variants():
    for name in (
        "borrow_overtake_safe",
        "borrow_overtake_blocked",
        "borrow_overtake_critical",
    ):
        _render_case(name)


if __name__ == "__main__":
    test_generic_renderer_handles_highway_merge()
    test_generic_renderer_handles_borrow_overtake_variants()
    print("generic renderer smoke tests ok")

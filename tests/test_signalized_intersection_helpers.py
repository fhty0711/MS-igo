"""Tests for the signalized intersection benchmark helpers."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_signalized_intersection_scenario_contract():
    from scenarios import get_scenario

    scenario = get_scenario("signalized_intersection")
    if scenario.name != "signalized_intersection":
        raise AssertionError(scenario.name)
    if scenario.n_agents != 1:
        raise AssertionError(f"expected one optimizing ego agent, got {scenario.n_agents}")
    if scenario.agent_names != ("ego",):
        raise AssertionError(scenario.agent_names)
    if scenario.solver_spec.n_blocks != 2:
        raise AssertionError(f"expected acc and steer blocks, got {scenario.solver_spec}")
    if scenario.cost_profile != "signalized_intersection":
        raise AssertionError(scenario.cost_profile)


def test_signalized_intersection_variants_are_registered():
    from scenarios import get_scenario

    expected = {
        "signalized_intersection_easy_pass",
        "signalized_intersection_must_stop",
        "signalized_intersection_critical",
    }
    for name in expected:
        scenario = get_scenario(name)
        if scenario.name != name:
            raise AssertionError((name, scenario.name))
        if scenario.cost_profile != "signalized_intersection":
            raise AssertionError((name, scenario.cost_profile))
        if scenario.n_mpc_steps is None or scenario.n_mpc_steps < 28:
            raise AssertionError(f"{name} should run long enough to show the full dilemma")


def test_signalized_intersection_variant_timing_order():
    from scenarios import get_scenario

    easy = get_scenario("signalized_intersection_easy_pass")
    stop = get_scenario("signalized_intersection_must_stop")
    critical = get_scenario("signalized_intersection_critical")

    if not easy.initial_states[0, 2] > stop.initial_states[0, 2]:
        raise AssertionError("easy pass should start faster than must-stop")
    if not stop.n_mpc_steps >= critical.n_mpc_steps:
        raise AssertionError("must-stop should be long enough to show waiting")


def test_signalized_intersection_variant_timing_reaches_cost_context():
    import jax.numpy as jnp
    from costs import signalized_intersection as cost
    from scenarios import get_scenario

    stop = get_scenario("signalized_intersection_must_stop")
    context = jnp.concatenate(
        [
            jnp.asarray(stop.initial_states.reshape(-1), dtype=jnp.float32),
            jnp.asarray(stop.v_refs, dtype=jnp.float32),
            jnp.asarray(stop.context_values, dtype=jnp.float32),
        ]
    )
    red_start = float(cost._red_start_from_context_arr(context))
    if abs(red_start - 1.6) > 1e-6:
        raise AssertionError(f"expected must-stop red start 1.6, got {red_start}")


def test_cross_traffic_noise_is_multimodal_and_small_by_default():
    import jax
    from costs import signalized_intersection as cost

    samples = cost._cross_traffic_noise(jax.random.PRNGKey(0), (40,))
    modes = set(int(v) for v in samples[:, cost.XI_MODE].tolist())
    expected = {
        cost.MODE_OBEY,
        cost.MODE_YELLOW_RUSH,
        cost.MODE_RED_RUN,
    }
    if modes != expected:
        raise AssertionError(f"expected modes {expected}, got {modes}")


def test_no_blocking_intersection_violation():
    import jax.numpy as jnp
    from costs import signalized_intersection as cost

    stopped_before = jnp.array([[30.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    blocking = jnp.array([[42.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    cleared = jnp.array([[55.0, 0.0, 8.0, 0.0, 0.0, 0.0]])

    if float(cost._no_blocking_intersection_from_ego_traj(stopped_before)) > 0.0:
        raise AssertionError("stopping before line should not block")
    if float(cost._no_blocking_intersection_from_ego_traj(cleared)) > 0.0:
        raise AssertionError("clearing intersection should not block")
    if float(cost._no_blocking_intersection_from_ego_traj(blocking)) <= 0.0:
        raise AssertionError("stopping inside conflict box should block")


def test_signalized_intersection_cost_profile_contract():
    import jax.numpy as jnp
    from costs import get_cost_functions
    from scenarios import get_scenario

    scenario = get_scenario("signalized_intersection")
    cost_functions = get_cost_functions("signalized_intersection")
    if len(cost_functions) != scenario.n_agents:
        raise AssertionError(
            f"expected {scenario.n_agents} cost function, got {len(cost_functions)}"
        )

    width = max(scenario.solver_spec.block_dims)
    sample = jnp.zeros((scenario.solver_spec.n_blocks * width,), dtype=jnp.float32)
    context = jnp.concatenate(
        [
            jnp.asarray(scenario.initial_states.reshape(-1), dtype=jnp.float32),
            jnp.asarray(scenario.v_refs, dtype=jnp.float32),
        ]
    )
    value = cost_functions[0](sample, context)
    if not bool(jnp.isfinite(value)):
        raise AssertionError(f"cost should be finite, got {value}")


def test_signalized_intersection_metrics_classify_stop_pass_and_blocking():
    import numpy as np
    from costs import signalized_intersection as cost

    stop_traj = np.array([[30.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    pass_traj = np.array([[56.0, 0.0, 6.0, 0.0, 0.0, 0.0]])
    block_traj = np.array([[42.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    if cost.classify_ego_mode(stop_traj) != "stop":
        raise AssertionError("expected stop")
    if cost.classify_ego_mode(pass_traj) != "pass":
        raise AssertionError("expected pass")
    if cost.classify_ego_mode(block_traj) != "undecided":
        raise AssertionError("expected undecided/blocking")


def test_signalized_intersection_visual_metrics_have_expected_keys():
    import numpy as np
    from costs import signalized_intersection as cost

    ego_traj = np.array(
        [
            [30.0, 0.0, 4.0, 0.0, 0.0, 0.0],
            [34.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [38.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    metrics = cost.estimate_visual_metrics(ego_traj, n_samples=12)
    expected = {
        "mode",
        "min_clearance",
        "risk_quantile",
        "red_legal",
        "no_blocking",
        "critical_sample",
    }
    if set(metrics) != expected:
        raise AssertionError(metrics)


def test_signalized_intersection_render_smoke():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scenarios import get_scenario
    from viz_utils import render_agents_panel

    scenario = get_scenario("signalized_intersection")
    states_by_agent = {"ego": scenario.initial_states[0]}
    fig, ax = plt.subplots(figsize=(5, 3))
    render_agents_panel(
        ax,
        scenario,
        states_by_agent=states_by_agent,
        trajectories_by_agent={"ego": []},
        history_by_agent={"ego": [scenario.initial_states[0]]},
        focus_agent="ego",
        x_win=44.0,
        title="smoke",
    )
    if len(ax.patches) == 0:
        raise AssertionError("expected vehicle/intersection patches")
    plt.close(fig)


if __name__ == "__main__":
    test_signalized_intersection_scenario_contract()
    test_signalized_intersection_variants_are_registered()
    test_signalized_intersection_variant_timing_order()
    test_signalized_intersection_variant_timing_reaches_cost_context()
    test_cross_traffic_noise_is_multimodal_and_small_by_default()
    test_no_blocking_intersection_violation()
    test_signalized_intersection_cost_profile_contract()
    test_signalized_intersection_metrics_classify_stop_pass_and_blocking()
    test_signalized_intersection_visual_metrics_have_expected_keys()
    test_signalized_intersection_render_smoke()
    print("signalized intersection helper tests ok")

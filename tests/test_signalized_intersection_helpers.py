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


def test_signalized_intersection_uses_double_lane_intersection_geometry():
    from scenarios import get_scenario
    from scenarios import signalized_intersection as geom

    scenario = get_scenario("signalized_intersection")
    if scenario.road.lane_centers != geom.EGO_ROAD_LANE_CENTERS:
        raise AssertionError(
            f"horizontal road lanes should match geometry constants: "
            f"{scenario.road.lane_centers} != {geom.EGO_ROAD_LANE_CENTERS}"
        )
    if len(scenario.road.lane_centers) != 4:
        raise AssertionError(f"expected two lanes per direction, got {scenario.road}")
    if geom.EGO_LANE_Y not in scenario.road.lane_centers:
        raise AssertionError(f"ego lane {geom.EGO_LANE_Y} not in {scenario.road}")
    if geom.CROSS_LANE_X not in geom.CROSS_ROAD_LANE_CENTERS:
        raise AssertionError(
            f"cross lane {geom.CROSS_LANE_X} not in {geom.CROSS_ROAD_LANE_CENTERS}"
        )
    if len(geom.CROSS_ROAD_LANE_CENTERS) != 4:
        raise AssertionError(
            f"expected vertical road to have two lanes per direction, got "
            f"{geom.CROSS_ROAD_LANE_CENTERS}"
        )
    if not geom.STOP_LINE_X < geom.INTERSECTION_ENTRY_X < geom.CROSS_LANE_X:
        raise AssertionError(
            "stop line should precede conflict box and selected cross lane"
        )
    if not geom.CROSS_LANE_X < geom.INTERSECTION_EXIT_X:
        raise AssertionError("selected cross lane should lie inside conflict box")


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


def test_cross_traffic_rollout_uses_selected_vertical_lane():
    import jax.numpy as jnp
    from costs import signalized_intersection as cost
    from scenarios import signalized_intersection as geom

    xi = jnp.array([float(cost.MODE_YELLOW_RUSH), 0.0, 1.0, 0.0], dtype=jnp.float32)
    traj = cost._cross_traj_for_xi(xi, 8)
    if not bool(jnp.allclose(traj[:, 0], geom.CROSS_LANE_X)):
        raise AssertionError(f"cross traffic should stay in selected vertical lane: {traj[:, 0]}")


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


def test_signalized_intersection_cost_hierarchy_uses_constran_presets():
    from costs import signalized_intersection as cost

    specs = cost.EGO_CONSTRAINT_SPECS
    if len(specs) != 5:
        raise AssertionError(f"expected five ego constraint layers, got {len(specs)}")

    by_name = {name: spec for name, spec in specs}
    expected = {
        "red_light",
        "road_boundary",
        "no_blocking_intersection",
        "cross_traffic_chance",
        "dilemma_task",
    }
    if set(by_name) != expected:
        raise AssertionError(f"unexpected constraint names {set(by_name)}")

    for name in ("red_light", "road_boundary", "no_blocking_intersection"):
        spec = by_name[name]
        if spec.mode != "tunable" or spec.tune_preset != "__hard__":
            raise AssertionError(f"{name} should be hard-normalized, got {spec}")
        if spec.priority != 1 or spec.transform != "sharp":
            raise AssertionError(f"{name} should be priority-1 sharp hard layer, got {spec}")

    risk = by_name["cross_traffic_chance"]
    if risk.priority != 2 or risk.mode != "tunable":
        raise AssertionError(risk)
    if risk.tune_preset != "firm" or risk.transform != "standard":
        raise AssertionError(
            "cross-traffic chance should use Constran firm/standard presets, "
            f"got tune={risk.tune_preset}, transform={risk.transform}"
        )
    if risk.n_samples != cost.DEV_N_SAMPLES or risk.aggregate != "":
        raise AssertionError(risk)

    dilemma = by_name["dilemma_task"]
    if dilemma.priority != 3 or dilemma.mode != "tunable":
        raise AssertionError(dilemma)
    if dilemma.tune_preset != "standard" or dilemma.transform != "standard":
        raise AssertionError(dilemma)


def test_signalized_intersection_objective_tracks_ego_lane_center():
    import jax.numpy as jnp
    from costs import signalized_intersection as cost
    from scenarios import get_scenario

    scenario = get_scenario("signalized_intersection")
    n_steps = 6
    base = jnp.zeros((n_steps, scenario.n_agents, scenario.state_dim), dtype=jnp.float32)
    x = jnp.linspace(0.0, 20.0, n_steps)
    lane_traj = base.at[:, 0, 0].set(x).at[:, 0, 1].set(scenario.target_y).at[:, 0, 2].set(10.0)
    center_traj = base.at[:, 0, 0].set(x).at[:, 0, 1].set(0.0).at[:, 0, 2].set(10.0)
    decisions = {
        "ego_acc": jnp.zeros((scenario.control_horizon,), dtype=jnp.float32),
        "ego_steer": jnp.zeros((scenario.control_horizon,), dtype=jnp.float32),
    }
    context_arr = jnp.concatenate(
        [
            jnp.asarray(scenario.initial_states.reshape(-1), dtype=jnp.float32),
            jnp.asarray(scenario.v_refs, dtype=jnp.float32),
            jnp.asarray(scenario.context_values, dtype=jnp.float32),
        ]
    )

    lane_value = cost._ego_objective(
        None,
        {"dense_traj": lane_traj, "decisions": decisions, "context_arr": context_arr},
    )
    center_value = cost._ego_objective(
        None,
        {"dense_traj": center_traj, "decisions": decisions, "context_arr": context_arr},
    )
    if not float(lane_value) < float(center_value):
        raise AssertionError(
            f"ego objective should prefer target lane center {scenario.target_y}: "
            f"lane={lane_value}, center={center_value}"
        )


def test_signalized_intersection_ablation_profiles_are_registered_and_finite():
    import jax.numpy as jnp
    from costs import get_cost_functions
    from scenarios import get_scenario

    scenario = get_scenario("signalized_intersection")
    width = max(scenario.solver_spec.block_dims)
    sample = jnp.zeros((scenario.solver_spec.n_blocks * width,), dtype=jnp.float32)
    context = jnp.concatenate(
        [
            jnp.asarray(scenario.initial_states.reshape(-1), dtype=jnp.float32),
            jnp.asarray(scenario.v_refs, dtype=jnp.float32),
            jnp.asarray(scenario.context_values, dtype=jnp.float32),
        ]
    )
    profiles = (
        "signalized_intersection_no_chance",
        "signalized_intersection_single_mode",
        "signalized_intersection_soft_dilemma",
    )
    for profile in profiles:
        cost_functions = get_cost_functions(profile)
        if len(cost_functions) != scenario.n_agents:
            raise AssertionError((profile, len(cost_functions)))
        value = cost_functions[0](sample, context)
        if not bool(jnp.isfinite(value)):
            raise AssertionError(f"{profile} cost should be finite, got {value}")


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


def test_signalized_intersection_comparison_runner_contract():
    import numpy as np

    import compare_signalized_intersection_profiles as compare
    from scenarios import get_scenario

    expected_scenarios = (
        "signalized_intersection_easy_pass",
        "signalized_intersection_must_stop",
        "signalized_intersection_critical",
    )
    expected_costs = (
        "signalized_intersection",
        "signalized_intersection_no_chance",
        "signalized_intersection_single_mode",
        "signalized_intersection_soft_dilemma",
    )
    if compare.DEFAULT_SCENARIOS != expected_scenarios:
        raise AssertionError(compare.DEFAULT_SCENARIOS)
    if compare.DEFAULT_COSTS != expected_costs:
        raise AssertionError(compare.DEFAULT_COSTS)
    expected_outcomes = (
        "mode_outcome",
        "red_legal",
        "no_blocking",
        "cleared_intersection",
        "stopped_before_line",
    )
    if compare.OUTCOME_METRICS != expected_outcomes:
        raise AssertionError(compare.OUTCOME_METRICS)

    scenario = get_scenario("signalized_intersection")
    ego = np.array(
        [
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            [30.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [33.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    metrics = compare._compute_metrics(scenario, {"ego": ego})
    expected_keys = {
        "mode",
        "final_x",
        "final_y",
        "final_v",
        "min_clearance",
        "risk_quantile",
        "red_legal",
        "no_blocking",
        "cleared_intersection",
        "stopped_before_line",
    }
    if set(metrics) != expected_keys:
        raise AssertionError(metrics)
    if metrics["mode"] != "stop":
        raise AssertionError(metrics)


def test_signalized_intersection_report_rows_render():
    import generate_signalized_intersection_report as report

    rows = [
        {
            "scenario": "signalized_intersection_must_stop",
            "cost_profile": "signalized_intersection",
            "mode": "stop",
            "final_x": "33.2",
            "final_v": "0.0",
            "min_clearance": "12.3",
            "risk_quantile": "-8.0",
            "red_legal": "True",
            "no_blocking": "True",
            "cleared_intersection": "False",
            "stopped_before_line": "True",
        }
    ]
    markdown = report._build_markdown(rows, generated="TEST")
    html = report._build_html(rows, generated="TEST")
    required = (
        "black-box",
        "multi-modal",
        "signalized_intersection_no_chance",
        "signalized_intersection_single_mode",
        "signalized_intersection_soft_dilemma",
        "must_stop",
    )
    for token in required:
        if token not in markdown:
            raise AssertionError(token)
    if "<table>" not in html or "Signalized Intersection" not in html:
        raise AssertionError(html[:200])
    if "overview_outcomes.png" not in markdown or "overview_outcomes.png" not in html:
        raise AssertionError("report should include outcome metrics figure")


def test_signalized_semantic_layers_contract():
    from scenarios import get_scenario
    import viz_signalized

    scenario = get_scenario("signalized_intersection")
    layers = viz_signalized.semantic_layer_summary(scenario)
    expected_exact = {
        "horizontal_lanes": 4,
        "vertical_lanes": 4,
        "crosswalks": 2,
    }
    for key, expected in expected_exact.items():
        if layers.get(key) != expected:
            raise AssertionError(f"{key}: expected {expected}, got {layers}")
    minimums = {
        "stop_lines": 1,
        "direction_arrows": 4,
        "risk_cloud_samples": 30,
        "traffic_signals": 1,
    }
    for key, minimum in minimums.items():
        if layers.get(key, 0) < minimum:
            raise AssertionError(f"{key}: expected at least {minimum}, got {layers}")


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
    if len(ax.patches) < 6:
        raise AssertionError("expected vehicle plus multi-lane intersection patches")
    gids = {patch.get_gid() for patch in ax.patches if patch.get_gid()}
    expected_gids = {
        "lane_polygon",
        "crosswalk",
        "conflict_box",
        "traffic_signal",
    }
    missing = expected_gids - gids
    if missing:
        raise AssertionError(f"missing semantic map patch labels: {missing}")
    plt.close(fig)


if __name__ == "__main__":
    test_signalized_intersection_scenario_contract()
    test_signalized_intersection_uses_double_lane_intersection_geometry()
    test_signalized_intersection_variants_are_registered()
    test_signalized_intersection_variant_timing_order()
    test_signalized_intersection_variant_timing_reaches_cost_context()
    test_cross_traffic_noise_is_multimodal_and_small_by_default()
    test_cross_traffic_rollout_uses_selected_vertical_lane()
    test_no_blocking_intersection_violation()
    test_signalized_intersection_cost_profile_contract()
    test_signalized_intersection_cost_hierarchy_uses_constran_presets()
    test_signalized_intersection_objective_tracks_ego_lane_center()
    test_signalized_intersection_ablation_profiles_are_registered_and_finite()
    test_signalized_intersection_metrics_classify_stop_pass_and_blocking()
    test_signalized_intersection_visual_metrics_have_expected_keys()
    test_signalized_intersection_comparison_runner_contract()
    test_signalized_intersection_report_rows_render()
    test_signalized_semantic_layers_contract()
    test_signalized_intersection_render_smoke()
    print("signalized intersection helper tests ok")

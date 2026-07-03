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
    if scenario.exec_mode != "cost_select":
        raise AssertionError("signalized benchmark should execute by physical cost")
    if not scenario.initial_component_means:
        raise AssertionError("signalized benchmark should seed stop/pass mixture components")


def test_scenario_spec_accepts_bspline_decision_kinds():
    import numpy as np

    from scenarios.spec import AgentSpec, BlockSpec, DecisionSpec, ScenarioSpec, state

    scenario = ScenarioSpec(
        name="bspline_contract",
        title="B-spline contract",
        description="ScenarioSpec should accept Frenet B-spline decision vectors.",
        output_prefix="bspline_contract",
        cost_profile="signalized_intersection",
        initial_states=np.array([state(0.0, 0.0, 10.0)]),
        v_refs=np.array([10.0]),
        target_y=0.0,
        lane_roles=("ego",),
        agent_roles=("ego",),
        agents=(
            AgentSpec(
                name="ego",
                role="ego",
                dynamics="bicycle",
                state_index=0,
                reference_index=0,
            ),
        ),
        decisions=(
            DecisionSpec("ego_ctrl_s", "ego", "ctrl_s", (7,)),
            DecisionSpec("ego_ctrl_d", "ego", "ctrl_d", (5,)),
        ),
        blocks=(
            BlockSpec("ego_ctrl_s", "ego", ("ego_ctrl_s",), 0),
            BlockSpec("ego_ctrl_d", "ego", ("ego_ctrl_d",), 1),
        ),
        snap_labels=("start", "middle", "end"),
        trajectory_model="frenet_bspline",
        control_horizon=12,
        initial_component_means=(((0.0, 5.0, 12.0), (0.0, 0.0, 0.0)),),
    )

    if scenario.trajectory_model != "frenet_bspline":
        raise AssertionError(scenario.trajectory_model)
    if scenario.solver_spec.block_dims != (7, 5):
        raise AssertionError(scenario.solver_spec.block_dims)


def test_signalized_intersection_bspline_scenario_contract():
    from scenarios import get_scenario

    scenario = get_scenario("signalized_intersection_bspline")
    if scenario.name != "signalized_intersection_bspline":
        raise AssertionError(scenario.name)
    if scenario.trajectory_model != "frenet_bspline":
        raise AssertionError(scenario.trajectory_model)
    if scenario.cost_profile != "signalized_intersection_bspline":
        raise AssertionError(scenario.cost_profile)
    if scenario.agent_names != ("ego",):
        raise AssertionError(scenario.agent_names)
    if tuple(decision.name for decision in scenario.decisions) != (
        "ego_ctrl_s",
        "ego_ctrl_d",
    ):
        raise AssertionError(scenario.decisions)
    if tuple(decision.kind for decision in scenario.decisions) != ("ctrl_s", "ctrl_d"):
        raise AssertionError(scenario.decisions)
    if scenario.solver_spec.block_dims != (9, 9):
        raise AssertionError(scenario.solver_spec.block_dims)
    if len(scenario.initial_component_means) != scenario.solver_spec.n_blocks:
        raise AssertionError(scenario.initial_component_means)
    for block_idx, block_means in enumerate(scenario.initial_component_means):
        if len(block_means) < 3:
            raise AssertionError(block_means)
        for component_mean in block_means:
            if len(component_mean) != scenario.solver_spec.block_dims[block_idx]:
                raise AssertionError((block_idx, component_mean))


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


def test_axis_aligned_penetration_requires_overlap_on_both_axes():
    import jax.numpy as jnp
    import numpy as np
    from costs import signalized_intersection as cost

    origin = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    same_x_far_y = jnp.array([[0.0, 20.0, 0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    far_x_same_y = jnp.array([[20.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    overlapping = jnp.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)

    if float(cost._axis_aligned_pair_penetration(origin, same_x_far_y)[0]) > 0.0:
        raise AssertionError("same x but separated y should not collide")
    if float(cost._axis_aligned_pair_penetration(origin, far_x_same_y)[0]) > 0.0:
        raise AssertionError("same y but separated x should not collide")
    if float(cost._axis_aligned_pair_penetration(origin, overlapping)[0]) <= 0.0:
        raise AssertionError("overlap on both AABB axes should collide")

    if cost._np_pair_clearance(np.asarray(origin), np.asarray(same_x_far_y)) <= 0.0:
        raise AssertionError("NumPy clearance should be positive when y axis separates")
    if cost._np_pair_clearance(np.asarray(origin), np.asarray(far_x_same_y)) <= 0.0:
        raise AssertionError("NumPy clearance should be positive when x axis separates")
    if cost._np_pair_clearance(np.asarray(origin), np.asarray(overlapping)) >= 0.0:
        raise AssertionError("NumPy clearance should be negative when AABBs overlap")


def test_obey_cross_traffic_stops_before_horizontal_road_conflict_band():
    import jax.numpy as jnp
    from costs import signalized_intersection as cost

    xi = jnp.array([float(cost.MODE_OBEY), 0.0, 1.0, 0.0], dtype=jnp.float32)
    traj = cost._cross_traj_for_xi(xi, 40)
    front_y = traj[:, 1] + 0.5 * cost._SCENARIO.vehicle_geometry.length
    expected_front_limit = (
        cost._SCENARIO.road.road_min_y
        - 0.5 * cost._SCENARIO.vehicle_geometry.safe_gap
    )
    if float(jnp.max(front_y)) > expected_front_limit + 1e-6:
        raise AssertionError(
            "obey-mode cross traffic should stop before entering the horizontal road: "
            f"front_y={float(jnp.max(front_y))}, limit={expected_front_limit}"
        )


def test_no_blocking_intersection_violation():
    import jax.numpy as jnp
    from costs import signalized_intersection as cost

    stopped_before = jnp.array([[30.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    blocking = jnp.array([[42.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    crawling = jnp.array([[42.0, 0.0, 1.1, 0.0, 0.0, 0.0]])
    cleared = jnp.array([[55.0, 0.0, 8.0, 0.0, 0.0, 0.0]])

    if float(cost._no_blocking_intersection_from_ego_traj(stopped_before)) > 0.0:
        raise AssertionError("stopping before line should not block")
    if float(cost._no_blocking_intersection_from_ego_traj(cleared)) > 0.0:
        raise AssertionError("clearing intersection should not block")
    if float(cost._no_blocking_intersection_from_ego_traj(blocking)) <= 0.0:
        raise AssertionError("stopping inside conflict box should block")
    if float(cost._no_blocking_intersection_from_ego_traj(crawling)) <= 0.0:
        raise AssertionError("ending the horizon inside the conflict box should block")


def test_red_light_violation_detects_illegal_crossing_and_allows_stop_or_clear():
    import jax.numpy as jnp
    from costs import signalized_intersection as cost

    context_arr = jnp.concatenate(
        [
            jnp.asarray(cost._SCENARIO.initial_states.reshape(-1), dtype=jnp.float32),
            jnp.asarray(cost._SCENARIO.v_refs, dtype=jnp.float32),
            jnp.array([cost.YELLOW_START_S, 0.0], dtype=jnp.float32),
        ]
    )
    stopped_before = jnp.array(
        [
            [cost.STOP_LINE_X - 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [cost.STOP_LINE_X - 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    already_cleared = jnp.array(
        [
            [cost.INTERSECTION_EXIT_X + 1.0, 0.0, 8.0, 0.0, 0.0, 0.0],
            [cost.INTERSECTION_EXIT_X + 2.0, 0.0, 8.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    illegal_crossing = jnp.array(
        [
            [cost.STOP_LINE_X + 1.0, 0.0, 8.0, 0.0, 0.0, 0.0],
            [cost.INTERSECTION_ENTRY_X + 1.0, 0.0, 8.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    def violation(ego_traj):
        dense = ego_traj.reshape((ego_traj.shape[0], 1, ego_traj.shape[1]))
        return cost._ego_red_light_violation(None, {"dense_traj": dense, "context_arr": context_arr})

    if float(violation(illegal_crossing)) <= 0.0:
        raise AssertionError("illegal red-light crossing should be a positive violation")
    if float(violation(stopped_before)) > 0.0:
        raise AssertionError("stopping before the line on red should be legal")
    if float(violation(already_cleared)) > 0.0:
        raise AssertionError("already-cleared trajectory on red should be legal")


def test_red_light_violation_orders_illegal_depth_and_duration():
    import jax.numpy as jnp
    from costs import signalized_intersection as cost

    context_arr = jnp.concatenate(
        [
            jnp.asarray(cost._SCENARIO.initial_states.reshape(-1), dtype=jnp.float32),
            jnp.asarray(cost._SCENARIO.v_refs, dtype=jnp.float32),
            jnp.array([cost.YELLOW_START_S, 0.0], dtype=jnp.float32),
        ]
    )
    shallow_crossing = jnp.array(
        [
            [cost.STOP_LINE_X + 0.1, 0.0, 3.0, 0.0, 0.0, 0.0],
            [cost.STOP_LINE_X + 0.2, 0.0, 3.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    deeper_longer_crossing = jnp.array(
        [
            [cost.STOP_LINE_X + 2.0, 0.0, 3.0, 0.0, 0.0, 0.0],
            [cost.INTERSECTION_ENTRY_X + 1.0, 0.0, 3.0, 0.0, 0.0, 0.0],
            [cost.INTERSECTION_EXIT_X - 1.0, 0.0, 3.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    def violation(ego_traj):
        dense = ego_traj.reshape((ego_traj.shape[0], 1, ego_traj.shape[1]))
        return cost._ego_red_light_violation(None, {"dense_traj": dense, "context_arr": context_arr})

    shallow_value = float(violation(shallow_crossing))
    deeper_value = float(violation(deeper_longer_crossing))
    if shallow_value <= 0.0 or deeper_value <= 0.0:
        raise AssertionError(
            "red-light crossings after red should remain positive violations: "
            f"shallow={shallow_value}, deeper={deeper_value}"
        )
    if not deeper_value > shallow_value:
        raise AssertionError(
            "red-light violation should distinguish depth/duration instead of returning "
            f"a constant value: shallow={shallow_value}, deeper={deeper_value}"
        )


def test_red_light_violation_uses_elapsed_mpc_time():
    import jax.numpy as jnp
    from costs import signalized_intersection as cost

    red_start = 1.6
    context_arr = jnp.concatenate(
        [
            jnp.asarray(cost._SCENARIO.initial_states.reshape(-1), dtype=jnp.float32),
            jnp.asarray(cost._SCENARIO.v_refs, dtype=jnp.float32),
            jnp.array([0.2, red_start, red_start + 0.2], dtype=jnp.float32),
        ]
    )
    ego = jnp.array(
        [
            [cost.STOP_LINE_X + 0.5, 0.0, 6.0, 0.0, 0.0, 0.0],
            [cost.INTERSECTION_ENTRY_X + 0.5, 0.0, 6.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    dense = ego.reshape((ego.shape[0], 1, ego.shape[1]))
    value = cost._ego_red_light_violation(None, {"dense_traj": dense, "context_arr": context_arr})
    if float(value) <= 0.0:
        raise AssertionError("elapsed MPC time after red start should make crossing illegal")


def test_planner_appends_elapsed_time_to_context():
    import jax
    import jax.numpy as jnp
    import numpy as np

    import planner
    from config import K_COMP
    from scenarios import get_scenario

    scenario = get_scenario("signalized_intersection_must_stop")
    captured = {}

    def fake_solver(**kwargs):
        context = kwargs["context"]
        captured["context"] = np.asarray(context, dtype=float)
        n_blocks = kwargs["N_blocks"]
        width = max(kwargs["dims"])
        mu = jnp.zeros((n_blocks, K_COMP, width), dtype=jnp.float32)
        L = jnp.tile(
            jnp.eye(width, dtype=jnp.float32)[None, None],
            (n_blocks, K_COMP, 1, 1),
        )
        pi = jnp.ones((n_blocks, K_COMP), dtype=jnp.float32) / K_COMP
        return mu, L, pi, None

    old_solver = planner.mmog_igo_rne_blocks_solver
    old_advance = planner.advance_one_macro_step
    old_dense = planner.dense_rollout_np
    old_prediction = planner.prediction_trajs
    try:
        planner.mmog_igo_rne_blocks_solver = fake_solver
        planner.advance_one_macro_step = (
            lambda _scenario, current_states, _decisions: np.asarray(current_states)
        )
        planner.dense_rollout_np = (
            lambda _scenario, current_states, _decisions:
            np.asarray(current_states, dtype=float)[None, :, :]
        )
        planner.prediction_trajs = (
            lambda _scenario, current_states, _decisions:
            {"ego": np.asarray(current_states, dtype=float)}
        )
        planner.plan(
            jax.random.PRNGKey(0),
            scenario.initial_states,
            scenario.v_refs,
            cost_profile=scenario.cost_profile,
            scenario=scenario,
            solver_spec=scenario.solver_spec,
            elapsed_time_s=2.5,
        )
    finally:
        planner.mmog_igo_rne_blocks_solver = old_solver
        planner.advance_one_macro_step = old_advance
        planner.dense_rollout_np = old_dense
        planner.prediction_trajs = old_prediction

    if not captured:
        raise AssertionError("planner did not call solver")
    if abs(captured["context"][-1] - 2.5) > 1e-6:
        raise AssertionError(f"expected elapsed context 2.5, got {captured['context'][-1]}")


def test_planner_accepts_vector_initial_component_means():
    import jax
    import jax.numpy as jnp
    import numpy as np

    import planner
    from config import K_COMP
    from scenarios import get_scenario

    scenario = get_scenario("signalized_intersection_bspline")
    captured = {}

    def fake_solver(**kwargs):
        captured["initial_mu_k"] = np.asarray(kwargs["initial_mu_k"], dtype=float)
        n_blocks = kwargs["N_blocks"]
        width = max(kwargs["dims"])
        mu = jnp.asarray(kwargs["initial_mu_k"], dtype=jnp.float32)
        L = jnp.tile(
            jnp.eye(width, dtype=jnp.float32)[None, None],
            (n_blocks, K_COMP, 1, 1),
        )
        pi = jnp.ones((n_blocks, K_COMP), dtype=jnp.float32) / K_COMP
        return mu, L, pi, None

    old_solver = planner.mmog_igo_rne_blocks_solver
    old_advance = planner.advance_one_macro_step
    old_dense = planner.dense_rollout_np
    old_prediction = planner.prediction_trajs
    try:
        planner.mmog_igo_rne_blocks_solver = fake_solver
        planner.advance_one_macro_step = (
            lambda _scenario, current_states, _decisions: np.asarray(current_states)
        )
        planner.dense_rollout_np = (
            lambda _scenario, current_states, _decisions:
            np.asarray(current_states, dtype=float)[None, :, :]
        )
        planner.prediction_trajs = (
            lambda _scenario, current_states, _decisions:
            {"ego": np.asarray(current_states, dtype=float)}
        )
        planner.plan(
            jax.random.PRNGKey(0),
            scenario.initial_states,
            scenario.v_refs,
            cost_profile=scenario.cost_profile,
            scenario=scenario,
            solver_spec=scenario.solver_spec,
        )
    finally:
        planner.mmog_igo_rne_blocks_solver = old_solver
        planner.advance_one_macro_step = old_advance
        planner.dense_rollout_np = old_dense
        planner.prediction_trajs = old_prediction

    if "initial_mu_k" not in captured:
        raise AssertionError("planner did not call solver")
    mu0 = captured["initial_mu_k"]
    for block_idx, block_means in enumerate(scenario.initial_component_means):
        for comp_idx, expected in enumerate(block_means):
            diff = mu0[block_idx, comp_idx, : scenario.solver_spec.block_dims[block_idx]] - np.asarray(expected)
            if not np.max(np.abs(diff)) < 1.0:
                raise AssertionError((block_idx, comp_idx, diff, expected))


def test_planner_bspline_warm_start_keeps_monotone_longitudinal_seed():
    import jax
    import jax.numpy as jnp
    import numpy as np

    import planner
    from config import K_COMP
    from scenarios import get_scenario

    scenario = get_scenario("signalized_intersection_bspline")
    captured = {}

    def fake_solver(**kwargs):
        captured["initial_mu_k"] = np.asarray(kwargs["initial_mu_k"], dtype=float)
        n_blocks = kwargs["N_blocks"]
        width = max(kwargs["dims"])
        mu = jnp.asarray(kwargs["initial_mu_k"], dtype=jnp.float32)
        L = jnp.tile(
            jnp.eye(width, dtype=jnp.float32)[None, None],
            (n_blocks, K_COMP, 1, 1),
        )
        pi = jnp.ones((n_blocks, K_COMP), dtype=jnp.float32) / K_COMP
        return mu, L, pi, None

    width = max(scenario.solver_spec.block_dims)
    bad_prev = np.zeros((scenario.solver_spec.n_blocks, K_COMP, width), dtype=np.float32)
    bad_selected = np.zeros((scenario.solver_spec.n_blocks, width), dtype=np.float32)
    bad_selected[0, :] = np.linspace(15.0, 90.0, width, dtype=np.float32)
    bad_selected[1, :] = scenario.initial_states[0, 1]
    warm = (bad_prev, np.zeros((scenario.solver_spec.n_blocks,), dtype=np.int32), bad_selected)
    current_states = scenario.initial_states.copy()
    current_states[0, 0] = 25.0

    old_solver = planner.mmog_igo_rne_blocks_solver
    old_advance = planner.advance_one_macro_step
    old_dense = planner.dense_rollout_np
    old_prediction = planner.prediction_trajs
    try:
        planner.mmog_igo_rne_blocks_solver = fake_solver
        planner.advance_one_macro_step = (
            lambda _scenario, states, _decisions: np.asarray(states)
        )
        planner.dense_rollout_np = (
            lambda _scenario, states, _decisions:
            np.asarray(states, dtype=float)[None, :, :]
        )
        planner.prediction_trajs = (
            lambda _scenario, states, _decisions:
            {"ego": np.asarray(states, dtype=float)}
        )
        planner.plan(
            jax.random.PRNGKey(0),
            current_states,
            scenario.v_refs,
            warm=warm,
            cost_profile=scenario.cost_profile,
            scenario=scenario,
            solver_spec=scenario.solver_spec,
        )
    finally:
        planner.mmog_igo_rne_blocks_solver = old_solver
        planner.advance_one_macro_step = old_advance
        planner.dense_rollout_np = old_dense
        planner.prediction_trajs = old_prediction

    seed = captured["initial_mu_k"][0, 0, : scenario.solver_spec.block_dims[0]]
    if np.any(np.diff(seed) <= 0.0):
        raise AssertionError(seed)
    if seed[0] < current_states[0, 0]:
        raise AssertionError((seed, current_states[0, 0]))


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


def test_signalized_intersection_bspline_cost_profile_is_registered_and_finite():
    import jax.numpy as jnp
    from costs import get_cost_functions
    from scenarios import get_scenario

    scenario = get_scenario("signalized_intersection_bspline")
    cost_functions = get_cost_functions("signalized_intersection_bspline")
    if len(cost_functions) != scenario.n_agents:
        raise AssertionError((len(cost_functions), scenario.n_agents))

    width = max(scenario.solver_spec.block_dims)
    sample_blocks = jnp.zeros((scenario.solver_spec.n_blocks, width), dtype=jnp.float32)
    for block_idx, block_means in enumerate(scenario.initial_component_means):
        sample_blocks = sample_blocks.at[
            block_idx, : scenario.solver_spec.block_dims[block_idx]
        ].set(jnp.asarray(block_means[0], dtype=jnp.float32))
    sample = sample_blocks.reshape(-1)
    context = jnp.concatenate(
        [
            jnp.asarray(scenario.initial_states.reshape(-1), dtype=jnp.float32),
            jnp.asarray(scenario.v_refs, dtype=jnp.float32),
            jnp.asarray(scenario.context_values, dtype=jnp.float32),
        ]
    )
    value = cost_functions[0](sample, context)
    if not bool(jnp.isfinite(value)):
        raise AssertionError(f"B-spline signalized cost should be finite, got {value}")


def test_signalized_intersection_bspline_has_hard_physical_feasibility_layer():
    import jax.numpy as jnp
    from costs import signalized_intersection_bspline as cost
    from config import MAX_SPEED

    bad_traj = jnp.array(
        [
            [10.0, -1.75, 12.0, 0.0, 0.0, 0.0],
            [20.0, -1.75, MAX_SPEED + 5.0, 0.0, 0.0, 0.0],
            [18.0, -1.75, 12.0, 3.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    good_traj = jnp.array(
        [
            [10.0, -1.75, 12.0, 0.0, 0.0, 0.0],
            [16.0, -1.75, 12.0, 0.0, 0.0, 0.0],
            [22.0, -1.75, 12.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    if float(cost._bspline_physical_feasibility_violation_from_traj(bad_traj)) <= 0.0:
        raise AssertionError("speed/reversal/heading violation should be positive")
    if float(cost._bspline_physical_feasibility_violation_from_traj(good_traj)) > 0.0:
        raise AssertionError("straight feasible B-spline rollout should satisfy the hard layer")


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

    terminal_in_box = np.array(
        [
            [30.0, 0.0, 5.0, 0.0, 0.0, 0.0],
            [42.0, 0.0, 1.1, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    terminal_metrics = cost.estimate_visual_metrics(terminal_in_box, n_samples=12)
    if terminal_metrics["no_blocking"]:
        raise AssertionError("visual metrics should flag terminal-in-box blocking")

    red_metrics = cost.estimate_visual_metrics(
        terminal_in_box,
        n_samples=12,
        red_start_s=0.5,
        dt=0.5,
        time_offset_s=0.5,
    )
    if red_metrics["red_legal"]:
        raise AssertionError("visual metrics should honor scenario red timing")


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
        "task_success",
        "safety_success",
        "scheme_a_success",
    )
    if set(compare.OUTCOME_METRICS) != set(expected_outcomes):
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
        "scenario_intent",
        "task_success",
        "safety_success",
        "scheme_a_success",
        "paper_claim",
        "failure_reason",
    }
    if set(metrics) != expected_keys:
        raise AssertionError(metrics)
    if metrics["mode"] != "stop":
        raise AssertionError(metrics)

    late_illegal = np.array(
        [
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            [35.0, 0.0, 5.0, 0.0, 0.0, 0.0],
            [38.0, 0.0, 5.0, 0.0, 0.0, 0.0],
            [40.0, 0.0, 5.0, 0.0, 0.0, 0.0],
            [42.0, 0.0, 5.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    stop_scenario = get_scenario("signalized_intersection_must_stop")
    late_metrics = compare._compute_metrics(stop_scenario, {"ego": late_illegal})
    if late_metrics["red_legal"]:
        raise AssertionError("macro-step metrics should use must-stop red timing")


def test_signalized_intersection_metrics_mark_must_stop_as_safe_stop():
    import numpy as np

    import compare_signalized_intersection_profiles as compare
    from costs import signalized_intersection as cost
    from scenarios import get_scenario

    scenario = get_scenario("signalized_intersection_must_stop")
    ego = np.array(
        [
            [4.0, scenario.target_y, 11.0, 0.0, 0.0, 0.0],
            [18.0, scenario.target_y, 8.0, 0.0, 0.0, 0.0],
            [28.0, scenario.target_y, 3.0, 0.0, 0.0, 0.0],
            [cost.STOP_LINE_X - 1.5, scenario.target_y, 0.4, 0.0, 0.0, 0.0],
            [cost.STOP_LINE_X - 1.5, scenario.target_y, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    metrics = compare._compute_metrics(scenario, {"ego": ego})
    if metrics["scenario_intent"] != "must_stop":
        raise AssertionError(metrics)
    if metrics["mode"] != "stop" or not metrics["stopped_before_line"]:
        raise AssertionError(metrics)
    if not metrics["red_legal"] or not metrics["no_blocking"]:
        raise AssertionError(metrics)
    if not metrics["safety_success"] or not metrics["task_success"]:
        raise AssertionError(metrics)
    if metrics["paper_claim"] != "safe_stop":
        raise AssertionError(metrics)


def test_signalized_intersection_metrics_mark_easy_pass_as_safe_pass():
    import numpy as np

    import compare_signalized_intersection_profiles as compare
    from costs import signalized_intersection as cost
    from scenarios import get_scenario

    scenario = get_scenario("signalized_intersection_easy_pass")
    ego = np.array(
        [
            [0.0, scenario.target_y, 15.0, 0.0, 0.0, 0.0],
            [18.0, scenario.target_y, 15.0, 0.0, 0.0, 0.0],
            [cost.INTERSECTION_EXIT_X + 8.0, scenario.target_y, 14.0, 0.0, 0.0, 0.0],
            [cost.INTERSECTION_EXIT_X + 24.0, scenario.target_y, 13.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    metrics = compare._compute_metrics(scenario, {"ego": ego})
    if metrics["scenario_intent"] != "easy_pass":
        raise AssertionError(metrics)
    if metrics["mode"] != "pass" or not metrics["cleared_intersection"]:
        raise AssertionError(metrics)
    if not metrics["red_legal"] or not metrics["no_blocking"]:
        raise AssertionError(metrics)
    if not metrics["safety_success"] or not metrics["task_success"]:
        raise AssertionError(metrics)
    if metrics["paper_claim"] != "safe_pass":
        raise AssertionError(metrics)


def test_signalized_intersection_metrics_allow_critical_safe_stop_or_pass():
    import numpy as np

    import compare_signalized_intersection_profiles as compare
    from costs import signalized_intersection as cost
    from scenarios import get_scenario

    scenario = get_scenario("signalized_intersection_critical")
    safe_stop = np.array(
        [
            [4.0, scenario.target_y, 10.0, 0.0, 0.0, 0.0],
            [18.0, scenario.target_y, 7.0, 0.0, 0.0, 0.0],
            [28.0, scenario.target_y, 3.0, 0.0, 0.0, 0.0],
            [cost.STOP_LINE_X - 1.0, scenario.target_y, 0.5, 0.0, 0.0, 0.0],
            [cost.STOP_LINE_X - 1.0, scenario.target_y, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    safe_pass = np.array(
        [
            [4.0, scenario.target_y, 15.0, 0.0, 0.0, 0.0],
            [cost.INTERSECTION_EXIT_X + 6.0, scenario.target_y, 14.0, 0.0, 0.0, 0.0],
            [cost.INTERSECTION_EXIT_X + 18.0, scenario.target_y, 13.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    stop_metrics = compare._compute_metrics(scenario, {"ego": safe_stop})
    if stop_metrics["scenario_intent"] != "dilemma":
        raise AssertionError(stop_metrics)
    if not stop_metrics["task_success"] or not stop_metrics["safety_success"]:
        raise AssertionError(stop_metrics)
    if stop_metrics["paper_claim"] != "safe_stop":
        raise AssertionError(stop_metrics)

    pass_metrics = compare._compute_metrics(scenario, {"ego": safe_pass})
    if pass_metrics["scenario_intent"] != "dilemma":
        raise AssertionError(pass_metrics)
    if not pass_metrics["task_success"] or not pass_metrics["safety_success"]:
        raise AssertionError(pass_metrics)
    if pass_metrics["paper_claim"] != "safe_pass":
        raise AssertionError(pass_metrics)


def test_signalized_intersection_report_rows_render():
    import generate_signalized_intersection_report as report

    old_row = {
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
    new_row = {
        **old_row,
        "scenario": "signalized_intersection_easy_pass",
        "mode": "pass",
        "scenario_intent": "easy_pass",
        "task_success": "True",
        "safety_success": "True",
        "paper_claim": "safe_pass",
    }
    rows = [old_row, new_row]
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
    new_metric_tokens = ("task success", "paper claim", "safe_pass")
    if not any(token in markdown for token in new_metric_tokens):
        raise AssertionError("new derived metric columns should render in markdown")
    if not any(token in html for token in new_metric_tokens):
        raise AssertionError("new derived metric columns should render in html")
    if "<table>" not in html or "Signalized Intersection" not in html:
        raise AssertionError(html[:200])
    if "overview_outcomes.png" not in markdown or "overview_outcomes.png" not in html:
        raise AssertionError("report should include outcome metrics figure")


def test_signalized_intersection_report_claim_is_conditioned_on_success():
    import generate_signalized_intersection_report as report

    rows = [
        {
            "scenario": "signalized_intersection_critical",
            "cost_profile": "signalized_intersection",
            "mode": "pass",
            "final_x": "94.0",
            "final_v": "8.0",
            "min_clearance": "-1.0",
            "risk_quantile": "0.5",
            "scenario_intent": "dilemma",
            "task_success": "False",
            "safety_success": "False",
            "paper_claim": "unsafe_or_blocked",
            "red_legal": "True",
            "no_blocking": "True",
            "cleared_intersection": "True",
            "stopped_before_line": "False",
        }
    ]
    markdown = report._build_markdown(rows, generated="TEST")
    html = report._build_html(rows, generated="TEST")
    forbidden = "The result supports the paper-level claim"
    if forbidden in markdown or forbidden in html:
        raise AssertionError("failed full profile must not be reported as supporting the claim")
    required = "does not yet support the paper-level success claim"
    if required not in markdown or required not in html:
        raise AssertionError("failed full profile should produce a conditional failure takeaway")


def test_signalized_intersection_report_has_profile_aggregates_and_failure_reasons():
    import generate_signalized_intersection_report as report

    rows = [
        {
            "scenario": "signalized_intersection_easy_pass",
            "cost_profile": "signalized_intersection",
            "mode": "pass",
            "final_x": "120.0",
            "final_v": "5.0",
            "min_clearance": "2.0",
            "risk_quantile": "-1.0",
            "scenario_intent": "easy_pass",
            "task_success": "True",
            "safety_success": "True",
            "paper_claim": "safe_pass",
            "red_legal": "True",
            "no_blocking": "True",
            "cleared_intersection": "True",
            "stopped_before_line": "False",
        },
        {
            "scenario": "signalized_intersection_easy_pass",
            "cost_profile": "signalized_intersection_single_mode",
            "mode": "pass",
            "final_x": "120.0",
            "final_v": "5.0",
            "min_clearance": "-0.5",
            "risk_quantile": "0.2",
            "scenario_intent": "easy_pass",
            "task_success": "False",
            "safety_success": "False",
            "paper_claim": "unsafe_or_blocked",
            "red_legal": "True",
            "no_blocking": "True",
            "cleared_intersection": "True",
            "stopped_before_line": "False",
        },
    ]
    markdown = report._build_markdown(rows, generated="TEST")
    html = report._build_html(rows, generated="TEST")
    for token in ("Profile aggregates", "failure reason", "cross_traffic_conflict"):
        if token not in markdown:
            raise AssertionError(token)
        if token not in html:
            raise AssertionError(token)


def test_signalized_intersection_comparison_declares_manifest_path():
    import compare_signalized_intersection_profiles as compare
    import generate_signalized_intersection_report as report

    if not compare.MANIFEST_PATH.endswith("manifest.json"):
        raise AssertionError(compare.MANIFEST_PATH)
    if "manifest" not in report.FIGURES:
        raise AssertionError(report.FIGURES)


def test_signalized_intersection_derive_metrics_reports_failure_reason():
    import compare_signalized_intersection_profiles as compare

    metrics = {
        "red_legal": True,
        "no_blocking": True,
        "cleared_intersection": True,
        "stopped_before_line": False,
        "min_clearance": -0.2,
    }
    derived = compare._derive_task_metrics("signalized_intersection_easy_pass", metrics)
    if derived["failure_reason"] != "cross_traffic_conflict":
        raise AssertionError(derived)
    if derived["scheme_a_success"]:
        raise AssertionError(derived)


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


def test_signalized_scene_traffic_signal_phase_uses_elapsed_time_gids():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scenarios import get_scenario
    import viz_signalized

    scenario = get_scenario("signalized_intersection")
    if scenario.n_agents != 1:
        raise AssertionError(
            "phase-only visualization should keep cross traffic exogenous, "
            f"got {scenario.n_agents} ScenarioSpec agents"
        )

    yellow_start_s, red_start_s = scenario.context_values[:2]
    phase_cases = (
        (max(0.0, float(yellow_start_s) - 0.1), "traffic_signal_green_active"),
        (0.5 * (float(yellow_start_s) + float(red_start_s)), "traffic_signal_yellow_active"),
        (float(red_start_s) + 0.1, "traffic_signal_red_active"),
    )
    for elapsed_time_s, expected_gid in phase_cases:
        fig, ax = plt.subplots(figsize=(5, 3))
        try:
            viz_signalized.draw_signalized_scene(
                ax,
                scenario,
                20.0,
                60.0,
                elapsed_time_s=elapsed_time_s,
            )
            gids = {patch.get_gid() for patch in ax.patches if patch.get_gid()}
            if expected_gid not in gids:
                raise AssertionError(
                    f"elapsed_time_s={elapsed_time_s} should mark {expected_gid}; "
                    f"got semantic patch gids {gids}"
                )
        finally:
            plt.close(fig)


def test_signalized_render_panel_passes_elapsed_time_to_signal_phase():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scenarios import get_scenario
    from viz_utils import render_agents_panel

    scenario = get_scenario("signalized_intersection_must_stop")
    red_start_s = float(scenario.context_values[1])
    fig, ax = plt.subplots(figsize=(5, 3))
    try:
        render_agents_panel(
            ax,
            scenario,
            states_by_agent={"ego": scenario.initial_states[0]},
            trajectories_by_agent={"ego": []},
            history_by_agent={"ego": [scenario.initial_states[0]]},
            focus_agent="ego",
            x_win=44.0,
            elapsed_time_s=red_start_s + 0.1,
        )
        gids = {patch.get_gid() for patch in ax.patches if patch.get_gid()}
        if "traffic_signal_red_active" not in gids:
            raise AssertionError(
                "render_agents_panel should pass elapsed_time_s to signal scene; "
                f"got gids {gids}"
            )
    finally:
        plt.close(fig)


def test_signalized_cross_traffic_cloud_draws_vehicle_footprint_patches():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from costs import signalized_intersection as cost
    from scenarios import get_scenario
    import viz_signalized

    scenario = get_scenario("signalized_intersection")
    if scenario.n_agents != 1 or scenario.agent_names != ("ego",):
        raise AssertionError(
            "cross traffic should remain an exogenous probability model, "
            f"not a ScenarioSpec agent: {scenario.agent_names}"
        )

    ego_traj = np.array(
        [
            [cost.STOP_LINE_X - 4.0, scenario.target_y, 6.0, 0.0, 0.0, 0.0],
            [cost.INTERSECTION_ENTRY_X + 1.0, scenario.target_y, 6.0, 0.0, 0.0, 0.0],
            [cost.INTERSECTION_EXIT_X + 2.0, scenario.target_y, 6.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(5, 3))
    try:
        viz_signalized.draw_signalized_scene(ax, scenario, 20.0, 60.0)
        viz_signalized.draw_cross_traffic_cloud(ax, scenario, ego_traj=ego_traj)
        gids = {patch.get_gid() for patch in ax.patches if patch.get_gid()}
        expected = {"cross_vehicle_sample", "cross_vehicle_critical"}
        missing = expected - gids
        if missing:
            raise AssertionError(
                "cross traffic visualization should draw semantic vehicle footprint "
                f"patches with stable gids {expected}, missing {missing}; got {gids}"
            )
    finally:
        plt.close(fig)


if __name__ == "__main__":
    test_signalized_intersection_scenario_contract()
    test_scenario_spec_accepts_bspline_decision_kinds()
    test_signalized_intersection_bspline_scenario_contract()
    test_signalized_intersection_uses_double_lane_intersection_geometry()
    test_signalized_intersection_variants_are_registered()
    test_signalized_intersection_variant_timing_order()
    test_signalized_intersection_variant_timing_reaches_cost_context()
    test_cross_traffic_noise_is_multimodal_and_small_by_default()
    test_cross_traffic_rollout_uses_selected_vertical_lane()
    test_axis_aligned_penetration_requires_overlap_on_both_axes()
    test_obey_cross_traffic_stops_before_horizontal_road_conflict_band()
    test_no_blocking_intersection_violation()
    test_red_light_violation_detects_illegal_crossing_and_allows_stop_or_clear()
    test_red_light_violation_orders_illegal_depth_and_duration()
    test_red_light_violation_uses_elapsed_mpc_time()
    test_planner_appends_elapsed_time_to_context()
    test_planner_accepts_vector_initial_component_means()
    test_planner_bspline_warm_start_keeps_monotone_longitudinal_seed()
    test_signalized_intersection_cost_profile_contract()
    test_signalized_intersection_cost_hierarchy_uses_constran_presets()
    test_signalized_intersection_objective_tracks_ego_lane_center()
    test_signalized_intersection_ablation_profiles_are_registered_and_finite()
    test_signalized_intersection_bspline_cost_profile_is_registered_and_finite()
    test_signalized_intersection_bspline_has_hard_physical_feasibility_layer()
    test_signalized_intersection_metrics_classify_stop_pass_and_blocking()
    test_signalized_intersection_visual_metrics_have_expected_keys()
    test_signalized_intersection_comparison_runner_contract()
    test_signalized_intersection_metrics_mark_must_stop_as_safe_stop()
    test_signalized_intersection_metrics_mark_easy_pass_as_safe_pass()
    test_signalized_intersection_metrics_allow_critical_safe_stop_or_pass()
    test_signalized_intersection_report_rows_render()
    test_signalized_intersection_report_claim_is_conditioned_on_success()
    test_signalized_intersection_report_has_profile_aggregates_and_failure_reasons()
    test_signalized_intersection_comparison_declares_manifest_path()
    test_signalized_intersection_derive_metrics_reports_failure_reason()
    test_signalized_semantic_layers_contract()
    test_signalized_intersection_render_smoke()
    test_signalized_scene_traffic_signal_phase_uses_elapsed_time_gids()
    test_signalized_render_panel_passes_elapsed_time_to_signal_phase()
    test_signalized_cross_traffic_cloud_draws_vehicle_footprint_patches()
    print("signalized intersection helper tests ok")

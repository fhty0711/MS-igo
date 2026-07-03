"""Tests for Frenet B-spline trajectory helpers."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax.numpy as jnp
import numpy as np


def test_straight_reference_path_maps_frenet_to_cartesian():
    from trajectory.reference_path import StraightReference

    ref = StraightReference()
    s = jnp.array([0.0, 10.0, 25.0])
    d = jnp.array([-1.75, 0.0, 2.0])
    x, y = ref.frenet_to_cartesian(s, d)

    if not np.allclose(np.asarray(x), np.asarray(s)):
        raise AssertionError((x, s))
    if not np.allclose(np.asarray(y), np.asarray(d)):
        raise AssertionError((y, d))


def test_frenet_bspline_loads_basis_asset():
    from trajectory.frenet_bspline import FrenetBSplineTrajectory
    from trajectory.reference_path import StraightReference

    basis = Path("trajectory/assets/bspline_basis.npz")
    gen = FrenetBSplineTrajectory(basis, StraightReference())

    if gen.n_ctrl != 12:
        raise AssertionError(gen.n_ctrl)
    if gen.n_free != 9:
        raise AssertionError(gen.n_free)
    if gen.T != 100:
        raise AssertionError(gen.T)


def test_bspline_rollout_returns_dense_ego_states():
    from trajectory.frenet_bspline import FrenetBSplineTrajectory
    from trajectory.reference_path import StraightReference
    from trajectory.rollout import bspline_ego_rollout
    from trajectory.warmstart import tangent_control_points

    gen = FrenetBSplineTrajectory("trajectory/assets/bspline_basis.npz", StraightReference())
    ctrl_s, ctrl_d = tangent_control_points(gen, s0=5.0, s_dot0=8.0, d0=0.5)
    ctx = {
        "s0": 5.0,
        "s_dot0": 8.0,
        "s_ddot0": 0.0,
        "d0": 0.5,
        "d_dot0": 0.0,
        "d_ddot0": 0.0,
    }

    traj = bspline_ego_rollout(gen, jnp.asarray(ctrl_s), jnp.asarray(ctrl_d), ctx)

    if traj.shape != (gen.T, 6):
        raise AssertionError(traj.shape)
    if not np.all(np.isfinite(np.asarray(traj))):
        raise AssertionError(traj)
    if not np.allclose(np.asarray(traj[0, :2]), np.array([5.0, 0.5]), atol=1e-5):
        raise AssertionError(traj[0, :2])


def test_bspline_rollout_bounds_longitudinal_speed_from_raw_control_points():
    from config import MAX_SPEED
    from trajectory.frenet_bspline import FrenetBSplineTrajectory
    from trajectory.reference_path import StraightReference
    from trajectory.rollout import bspline_ego_rollout
    from trajectory.warmstart import tangent_control_points

    gen = FrenetBSplineTrajectory("trajectory/assets/bspline_basis.npz", StraightReference())
    ctrl_s, ctrl_d = tangent_control_points(gen, s0=0.0, s_dot0=14.0, d0=-1.75)
    raw_fast_ctrl_s = jnp.asarray(ctrl_s) * 5.0
    ctx = {
        "s0": 0.0,
        "s_dot0": 14.0,
        "s_ddot0": 0.0,
        "d0": -1.75,
        "d_dot0": 0.0,
        "d_ddot0": 0.0,
    }
    traj = bspline_ego_rollout(gen, raw_fast_ctrl_s, jnp.asarray(ctrl_d), ctx)

    if float(jnp.max(traj[:, 2])) > MAX_SPEED + 1.0:
        raise AssertionError(float(jnp.max(traj[:, 2])))


def test_signalized_bspline_runtime_advances_and_predicts_compressed_ego():
    from config import DT
    from scenario_runtime import advance_one_macro_step, prediction_trajs
    from scenarios import get_scenario
    from trajectory.frenet_bspline import FrenetBSplineTrajectory
    from trajectory.reference_path import StraightReference

    scenario = get_scenario("signalized_intersection_bspline")
    gen = FrenetBSplineTrajectory("trajectory/assets/bspline_basis.npz", StraightReference())
    decision_sequences = {
        name: np.asarray(values[0], dtype=np.float64)
        for name, values in zip(
            ("ego_ctrl_s", "ego_ctrl_d"),
            scenario.initial_component_means,
        )
    }

    next_states = advance_one_macro_step(
        scenario,
        scenario.initial_states,
        decision_sequences,
    )
    predictions = prediction_trajs(
        scenario,
        scenario.initial_states,
        decision_sequences,
    )

    if next_states.shape != scenario.initial_states.shape:
        raise AssertionError(next_states.shape)
    if not np.all(np.isfinite(next_states)):
        raise AssertionError(next_states)
    if not next_states[0, 0] > scenario.initial_states[0, 0]:
        raise AssertionError((scenario.initial_states[0], next_states[0]))

    ego_traj = predictions["ego"]
    if ego_traj.shape[1] != scenario.state_dim:
        raise AssertionError(ego_traj.shape)
    if not ego_traj.shape[0] >= 2:
        raise AssertionError(ego_traj.shape)
    if not np.all(np.isfinite(ego_traj)):
        raise AssertionError(ego_traj)
    if not np.allclose(ego_traj[0], scenario.initial_states[0], atol=1e-6):
        raise AssertionError((ego_traj[0], scenario.initial_states[0]))
    macro_idx = int(round(DT / gen.dt))
    if not np.allclose(next_states[0], ego_traj[macro_idx], atol=1e-6):
        raise AssertionError((next_states[0], ego_traj[macro_idx], gen.dt))
    if abs(float(next_states[0, 0]) - 7.0) > 1.5:
        raise AssertionError("B-spline runtime should advance one 0.5s macro step")


def test_bspline_cost_and_runtime_use_same_initial_frenet_projection():
    from costs import signalized_intersection_bspline as cost
    from trajectory.rollout import bspline_context_from_state

    state = jnp.array([10.0, -1.0, 8.0, 0.3, 0.5, 0.0], dtype=jnp.float32)
    runtime_ctx = bspline_context_from_state(state)
    cost_ctx = cost._bspline_context_from_state(
        jnp.concatenate(
            [
                state,
                jnp.asarray([13.0], dtype=jnp.float32),
                jnp.asarray([0.6, 3.3], dtype=jnp.float32),
            ]
        )
    )

    for key in ("s0", "s_dot0", "s_ddot0", "d0", "d_dot0", "d_ddot0"):
        if not np.allclose(float(runtime_ctx[key]), float(cost_ctx[key]), atol=1e-6):
            raise AssertionError((key, runtime_ctx[key], cost_ctx[key]))


if __name__ == "__main__":
    test_straight_reference_path_maps_frenet_to_cartesian()
    test_frenet_bspline_loads_basis_asset()
    test_bspline_rollout_returns_dense_ego_states()
    test_bspline_rollout_bounds_longitudinal_speed_from_raw_control_points()
    test_signalized_bspline_runtime_advances_and_predicts_compressed_ego()
    test_bspline_cost_and_runtime_use_same_initial_frenet_projection()

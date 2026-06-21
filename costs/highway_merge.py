"""Constraint-DSL cost profile for the highway merge scenario.

This profile keeps the original rollout and agent structure, but expresses the
priority layers through ``constraint_dsl.build``.
"""

import jax
import jax.numpy as jnp

from config import (
    DT_C,
    SAFETY_CHECK_STEPS,
    WHEEL_BASE,
    LR,
    Z_REG_THRESH,
    Z_REG_WEIGHT,
)
from decision_layout import BlockDecoder
from scenarios import get_scenario
from .common import dense_horizon_pair_collision_cost, dense_rollout_from_decisions
from .constraint_dsl import Deterministic, build


_SCENARIO = get_scenario("highway_merge")
_DECODER = BlockDecoder(_SCENARIO)
_SOLVER_SPEC = _SCENARIO.solver_spec
_SOLVER_WIDTH = max(_SOLVER_SPEC.block_dims)
_STATE_DIM = _SCENARIO.state_dim
_CTX_STATE_DIM = _SCENARIO.context_state_dim
_EGO = _SCENARIO.agents[0]
_FRONT = _SCENARIO.agents[1]
_REAR = _SCENARIO.agents[2]
_LOWER_LANE_Y = min(_SCENARIO.road.lane_centers)
_UPPER_LANE_Y = max(_SCENARIO.road.lane_centers)
_GEOM = _SCENARIO.vehicle_geometry


def _decode_joint_sample(joint_sample_flat):
    """Decode one flattened joint MGIGO sample into named control sequences."""
    blocks = joint_sample_flat.reshape((_SOLVER_SPEC.n_blocks, _SOLVER_WIDTH))
    return _DECODER.decode(blocks)


def _shared_context(joint_sample_flat, context_arr):
    """Build reusable rollout context for one candidate joint sample.

    context_arr is packed by planner.py as:
        [all agent states flattened, all velocity references]
    """
    current_states = context_arr[:_CTX_STATE_DIM].reshape(_SCENARIO.n_agents, _STATE_DIM)
    decisions = _decode_joint_sample(joint_sample_flat)
    dense_traj = dense_rollout_from_decisions(_SCENARIO, current_states, decisions)
    return {
        "decisions": decisions,
        "dense_traj": dense_traj,
        "current_states": current_states,
        "context_arr": context_arr,
    }


def _ego_objective(x, ctx):
    """Ego performance objective before constraint layers.

    Tracks target speed/lane, discourages lateral velocity and heading error,
    and regularizes both physical controls and raw solver z values.
    """
    ego_traj = ctx["dense_traj"][:, _EGO.state_index, :]
    decisions = ctx["decisions"]
    v_ref = ctx["context_arr"][_CTX_STATE_DIM + _EGO.reference_index]

    ego_v = ego_traj[:, 2]
    ego_y = ego_traj[:, 1]
    ego_psi = ego_traj[:, 3]
    ego_acc = ego_traj[:, 4]
    ego_steer = ego_traj[:, 5]

    beta = jnp.arctan(LR * jnp.tan(ego_steer) / WHEEL_BASE)
    ego_vy = ego_v * jnp.sin(ego_psi + beta)

    c_vel = jnp.sum(3.0 * (ego_v - v_ref) ** 2 * DT_C)
    c_lane = jnp.sum(10.0 * (ego_y - _SCENARIO.target_y) ** 2 * DT_C)
    c_vy = jnp.sum(5.0 * ego_vy ** 2 * DT_C)
    c_hdg = jnp.sum(10.8 * ego_psi ** 2 * DT_C)
    c_ctrl = 0.5 * jnp.sum(ego_acc ** 2) + 0.5 * jnp.sum(ego_steer ** 2)
    c_dctrl = (
        jnp.sum(jnp.diff(ego_acc) ** 2)
        + jnp.sum(jnp.diff(ego_steer) ** 2)
    )
    z_reg = Z_REG_WEIGHT * (
        jnp.sum(jax.nn.relu(jnp.abs(decisions["ego_acc"]) - Z_REG_THRESH) ** 2)
        + jnp.sum(jax.nn.relu(jnp.abs(decisions["ego_steer"]) - Z_REG_THRESH) ** 2)
    )
    return c_vel + c_lane + c_vy + c_hdg + c_ctrl + c_dctrl + z_reg


def _ego_terminal_violation(x, ctx):
    """Soft/tunable terminal merge-quality violation for ego."""
    ego_traj = ctx["dense_traj"][:, _EGO.state_index, :]
    ego_v = ego_traj[:, 2]
    ego_y = ego_traj[:, 1]
    ego_psi = ego_traj[:, 3]
    ego_steer = ego_traj[:, 5]
    beta = jnp.arctan(LR * jnp.tan(ego_steer) / WHEEL_BASE)
    ego_vy = ego_v * jnp.sin(ego_psi + beta)

    terminal = (
        50.0 * (ego_y[-1] - _SCENARIO.target_y) ** 2
        + 65.0 * ego_vy[-1] ** 2
        + 65.0 * ego_psi[-1] ** 2
    )
    asymmetry = (
        jnp.sum(jnp.where(ego_y - _SCENARIO.target_y > 0.5, 50.0, 0.0) * DT_C)
        + jnp.sum(jnp.where(ego_y - _SCENARIO.target_y < -0.5, 15.0, 0.0) * DT_C)
        + jnp.sum(jnp.where(ego_y < _LOWER_LANE_Y - 1.0, 100.0, 0.0) * DT_C)
        + jnp.sum(jnp.where(ego_y > _UPPER_LANE_Y + 1.0, 100.0, 0.0) * DT_C)
    )
    return terminal + asymmetry


def _ego_collision_violation(x, ctx):
    """Hard ego safety violation against front and rear vehicles."""
    dense_traj = ctx["dense_traj"]
    ego_traj = dense_traj[:, _EGO.state_index, :]
    front_traj = dense_traj[:, _FRONT.state_index, :]
    rear_traj = dense_traj[:, _REAR.state_index, :]
    return 100.0 * (
        dense_horizon_pair_collision_cost(
            ego_traj, front_traj, _GEOM.length, _GEOM.width, _GEOM.safe_gap
        )
        + dense_horizon_pair_collision_cost(
            ego_traj, rear_traj, _GEOM.length, _GEOM.width, _GEOM.safe_gap
        )
    )


def _ego_boundary_violation(x, ctx):
    """Hard road-boundary violation for the ego vehicle footprint."""
    ego_y = ctx["dense_traj"][:, _EGO.state_index, 1]
    half_w = 0.5 * _GEOM.width
    road_min = _SCENARIO.road.road_min_y
    road_max = _SCENARIO.road.road_max_y
    return jnp.sum(
        jax.nn.relu((road_min + half_w) - ego_y)
        + jax.nn.relu(ego_y - (road_max - half_w))
    ) * 250.0


_ego_base_cost = build(
    _ego_objective,
    [
        Deterministic(g_fn=_ego_terminal_violation, mode="tunable", priority=2),
        Deterministic(g_fn=_ego_collision_violation, mode="hard", priority=1),
        Deterministic(g_fn=_ego_boundary_violation, mode="hard", priority=1),
    ],
    k_inner=0.1,
    penalize_only_soft=True,
    jit_cost=False,
)


def ego_cost(joint_sample_flat, context_arr):
    """Registered ego cost entry called by the MGIGO fitness dispatcher."""
    return _ego_base_cost(joint_sample_flat, _shared_context(joint_sample_flat, context_arr))


def _front_objective(x, ctx):
    """Front vehicle objective: keep reference speed and smooth acceleration."""
    front_traj = ctx["dense_traj"][:, _FRONT.state_index, :]
    decisions = ctx["decisions"]
    v_ref = ctx["context_arr"][_CTX_STATE_DIM + _FRONT.reference_index]
    c_vel = jnp.sum(3.0 * (front_traj[:, 2] - v_ref) ** 2 * DT_C)
    front_acc = front_traj[:, 4]
    c_ctrl = 2.0 * jnp.sum(front_acc ** 2) + 2.0 * jnp.sum(jnp.diff(front_acc) ** 2)
    z_reg = Z_REG_WEIGHT * jnp.sum(
        jax.nn.relu(jnp.abs(decisions["front_acc"]) - Z_REG_THRESH) ** 2
    )
    return c_vel + c_ctrl + z_reg


def _front_terminal_violation(x, ctx):
    """Soft/tunable terminal speed tracking violation for the front vehicle."""
    front_traj = ctx["dense_traj"][:, _FRONT.state_index, :]
    v_ref = ctx["context_arr"][_CTX_STATE_DIM + _FRONT.reference_index]
    return 100.0 * (front_traj[-1, 2] - v_ref) ** 2


def _front_collision_violation(x, ctx):
    """Hard short-horizon collision violation between front vehicle and ego."""
    front_traj = ctx["dense_traj"][:, _FRONT.state_index, :]
    ego_traj = ctx["dense_traj"][:, _EGO.state_index, :]
    return 100.0 * dense_horizon_pair_collision_cost(
        front_traj[:SAFETY_CHECK_STEPS],
        ego_traj[:SAFETY_CHECK_STEPS],
        _GEOM.length,
        _GEOM.width,
        _GEOM.safe_gap,
    )


_front_base_cost = build(
    _front_objective,
    [
        Deterministic(g_fn=_front_terminal_violation, mode="tunable", priority=2),
        Deterministic(g_fn=_front_collision_violation, mode="hard", priority=1),
    ],
    k_inner=0.1,
    penalize_only_soft=True,
    jit_cost=False,
)


def front_cost(joint_sample_flat, context_arr):
    """Registered front-vehicle cost entry called by MGIGO."""
    return _front_base_cost(joint_sample_flat, _shared_context(joint_sample_flat, context_arr))


def _rear_objective(x, ctx):
    """Rear vehicle objective: keep reference speed and smooth acceleration."""
    rear_traj = ctx["dense_traj"][:, _REAR.state_index, :]
    decisions = ctx["decisions"]
    v_ref = ctx["context_arr"][_CTX_STATE_DIM + _REAR.reference_index]
    c_vel = jnp.sum(3.0 * (rear_traj[:, 2] - v_ref) ** 2 * DT_C)
    rear_acc = rear_traj[:, 4]
    c_ctrl = 2.0 * jnp.sum(rear_acc ** 2) + 2.0 * jnp.sum(jnp.diff(rear_acc) ** 2)
    z_reg = Z_REG_WEIGHT * jnp.sum(
        jax.nn.relu(jnp.abs(decisions["rear_acc"]) - Z_REG_THRESH) ** 2
    )
    return c_vel + c_ctrl + z_reg


def _rear_terminal_violation(x, ctx):
    """Soft/tunable terminal speed tracking violation for the rear vehicle."""
    rear_traj = ctx["dense_traj"][:, _REAR.state_index, :]
    v_ref = ctx["context_arr"][_CTX_STATE_DIM + _REAR.reference_index]
    return 100.0 * (rear_traj[-1, 2] - v_ref) ** 2


def _rear_collision_violation(x, ctx):
    """Hard short-horizon collision violation between rear vehicle and ego."""
    rear_traj = ctx["dense_traj"][:, _REAR.state_index, :]
    ego_traj = ctx["dense_traj"][:, _EGO.state_index, :]
    return 100.0 * dense_horizon_pair_collision_cost(
        rear_traj[:SAFETY_CHECK_STEPS],
        ego_traj[:SAFETY_CHECK_STEPS],
        _GEOM.length,
        _GEOM.width,
        _GEOM.safe_gap,
    )


def _rear_headway_violation(x, ctx):
    """Hard headway violation keeping rear behind front with safe clearance."""
    rear_traj = ctx["dense_traj"][:, _REAR.state_index, :]
    front_traj = ctx["dense_traj"][:, _FRONT.state_index, :]
    dx_longitudinal = front_traj[:, 0] - rear_traj[:, 0]
    min_clearance = _GEOM.length + _GEOM.safe_gap
    headway_violation = jax.nn.relu(min_clearance - dx_longitudinal)
    overtake_violation = jax.nn.relu(-dx_longitudinal)
    return jnp.sum(
        (20.0 * headway_violation ** 2 + 500.0 * overtake_violation ** 2) * DT_C
    )


_rear_base_cost = build(
    _rear_objective,
    [
        Deterministic(g_fn=_rear_terminal_violation, mode="tunable", priority=2),
        Deterministic(g_fn=_rear_collision_violation, mode="hard", priority=1),
        Deterministic(g_fn=_rear_headway_violation, mode="hard", priority=1),
    ],
    k_inner=0.1,
    penalize_only_soft=True,
    jit_cost=False,
)


def rear_cost(joint_sample_flat, context_arr):
    """Registered rear-vehicle cost entry called by MGIGO."""
    return _rear_base_cost(joint_sample_flat, _shared_context(joint_sample_flat, context_arr))

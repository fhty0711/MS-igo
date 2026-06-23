"""Baseline hand-written cost profile for the highway merge scenario.

This module keeps the original hierarchical cost design available inside the
new scenario/cost registry, so it can be compared against the wrapper /
constraint-DSL profile without restoring the old fixed highway runner.
"""

import jax
import jax.numpy as jnp

from config import (
    DT_C,
    WHEEL_BASE, LR,
    SAFETY_CHECK_STEPS, Z_REG_WEIGHT, Z_REG_THRESH,
)
from decision_layout import BlockDecoder
from scenarios import get_scenario
from .common import (
    dense_horizon_pair_collision_cost,
    dense_rollout_from_decisions,
    hierarchical_cost,
)


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
    blocks = joint_sample_flat.reshape(
        (_SOLVER_SPEC.n_blocks, _SOLVER_WIDTH)
    )
    return _DECODER.decode(blocks)


def ego_cost(joint_sample_flat, context_arr):
    current_states = context_arr[:_CTX_STATE_DIM].reshape(_SCENARIO.n_agents, _STATE_DIM)
    v_ref = context_arr[_CTX_STATE_DIM + _EGO.reference_index]
    blocks = _decode_joint_sample(joint_sample_flat)

    dense_traj = dense_rollout_from_decisions(_SCENARIO, current_states, blocks)

    ego_traj = dense_traj[:, _EGO.state_index, :]
    front_traj = dense_traj[:, _FRONT.state_index, :]
    rear_traj = dense_traj[:, _REAR.state_index, :]

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
    c_state = c_vel + c_lane + c_vy + c_hdg

    c_ctrl = 0.5 * jnp.sum(ego_acc ** 2) + 0.5 * jnp.sum(ego_steer ** 2)
    c_dctrl = (
        1.0 * jnp.sum(jnp.diff(ego_acc) ** 2)
        + 1.0 * jnp.sum(jnp.diff(ego_steer) ** 2)
    )
    z_reg = Z_REG_WEIGHT * (
        jnp.sum(jax.nn.relu(jnp.abs(blocks["ego_acc"]) - Z_REG_THRESH) ** 2)
        + jnp.sum(jax.nn.relu(jnp.abs(blocks["ego_steer"]) - Z_REG_THRESH) ** 2)
    )
    f_perf = c_state + c_ctrl + c_dctrl + z_reg

    c_terminal = (
        50.0 * (ego_y[-1] - _SCENARIO.target_y) ** 2
        + 65.0 * ego_vy[-1] ** 2
        + 65.0 * ego_psi[-1] ** 2
    )
    c_asymmetry = (
        jnp.sum(jnp.where(ego_y - _SCENARIO.target_y > 0.5, 50.0, 0.0) * DT_C)
        + jnp.sum(jnp.where(ego_y - _SCENARIO.target_y < -0.5, 15.0, 0.0) * DT_C)
        + jnp.sum(jnp.where(ego_y < _LOWER_LANE_Y - 1.0, 100.0, 0.0) * DT_C)
        + jnp.sum(jnp.where(ego_y > _UPPER_LANE_Y + 1.0, 100.0, 0.0) * DT_C)
    )
    phi_terminal = c_terminal + c_asymmetry

    c_coll = 100.0 * (
        dense_horizon_pair_collision_cost(
            ego_traj, front_traj, _GEOM.length, _GEOM.width, _GEOM.safe_gap
        )
        + dense_horizon_pair_collision_cost(
            ego_traj, rear_traj, _GEOM.length, _GEOM.width, _GEOM.safe_gap
        )
    )
    half_w = 0.5 * _GEOM.width
    road_min = _SCENARIO.road.road_min_y
    road_max = _SCENARIO.road.road_max_y
    c_bnd = jnp.sum(
        jax.nn.relu((road_min + half_w) - ego_y)
        + jax.nn.relu(ego_y - (road_max - half_w))
    ) * 250.0
    phi_safe = c_coll + c_bnd

    return hierarchical_cost(f_perf, phi_terminal, phi_safe)


def front_cost(joint_sample_flat, context_arr):
    current_states = context_arr[:_CTX_STATE_DIM].reshape(_SCENARIO.n_agents, _STATE_DIM)
    v_ref = context_arr[_CTX_STATE_DIM + _FRONT.reference_index]
    blocks = _decode_joint_sample(joint_sample_flat)

    dense_traj = dense_rollout_from_decisions(_SCENARIO, current_states, blocks)

    front_traj = dense_traj[:, _FRONT.state_index, :]
    ego_traj = dense_traj[:, _EGO.state_index, :]

    c_vel = jnp.sum(3.0 * (front_traj[:, 2] - v_ref) ** 2 * DT_C)
    front_acc = front_traj[:, 4]
    c_ctrl = 2.0 * jnp.sum(front_acc ** 2) + 2.0 * jnp.sum(jnp.diff(front_acc) ** 2)
    z_reg = Z_REG_WEIGHT * jnp.sum(
        jax.nn.relu(jnp.abs(blocks["front_acc"]) - Z_REG_THRESH) ** 2
    )
    f_perf = c_vel + c_ctrl + z_reg

    phi_terminal = 100.0 * (front_traj[-1, 2] - v_ref) ** 2

    phi_safe = 100.0 * dense_horizon_pair_collision_cost(
        front_traj[:SAFETY_CHECK_STEPS],
        ego_traj[:SAFETY_CHECK_STEPS],
        _GEOM.length,
        _GEOM.width,
        _GEOM.safe_gap,
    )

    return hierarchical_cost(f_perf, phi_terminal, phi_safe)


def rear_cost(joint_sample_flat, context_arr):
    current_states = context_arr[:_CTX_STATE_DIM].reshape(_SCENARIO.n_agents, _STATE_DIM)
    v_ref = context_arr[_CTX_STATE_DIM + _REAR.reference_index]
    blocks = _decode_joint_sample(joint_sample_flat)

    dense_traj = dense_rollout_from_decisions(_SCENARIO, current_states, blocks)

    rear_traj = dense_traj[:, _REAR.state_index, :]
    ego_traj = dense_traj[:, _EGO.state_index, :]
    front_traj = dense_traj[:, _FRONT.state_index, :]

    c_vel = jnp.sum(3.0 * (rear_traj[:, 2] - v_ref) ** 2 * DT_C)
    rear_acc = rear_traj[:, 4]
    c_ctrl = 2.0 * jnp.sum(rear_acc ** 2) + 2.0 * jnp.sum(jnp.diff(rear_acc) ** 2)
    z_reg = Z_REG_WEIGHT * jnp.sum(
        jax.nn.relu(jnp.abs(blocks["rear_acc"]) - Z_REG_THRESH) ** 2
    )
    f_perf = c_vel + c_ctrl + z_reg

    phi_terminal = 100.0 * (rear_traj[-1, 2] - v_ref) ** 2

    c_coll_ego = dense_horizon_pair_collision_cost(
        rear_traj[:SAFETY_CHECK_STEPS],
        ego_traj[:SAFETY_CHECK_STEPS],
        _GEOM.length,
        _GEOM.width,
        _GEOM.safe_gap,
    )
    dx_longitudinal = front_traj[:, 0] - rear_traj[:, 0]
    safety_clearance = _GEOM.length
    dist_to_hazard = jnp.maximum(dx_longitudinal - safety_clearance, 0.1)
    c_headway = jnp.sum(20.0 / dist_to_hazard * DT_C)
    c_overtake = jnp.sum(jnp.where(dx_longitudinal <= 0.0, 500.0, 0.0) * DT_C)

    phi_safe = 100.0 * c_coll_ego + c_headway + c_overtake

    return hierarchical_cost(f_perf, phi_terminal, phi_safe)

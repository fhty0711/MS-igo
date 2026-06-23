"""Baseline hand-written cost profile for borrow-lane overtaking.

Safety is expressed with the same STL robustness semantics as the wrapper cost:

- G[0, T] no footprint overlap with other vehicles.
- G[0, T] ego vehicle stays inside the road boundaries.

Task performance remains conventional: speed tracking, smooth controls,
progress past the slow lead vehicle, and returning to the ego lane. The cost is
combined with the original hierarchical saturation rule instead of the
constraint-DSL wrapper, so this profile can be compared against
``borrow_overtake``.
"""

import jax
import jax.numpy as jnp

from config import (
    DT_C,
    LR,
    SAT_SCALE,
    SAFETY_CHECK_STEPS,
    WHEEL_BASE,
    Z_REG_THRESH,
    Z_REG_WEIGHT,
)
from decision_layout import BlockDecoder
from scenarios import get_scenario
from .common import dense_rollout_from_decisions, hierarchical_cost
from .stl import always, predicate, violation


_SCENARIO = get_scenario("borrow_overtake_critical")
_DECODER = BlockDecoder(_SCENARIO)
_SOLVER_SPEC = _SCENARIO.solver_spec
_SOLVER_WIDTH = max(_SOLVER_SPEC.block_dims)
_STATE_DIM = _SCENARIO.state_dim
_CTX_STATE_DIM = _SCENARIO.context_state_dim
_EGO = _SCENARIO.agents[0]
_SLOW = _SCENARIO.agents[1]
_ONCOMING = _SCENARIO.agents[2]
_GEOM = _SCENARIO.vehicle_geometry

_EGO_LANE_Y = min(_SCENARIO.road.lane_centers)
_ONCOMING_LANE_Y = max(_SCENARIO.road.lane_centers)
_ROAD_MIN_Y = _SCENARIO.road.road_min_y
_ROAD_MAX_Y = _SCENARIO.road.road_max_y
_PASS_CLEARANCE = 20.0
_ONCOMING_TIME_MARGIN = 1.5
_ONCOMING_DISTANCE_MARGIN = 28.0
_LANE_MID_Y = 0.5 * (_EGO_LANE_Y + _ONCOMING_LANE_Y)
_HARD_EXACT_DELTA = 1.25 * SAT_SCALE
_FOOTPRINT_HARD_GAP = 0.25
_FOLLOWING_GAP = 14.0
_PASS_ACTIVATION_DISTANCE = 300.0
_PASS_ACTIVATION_WIDTH = 80.0
_PASS_REQUIRED_DISTANCE = 380.0
_PASS_REQUIRED_WIDTH = 70.0
_PASS_EVENT_CLEARANCE = 24.0
_PASS_PROGRESS_PER_HORIZON = 14.0
_MIN_BORROW_LANE_SPEED = 8.0
_CENTERLINE_HARD_GAP = 0.10


def _decode_joint_sample(joint_sample_flat):
    """Decode one flattened MGIGO sample into named control sequences."""
    blocks = joint_sample_flat.reshape((_SOLVER_SPEC.n_blocks, _SOLVER_WIDTH))
    return _DECODER.decode(blocks)


def _shared_context(joint_sample_flat, context_arr):
    """Build reusable rollout context for one candidate joint sample."""
    current_states = context_arr[:_CTX_STATE_DIM].reshape(_SCENARIO.n_agents, _STATE_DIM)
    decisions = _decode_joint_sample(joint_sample_flat)
    dense_traj = dense_rollout_from_decisions(_SCENARIO, current_states, decisions)
    return {
        "decisions": decisions,
        "dense_traj": dense_traj,
        "current_states": current_states,
        "context_arr": context_arr,
    }


def _stl_always_violation(robustness):
    """Return the signed violation of ``G[0,T] rho >= 0``."""
    phi = always(predicate("legacy_trace_predicate", lambda _x, _ctx: robustness))
    return violation(phi, None, None)


def _hard_exact_penalty(g_raw):
    """Exact hard penalty layer for the hand-written baseline.

    ``hierarchical_cost`` saturates all inner content to approximately
    ``[-SAT_SCALE, SAT_SCALE]``. The delta jump is therefore set larger than
    that bound so any hard violation outranks all lower-priority feasible
    objective values, matching the paper's exact-penalty separation idea.
    """
    return jnp.where(
        g_raw > 0.0,
        _HARD_EXACT_DELTA + jnp.log1p(g_raw),
        0.0,
    )


def _pair_clearance_robustness(traj_a, traj_b, length, width, safe_gap):
    """STL robustness proxy for pairwise rectangular/elliptic clearance.

    Positive values mean the pair is outside the safety ellipse at each time.
    Negative values mean overlap. This mirrors pairwise_footprint_overlap_cost
    but returns a signed robustness trace for G[0,T] reasoning.
    """
    eff_len = length + safe_gap
    eff_wid = width

    def robustness_in_body_frame(src, dst):
        dx = src[:2] - dst[:2]
        c = jnp.cos(src[3])
        s = jnp.sin(src[3])
        rx = dx[0] * c + dx[1] * s
        ry = -dx[0] * s + dx[1] * c
        return (rx / eff_len) ** 2 + (ry / eff_wid) ** 2 - 1.0

    rho_a = jax.vmap(robustness_in_body_frame)(traj_a, traj_b)
    rho_b = jax.vmap(robustness_in_body_frame)(traj_b, traj_a)
    return jnp.minimum(rho_a, rho_b)


def _ego_objective(x, ctx):
    """Conventional performance terms for passing and returning."""
    ego_traj = ctx["dense_traj"][:, _EGO.state_index, :]
    slow_traj = ctx["dense_traj"][:, _SLOW.state_index, :]
    decisions = ctx["decisions"]
    v_ref = ctx["context_arr"][_CTX_STATE_DIM + _EGO.reference_index]

    ego_v = ego_traj[:, 2]
    ego_y = ego_traj[:, 1]
    ego_psi = ego_traj[:, 3]
    ego_acc = ego_traj[:, 4]
    ego_steer = ego_traj[:, 5]

    beta = jnp.arctan(LR * jnp.tan(ego_steer) / WHEEL_BASE)
    ego_vy = ego_v * jnp.sin(ego_psi + beta)

    relative_lead = ego_traj[:, 0] - slow_traj[:, 0]
    pass_progress = -10.0 * jnp.tanh(jnp.max(relative_lead - _GEOM.length) / 30.0)

    c_vel = jnp.sum(2.0 * (ego_v - v_ref) ** 2 * DT_C)
    # Pass completion and lane return are handled by separate STL task layers.
    # Keeping unconditional terminal penalties here makes safe overtaking and
    # blocked following fight inside the low-priority objective.
    c_lane_return = 0.0
    c_terminal_pass = 0.0
    c_vy = jnp.sum(3.5 * ego_vy ** 2 * DT_C)
    c_hdg = jnp.sum(7.0 * ego_psi ** 2 * DT_C)
    c_ctrl = 0.45 * jnp.sum(ego_acc ** 2) + 0.45 * jnp.sum(ego_steer ** 2)
    c_dctrl = jnp.sum(jnp.diff(ego_acc) ** 2) + jnp.sum(jnp.diff(ego_steer) ** 2)
    z_reg = Z_REG_WEIGHT * (
        jnp.sum(jax.nn.relu(jnp.abs(decisions["ego_acc"]) - Z_REG_THRESH) ** 2)
        + jnp.sum(jax.nn.relu(jnp.abs(decisions["ego_steer"]) - Z_REG_THRESH) ** 2)
    )
    return (
        c_vel
        + c_lane_return
        + c_terminal_pass
        + c_vy
        + c_hdg
        + c_ctrl
        + c_dctrl
        + z_reg
        + pass_progress
    )


def _ego_stl_footprint_collision_violation(x, ctx):
    """Highest priority STL: G[0,T] ego must avoid physical footprint overlap."""
    dense_traj = ctx["dense_traj"]
    ego_traj = dense_traj[:, _EGO.state_index, :]
    slow_traj = dense_traj[:, _SLOW.state_index, :]
    oncoming_traj = dense_traj[:, _ONCOMING.state_index, :]

    rho_slow = _pair_clearance_robustness(
        ego_traj, slow_traj, _GEOM.length, _GEOM.width, _FOOTPRINT_HARD_GAP
    )
    rho_oncoming = _pair_clearance_robustness(
        ego_traj, oncoming_traj, _GEOM.length, _GEOM.width, _FOOTPRINT_HARD_GAP
    )
    return jnp.maximum(
        _stl_always_violation(rho_slow),
        _stl_always_violation(rho_oncoming),
    )


def _ego_stl_road_boundary_violation(x, ctx):
    """STL: G[0,T] ego footprint should stay inside road boundaries."""
    ego_y = ctx["dense_traj"][:, _EGO.state_index, 1]
    half_w = 0.5 * _GEOM.width
    rho_lower = ego_y - (_ROAD_MIN_Y + half_w)
    rho_upper = (_ROAD_MAX_Y - half_w) - ego_y
    return jnp.maximum(
        _stl_always_violation(rho_lower),
        _stl_always_violation(rho_upper),
    )


def _ego_oncoming_gap_violation(x, ctx):
    """Second priority STL: borrowed-lane oncoming TTC/distance margin."""
    dense_traj = ctx["dense_traj"]
    ego_traj = dense_traj[:, _EGO.state_index, :]
    oncoming_traj = dense_traj[:, _ONCOMING.state_index, :]

    in_oncoming_lane = jnp.clip(
        (ego_traj[:, 1] - _LANE_MID_Y) / (0.5 * (_ONCOMING_LANE_Y - _EGO_LANE_Y)),
        0.0,
        1.0,
    )
    closing_gap = oncoming_traj[:, 0] - ego_traj[:, 0]
    closing_speed = jnp.maximum(ego_traj[:, 2] + oncoming_traj[:, 2], 1.0)
    ttc = closing_gap / closing_speed

    normalized_distance_violation = (_ONCOMING_DISTANCE_MARGIN - closing_gap) / _ONCOMING_DISTANCE_MARGIN
    normalized_ttc_violation = (_ONCOMING_TIME_MARGIN - ttc) / _ONCOMING_TIME_MARGIN
    crossed_violation = -closing_gap / _ONCOMING_DISTANCE_MARGIN
    return jnp.max(
        in_oncoming_lane
        * jnp.maximum(
            jnp.maximum(normalized_distance_violation, normalized_ttc_violation),
            crossed_violation,
        )
    )


def _ego_oncoming_lane_occupancy_violation(x, ctx):
    """Hard gate: do not occupy the oncoming lane when the available gap is low."""
    current_states = ctx["current_states"]
    ego_traj = ctx["dense_traj"][:, _EGO.state_index, :]
    current_oncoming_gap = current_states[_ONCOMING.state_index, 0] - current_states[_EGO.state_index, 0]
    danger_activation = jnp.clip(
        (_PASS_ACTIVATION_DISTANCE - current_oncoming_gap) / _PASS_ACTIVATION_WIDTH,
        0.0,
        1.0,
    )
    in_oncoming_lane = jnp.clip(
        (ego_traj[:, 1] - _LANE_MID_Y) / (0.5 * (_ONCOMING_LANE_Y - _EGO_LANE_Y)),
        0.0,
        1.0,
    )
    return danger_activation * jnp.max(in_oncoming_lane)


def _ego_centerline_clearance_violation(x, ctx):
    """Hard gate: keep the ego footprint off the centerline when blocked."""
    current_states = ctx["current_states"]
    ego_traj = ctx["dense_traj"][:, _EGO.state_index, :]
    slow_traj = ctx["dense_traj"][:, _SLOW.state_index, :]
    ego_y = ego_traj[:, 1]
    current_oncoming_gap = current_states[_ONCOMING.state_index, 0] - current_states[_EGO.state_index, 0]
    current_lead = current_states[_EGO.state_index, 0] - current_states[_SLOW.state_index, 0]
    terminal_lead = ego_traj[-1, 0] - slow_traj[-1, 0]
    pass_phase = jnp.clip(
        jnp.maximum(current_lead, terminal_lead - 0.75 * _PASS_EVENT_CLEARANCE)
        / (0.25 * _PASS_EVENT_CLEARANCE),
        0.0,
        1.0,
    )
    danger_activation = jnp.clip(
        (_PASS_ACTIVATION_DISTANCE - current_oncoming_gap) / _PASS_ACTIVATION_WIDTH,
        0.0,
        1.0,
    ) * (1.0 - pass_phase)
    footprint_crossing = (
        ego_y + 0.5 * _GEOM.width + _CENTERLINE_HARD_GAP - _LANE_MID_Y
    ) / _SCENARIO.road.lane_width
    return danger_activation * jnp.max(footprint_crossing)


def _ego_oncoming_lane_dwell_violation(x, ctx):
    """STL: while borrowing the oncoming lane, keep moving instead of waiting."""
    current_states = ctx["current_states"]
    ego_traj = ctx["dense_traj"][:, _EGO.state_index, :]
    initial_oncoming_gap = current_states[_ONCOMING.state_index, 0] - current_states[_EGO.state_index, 0]
    pass_required = jnp.clip(
        (initial_oncoming_gap - _PASS_REQUIRED_DISTANCE) / _PASS_REQUIRED_WIDTH,
        0.0,
        1.0,
    )
    in_oncoming_lane = jnp.clip(
        (ego_traj[:, 1] - _LANE_MID_Y) / (0.5 * (_ONCOMING_LANE_Y - _EGO_LANE_Y)),
        0.0,
        1.0,
    )
    low_speed_violation = (_MIN_BORROW_LANE_SPEED - ego_traj[:, 2]) / _MIN_BORROW_LANE_SPEED
    return pass_required * jnp.max(in_oncoming_lane * low_speed_violation)


def _ego_following_gap_violation(x, ctx):
    """Lower priority STL: keep a reasonable same-lane gap behind slow lead."""
    dense_traj = ctx["dense_traj"]
    ego_traj = dense_traj[:, _EGO.state_index, :]
    slow_traj = dense_traj[:, _SLOW.state_index, :]
    longitudinal_gap = slow_traj[:, 0] - ego_traj[:, 0] - _GEOM.length
    same_lane_factor = jnp.clip(
        1.0 - jnp.abs(ego_traj[:, 1] - slow_traj[:, 1]) / _SCENARIO.road.lane_width,
        0.0,
        1.0,
    )
    behind_factor = jnp.where(slow_traj[:, 0] >= ego_traj[:, 0], 1.0, 0.0)
    normalized_gap_violation = (_FOLLOWING_GAP - longitudinal_gap) / _FOLLOWING_GAP
    return jnp.max(same_lane_factor * behind_factor * normalized_gap_violation)


def _ego_completion_violation(x, ctx):
    """Task STL: if safe enough, make finite-horizon passing progress."""
    current_states = ctx["current_states"]
    ego_traj = ctx["dense_traj"][:, _EGO.state_index, :]
    slow_traj = ctx["dense_traj"][:, _SLOW.state_index, :]
    initial_oncoming_gap = current_states[_ONCOMING.state_index, 0] - current_states[_EGO.state_index, 0]
    initial_lead = current_states[_EGO.state_index, 0] - current_states[_SLOW.state_index, 0]
    gap_based_required = jnp.clip(
        (initial_oncoming_gap - _PASS_REQUIRED_DISTANCE) / _PASS_REQUIRED_WIDTH,
        0.0,
        1.0,
    )
    maneuver_started = jnp.clip(
        (current_states[_EGO.state_index, 1] - _LANE_MID_Y) / (0.5 * _SCENARIO.road.lane_width),
        0.0,
        1.0,
    )
    pass_required = jnp.maximum(gap_based_required, maneuver_started)

    terminal_lead = ego_traj[-1, 0] - slow_traj[-1, 0]
    rolling_required_lead = jnp.minimum(
        _PASS_EVENT_CLEARANCE,
        initial_lead + _PASS_PROGRESS_PER_HORIZON,
    )
    required_lead = (1.0 - maneuver_started) * rolling_required_lead + maneuver_started * _PASS_EVENT_CLEARANCE
    progress_violation = (required_lead - terminal_lead) / _PASS_PROGRESS_PER_HORIZON
    return pass_required * progress_violation


def _ego_return_lane_violation(x, ctx):
    """Task STL: once the pass is nearly complete, return to the ego lane."""
    current_states = ctx["current_states"]
    ego_traj = ctx["dense_traj"][:, _EGO.state_index, :]
    slow_traj = ctx["dense_traj"][:, _SLOW.state_index, :]
    current_lead = current_states[_EGO.state_index, 0] - current_states[_SLOW.state_index, 0]
    terminal_lead = ego_traj[-1, 0] - slow_traj[-1, 0]
    current_activation = jnp.clip(current_lead / _PASS_EVENT_CLEARANCE, 0.0, 1.0)
    predicted_complete_activation = jnp.clip(
        (terminal_lead - _PASS_EVENT_CLEARANCE) / (0.25 * _PASS_EVENT_CLEARANCE),
        0.0,
        1.0,
    )
    return_activation = jnp.maximum(current_activation, predicted_complete_activation)
    terminal = ego_traj[-1]
    current_y_error = jnp.abs(current_states[_EGO.state_index, 1] - _EGO_LANE_Y)
    terminal_y_error = jnp.abs(terminal[1] - _EGO_LANE_Y)
    required_terminal_error = jnp.maximum(0.12 * _SCENARIO.road.lane_width, current_y_error - 1.0)
    return_violation = (terminal_y_error - required_terminal_error) / _SCENARIO.road.lane_width
    heading_violation = jnp.abs(ego_traj[-1, 3]) / 0.15 - 1.0
    return return_activation * jnp.maximum(return_violation, heading_violation)


def ego_cost(joint_sample_flat, context_arr):
    """Registered ego baseline cost for borrow-lane overtaking."""
    ctx = _shared_context(joint_sample_flat, context_arr)
    f_perf = _ego_objective(joint_sample_flat, ctx)
    phi_stl_collision = _ego_stl_footprint_collision_violation(joint_sample_flat, ctx)
    phi_stl_boundary = _ego_stl_road_boundary_violation(joint_sample_flat, ctx)
    phi_oncoming_gap = _ego_oncoming_gap_violation(joint_sample_flat, ctx)
    phi_oncoming_occupancy = _ego_oncoming_lane_occupancy_violation(joint_sample_flat, ctx)
    phi_centerline_clearance = _ego_centerline_clearance_violation(joint_sample_flat, ctx)
    phi_oncoming_dwell = _ego_oncoming_lane_dwell_violation(joint_sample_flat, ctx)
    phi_following_gap = _ego_following_gap_violation(joint_sample_flat, ctx)
    phi_completion = _ego_completion_violation(joint_sample_flat, ctx)
    phi_return_lane = _ego_return_lane_violation(joint_sample_flat, ctx)
    return hierarchical_cost(
        f_perf,
        jax.nn.relu(phi_following_gap),
        _hard_exact_penalty(phi_completion),
        _hard_exact_penalty(phi_return_lane),
        _hard_exact_penalty(phi_oncoming_dwell),
        _hard_exact_penalty(phi_centerline_clearance),
        _hard_exact_penalty(phi_oncoming_occupancy),
        _hard_exact_penalty(phi_oncoming_gap),
        _hard_exact_penalty(phi_stl_boundary),
        _hard_exact_penalty(phi_stl_collision),
    )


def _longitudinal_objective(agent, decision_name, x, ctx):
    """Reference-speed and smooth-acceleration objective for non-ego vehicles."""
    traj = ctx["dense_traj"][:, agent.state_index, :]
    decisions = ctx["decisions"]
    v_ref = ctx["context_arr"][_CTX_STATE_DIM + agent.reference_index]
    acc = traj[:, 4]
    c_vel = jnp.sum(3.0 * (traj[:, 2] - v_ref) ** 2 * DT_C)
    c_ctrl = 2.0 * jnp.sum(acc ** 2) + 2.0 * jnp.sum(jnp.diff(acc) ** 2)
    z_reg = Z_REG_WEIGHT * jnp.sum(
        jax.nn.relu(jnp.abs(decisions[decision_name]) - Z_REG_THRESH) ** 2
    )
    return c_vel + c_ctrl + z_reg


def _slow_objective(x, ctx):
    return _longitudinal_objective(_SLOW, "slow_lead_acc", x, ctx)


def _oncoming_objective(x, ctx):
    return _longitudinal_objective(_ONCOMING, "oncoming_acc", x, ctx)


def _slow_collision_violation(x, ctx):
    """STL: short-horizon slow lead vehicle should not overlap ego."""
    slow_traj = ctx["dense_traj"][:SAFETY_CHECK_STEPS, _SLOW.state_index, :]
    ego_traj = ctx["dense_traj"][:SAFETY_CHECK_STEPS, _EGO.state_index, :]
    rho = _pair_clearance_robustness(
        slow_traj, ego_traj, _GEOM.length, _GEOM.width, _FOOTPRINT_HARD_GAP
    )
    return _stl_always_violation(rho)


def _oncoming_collision_violation(x, ctx):
    """STL: short-horizon oncoming vehicle should not overlap ego."""
    oncoming_traj = ctx["dense_traj"][:SAFETY_CHECK_STEPS, _ONCOMING.state_index, :]
    ego_traj = ctx["dense_traj"][:SAFETY_CHECK_STEPS, _EGO.state_index, :]
    rho = _pair_clearance_robustness(
        oncoming_traj, ego_traj, _GEOM.length, _GEOM.width, _FOOTPRINT_HARD_GAP
    )
    return _stl_always_violation(rho)


def slow_lead_cost(joint_sample_flat, context_arr):
    """Registered slow-lead vehicle baseline cost."""
    ctx = _shared_context(joint_sample_flat, context_arr)
    return hierarchical_cost(
        _slow_objective(joint_sample_flat, ctx),
        _hard_exact_penalty(_slow_collision_violation(joint_sample_flat, ctx)),
    )


def oncoming_cost(joint_sample_flat, context_arr):
    """Registered oncoming vehicle baseline cost."""
    ctx = _shared_context(joint_sample_flat, context_arr)
    return hierarchical_cost(
        _oncoming_objective(joint_sample_flat, ctx),
        _hard_exact_penalty(_oncoming_collision_violation(joint_sample_flat, ctx)),
    )

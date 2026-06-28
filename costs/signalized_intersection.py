"""Constraint-DSL cost profile for the signalized intersection dilemma."""

import jax.numpy as jnp
import numpy as np

from config import DT_C, MAX_ACC, VEH_L, VEH_W
from decision_layout import BlockDecoder
from scenarios import get_scenario
from scenarios.signalized_intersection import (
    CROSS_LANE_X,
    INTERSECTION_ENTRY_X,
    INTERSECTION_EXIT_X,
    STOP_LINE_X,
)
from .common import dense_rollout_from_decisions
from .constraint_dsl import Chance, Deterministic, build


_SCENARIO = get_scenario("signalized_intersection")
_DECODER = BlockDecoder(_SCENARIO)
_SOLVER_SPEC = _SCENARIO.solver_spec
_SOLVER_WIDTH = max(_SOLVER_SPEC.block_dims)
_STATE_DIM = _SCENARIO.state_dim
_CTX_STATE_DIM = _SCENARIO.context_state_dim
_CTX_REF_DIM = _SCENARIO.context_ref_dim
_CTX_EXTRA_OFFSET = _CTX_STATE_DIM + _CTX_REF_DIM
_EGO = _SCENARIO.agents[0]
_GEOM = _SCENARIO.vehicle_geometry
_CROSS_OBEY_STOP_BUFFER = 0.5 * _GEOM.safe_gap

CROSS_START_Y = -18.0
YELLOW_START_S = 0.6
YELLOW_DURATION_S = 2.4
RED_START_S = YELLOW_START_S + YELLOW_DURATION_S

MODE_OBEY = 0
MODE_YELLOW_RUSH = 1
MODE_RED_RUN = 2
XI_MODE = 0
XI_ARRIVAL_SHIFT = 1
XI_SPEED_SCALE = 2
XI_LATERAL_OFFSET = 3

DEV_N_SAMPLES = 40


def _yellow_start_from_context_arr(context_arr):
    if context_arr.shape[0] > _CTX_EXTRA_OFFSET:
        return context_arr[_CTX_EXTRA_OFFSET]
    return jnp.asarray(YELLOW_START_S, dtype=jnp.float32)


def _red_start_from_context_arr(context_arr):
    red_idx = _CTX_EXTRA_OFFSET + 1
    if context_arr.shape[0] > red_idx:
        return context_arr[red_idx]
    return jnp.asarray(RED_START_S, dtype=jnp.float32)


def _elapsed_time_from_context_arr(context_arr):
    elapsed_idx = _CTX_EXTRA_OFFSET + 2
    if context_arr.shape[0] > elapsed_idx:
        return context_arr[elapsed_idx]
    return jnp.asarray(0.0, dtype=jnp.float32)


def _cross_traffic_noise(key, shape):
    """Deterministic multimodal behavior table for Chance constraints."""
    del key
    n = int(shape[0])
    idx = jnp.arange(n)
    mode = jnp.where(
        idx < int(0.60 * n),
        MODE_OBEY,
        jnp.where(idx < int(0.90 * n), MODE_YELLOW_RUSH, MODE_RED_RUN),
    )
    phase = (idx.astype(jnp.float32) + 0.5) / jnp.maximum(float(n), 1.0)
    arrival_shift = jnp.where(
        mode == MODE_OBEY,
        2.2 + 0.8 * phase,
        jnp.where(mode == MODE_YELLOW_RUSH, -0.2 + 0.7 * phase, -0.8 + 0.5 * phase),
    )
    speed_scale = jnp.where(
        mode == MODE_OBEY,
        0.8 + 0.1 * phase,
        jnp.where(mode == MODE_YELLOW_RUSH, 1.05 + 0.15 * phase, 1.15 + 0.2 * phase),
    )
    lateral_offset = (phase - 0.5) * 0.7
    return jnp.stack(
        [
            mode.astype(jnp.float32),
            arrival_shift.astype(jnp.float32),
            speed_scale.astype(jnp.float32),
            lateral_offset.astype(jnp.float32),
        ],
        axis=1,
    )


def _no_blocking_intersection_from_ego_traj(ego_traj):
    """g<=0 if ego either stays before stop line or clears intersection."""
    x = ego_traj[:, 0]
    v = ego_traj[:, 2]
    in_box = jnp.logical_and(x > INTERSECTION_ENTRY_X, x < INTERSECTION_EXIT_X)
    stopped_in_box = jnp.logical_and(in_box, v < 1.0)
    stopped_penetration = jnp.where(stopped_in_box, 1.0 + INTERSECTION_EXIT_X - x, -1.0)
    terminal_in_box = jnp.logical_and(ego_traj[-1, 0] > INTERSECTION_ENTRY_X, ego_traj[-1, 0] < INTERSECTION_EXIT_X)
    terminal_penetration = jnp.where(terminal_in_box, 1.0 + INTERSECTION_EXIT_X - ego_traj[-1, 0], -1.0)
    return jnp.maximum(jnp.max(stopped_penetration), terminal_penetration)


def _time_grid(n_steps):
    return jnp.arange(n_steps, dtype=jnp.float32) * DT_C


def _cross_traj_for_xi(xi, n_steps):
    """Roll out one exogenous cross-traffic behavior sample."""
    t = _time_grid(n_steps)
    mode = xi[XI_MODE]
    arrival_shift = xi[XI_ARRIVAL_SHIFT]
    speed_scale = xi[XI_SPEED_SCALE]
    lateral_offset = xi[XI_LATERAL_OFFSET]
    base_speed = 7.5 * speed_scale
    y = CROSS_START_Y + base_speed * (t - arrival_shift)
    obey_stop_y = (
        _SCENARIO.road.road_min_y
        - 0.5 * _GEOM.length
        - _CROSS_OBEY_STOP_BUFFER
    )
    obey_y = jnp.minimum(y, obey_stop_y)
    y = jnp.where(mode == MODE_OBEY, obey_y, y)
    x = jnp.full_like(y, CROSS_LANE_X + lateral_offset)
    v = jnp.full_like(y, base_speed)
    psi = jnp.full_like(y, jnp.pi / 2.0)
    zeros = jnp.zeros_like(y)
    return jnp.stack([x, y, v, psi, zeros, zeros], axis=1)


def _decode_joint_sample(joint_sample_flat):
    blocks = joint_sample_flat.reshape((_SOLVER_SPEC.n_blocks, _SOLVER_WIDTH))
    return _DECODER.decode(blocks)


def _shared_context(joint_sample_flat, context_arr):
    current_states = context_arr[:_CTX_STATE_DIM].reshape(_SCENARIO.n_agents, _STATE_DIM)
    decisions = _decode_joint_sample(joint_sample_flat)
    dense_traj = dense_rollout_from_decisions(_SCENARIO, current_states, decisions)
    return {
        "decisions": decisions,
        "dense_traj": dense_traj,
        "current_states": current_states,
        "context_arr": context_arr,
    }


def _ego_traj(ctx):
    return ctx["dense_traj"][:, _EGO.state_index, :]


def _axis_aligned_pair_penetration(a_traj, b_traj):
    dx = jnp.abs(a_traj[:, 0] - b_traj[:, 0])
    dy = jnp.abs(a_traj[:, 1] - b_traj[:, 1])
    min_dx = VEH_L + _GEOM.safe_gap
    min_dy = VEH_W + 0.5 * _GEOM.safe_gap
    overlap_x = min_dx - dx
    overlap_y = min_dy - dy
    return jnp.minimum(overlap_x, overlap_y)


def classify_ego_mode(ego_traj_np):
    """Classify a displayed ego trajectory as stop, pass, or undecided."""
    ego = np.asarray(ego_traj_np, dtype=float)
    if ego.size == 0:
        return "undecided"
    x = ego[:, 0]
    v = ego[:, 2]
    stopped_before = np.any((x <= STOP_LINE_X) & (v <= 1.0))
    cleared = np.max(x) >= INTERSECTION_EXIT_X
    blocking = np.any(
        (x > INTERSECTION_ENTRY_X)
        & (x < INTERSECTION_EXIT_X)
        & (v <= 1.0)
    )
    if stopped_before and not cleared:
        return "stop"
    if cleared and not blocking:
        return "pass"
    return "undecided"


def _np_cross_traj_for_xi(xi, n_steps, *, dt=DT_C):
    """NumPy cross-traffic rollout for display metrics sampled at dt."""
    xi = np.asarray(xi, dtype=float)
    t = np.arange(n_steps, dtype=float) * float(dt)
    mode = int(xi[XI_MODE])
    arrival_shift = float(xi[XI_ARRIVAL_SHIFT])
    speed_scale = float(xi[XI_SPEED_SCALE])
    lateral_offset = float(xi[XI_LATERAL_OFFSET])
    base_speed = 7.5 * speed_scale
    y = CROSS_START_Y + base_speed * (t - arrival_shift)
    if mode == MODE_OBEY:
        obey_stop_y = (
            _SCENARIO.road.road_min_y
            - 0.5 * _GEOM.length
            - _CROSS_OBEY_STOP_BUFFER
        )
        y = np.minimum(y, obey_stop_y)
    x = np.full_like(y, CROSS_LANE_X + lateral_offset)
    v = np.full_like(y, base_speed)
    psi = np.full_like(y, np.pi / 2.0)
    zeros = np.zeros_like(y)
    return np.stack([x, y, v, psi, zeros, zeros], axis=1)


def _np_pair_clearance(a_traj, b_traj):
    a = np.asarray(a_traj, dtype=float)
    b = np.asarray(b_traj, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return float("inf")
    dx = np.abs(a[:n, 0] - b[:n, 0])
    dy = np.abs(a[:n, 1] - b[:n, 1])
    min_dx = VEH_L + _GEOM.safe_gap
    min_dy = VEH_W + 0.5 * _GEOM.safe_gap
    penetration = np.minimum(min_dx - dx, min_dy - dy)
    return float(-np.max(penetration))


def _np_red_legal(ego_traj):
    ego = np.asarray(ego_traj, dtype=float)
    if len(ego) == 0:
        return True
    t = np.arange(len(ego), dtype=float) * DT_C
    red = t >= RED_START_S
    legal = (ego[:, 0] <= STOP_LINE_X) | (ego[:, 0] >= INTERSECTION_EXIT_X)
    return bool(np.all(~red | legal))


def np_red_legal_for_history(ego_traj, *, red_start_s, dt, time_offset_s=0.0):
    """Evaluate red-light legality for a closed-loop history sampled at dt."""
    ego = np.asarray(ego_traj, dtype=float)
    if len(ego) == 0:
        return True
    t = float(time_offset_s) + np.arange(len(ego), dtype=float) * float(dt)
    red = t >= float(red_start_s)
    legal = (ego[:, 0] <= STOP_LINE_X) | (ego[:, 0] >= INTERSECTION_EXIT_X)
    return bool(np.all(~red | legal))


def _np_no_blocking(ego_traj):
    ego = np.asarray(ego_traj, dtype=float)
    if len(ego) == 0:
        return True
    x = ego[:, 0]
    v = ego[:, 2]
    in_box = (x > INTERSECTION_ENTRY_X) & (x < INTERSECTION_EXIT_X)
    stopped_in_box = in_box & (v < 1.0)
    terminal_in_box = bool(in_box[-1])
    return bool(not np.any(stopped_in_box) and not terminal_in_box)


def estimate_visual_metrics(
    ego_traj_np,
    n_samples=60,
    *,
    red_start_s=RED_START_S,
    dt=DT_C,
    time_offset_s=0.0,
):
    """Compute display-only intersection metrics from one ego trajectory."""
    ego = np.asarray(ego_traj_np, dtype=float)
    samples = np.asarray(_cross_traffic_noise(None, (n_samples,)), dtype=float)
    if len(ego) == 0:
        clearances = np.array([float("inf")])
        critical = samples[0] if len(samples) else np.zeros(4)
    else:
        clearances = np.array(
            [
                _np_pair_clearance(ego, _np_cross_traj_for_xi(xi, len(ego), dt=dt))
                for xi in samples
            ],
            dtype=float,
        )
        critical = samples[int(np.argmin(clearances))]
    return {
        "mode": classify_ego_mode(ego),
        "min_clearance": float(np.min(clearances)),
        "risk_quantile": float(np.quantile(-clearances, 0.9)),
        "red_legal": np_red_legal_for_history(
            ego,
            red_start_s=red_start_s,
            dt=dt,
            time_offset_s=time_offset_s,
        ),
        "no_blocking": _np_no_blocking(ego),
        "critical_sample": critical,
    }


def _cross_traffic_risk_violation(x, xi, ctx):
    del x
    ego = _ego_traj(ctx)
    cross = _cross_traj_for_xi(xi, ego.shape[0])
    return jnp.max(_axis_aligned_pair_penetration(ego, cross))


def _ego_road_boundary_violation(x, ctx):
    del x
    ego_y = _ego_traj(ctx)[:, 1]
    half_w = 0.5 * _GEOM.width
    road_min = _SCENARIO.road.road_min_y
    road_max = _SCENARIO.road.road_max_y
    return jnp.max(jnp.maximum((road_min + half_w) - ego_y, ego_y - (road_max - half_w)))


def _ego_red_light_violation(x, ctx):
    del x
    ego = _ego_traj(ctx)
    elapsed_time = _elapsed_time_from_context_arr(ctx["context_arr"])
    red_start = _red_start_from_context_arr(ctx["context_arr"])
    t = _time_grid(ego.shape[0]) + elapsed_time
    red = t >= red_start
    before_stop = ego[:, 0] <= STOP_LINE_X
    cleared = ego[:, 0] >= INTERSECTION_EXIT_X
    legal = jnp.logical_or(before_stop, cleared)
    illegal_red = jnp.logical_and(red, jnp.logical_not(legal))
    inside_depth = jnp.minimum(ego[:, 0] - STOP_LINE_X, INTERSECTION_EXIT_X - ego[:, 0])
    red_occupancy = jnp.where(illegal_red, 1.0 + jnp.maximum(inside_depth, 0.0), -1.0)

    # If red begins inside this prediction horizon, require the predicted state
    # at red onset to already be either stopped before the line or cleared.
    red_idx = jnp.argmin(jnp.abs(t - red_start))
    red_in_horizon = jnp.logical_and(red_start >= t[0], red_start <= t[-1])
    onset_x = ego[red_idx, 0]
    onset_v = ego[red_idx, 2]
    stop_violation = jnp.maximum(onset_x - STOP_LINE_X, onset_v - 1.0)
    clear_violation = INTERSECTION_EXIT_X - onset_x
    onset_violation = jnp.minimum(stop_violation, clear_violation)
    onset_violation = jnp.where(red_in_horizon, onset_violation, -1.0)

    cleared_before_red = jnp.any(jnp.logical_and(t <= red_start, cleared))
    stopping_distance = ego[:, 2] ** 2 / (2.0 * MAX_ACC)
    stop_feasibility = ego[:, 0] + stopping_distance - STOP_LINE_X
    stop_candidate = jnp.logical_and(t <= red_start, before_stop)
    stop_commitment = jnp.min(jnp.where(stop_candidate, stop_feasibility, 1.0))
    needs_stop_commitment = jnp.logical_and(red_in_horizon, jnp.logical_not(cleared_before_red))
    stop_commitment = jnp.where(needs_stop_commitment, stop_commitment, -1.0)
    return jnp.maximum(jnp.maximum(jnp.max(red_occupancy), onset_violation), stop_commitment)


def _no_blocking_intersection_violation(x, ctx):
    del x
    return _no_blocking_intersection_from_ego_traj(_ego_traj(ctx))


def _dilemma_task_violation(x, ctx):
    del x
    ego = _ego_traj(ctx)
    stopped_before = jnp.min(jnp.where(ego[:, 0] <= STOP_LINE_X, ego[:, 2], 1e3)) - 1.0
    cleared = INTERSECTION_EXIT_X - jnp.max(ego[:, 0])
    return jnp.minimum(stopped_before, cleared)


def _ego_objective(x, ctx):
    del x
    ego = _ego_traj(ctx)
    decisions = ctx["decisions"]
    v_ref = ctx["context_arr"][_CTX_STATE_DIM + _EGO.reference_index]
    stop_basin = (ego[-1, 0] - (STOP_LINE_X - 2.0)) ** 2 + 4.0 * ego[-1, 2] ** 2
    pass_basin = (ego[-1, 0] - (INTERSECTION_EXIT_X + 6.0)) ** 2
    mild_task = 0.05 * jnp.minimum(stop_basin, pass_basin)
    speed = 0.2 * jnp.sum((ego[:, 2] - v_ref) ** 2 * DT_C)
    lane = 4.0 * jnp.sum((ego[:, 1] - _SCENARIO.target_y) ** 2 * DT_C)
    heading = 3.0 * jnp.sum(ego[:, 3] ** 2 * DT_C)
    ctrl = 0.2 * (
        jnp.sum(ego[:, 4] ** 2)
        + jnp.sum(ego[:, 5] ** 2)
        + jnp.sum(jnp.diff(decisions["ego_acc"]) ** 2)
        + jnp.sum(jnp.diff(decisions["ego_steer"]) ** 2)
    )
    return mild_task + speed + lane + heading + ctrl


EGO_CONSTRAINT_SPECS = (
    (
        "red_light",
        Deterministic(
            g_fn=_ego_red_light_violation,
            mode="hard",
            priority=1,
            transform="sharp",
        ),
    ),
    (
        "road_boundary",
        Deterministic(
            g_fn=_ego_road_boundary_violation,
            mode="hard",
            priority=1,
            transform="sharp",
        ),
    ),
    (
        "no_blocking_intersection",
        Deterministic(
            g_fn=_no_blocking_intersection_violation,
            mode="hard",
            priority=1,
            transform="sharp",
        ),
    ),
    (
        "cross_traffic_chance",
        Chance(
            g_fn=_cross_traffic_risk_violation,
            noise_fn=_cross_traffic_noise,
            alpha=0.1,
            n_samples=DEV_N_SAMPLES,
            mode="tunable",
            priority=2,
            tune_preset="firm",
            transform="standard",
        ),
    ),
    (
        "dilemma_task",
        Deterministic(
            g_fn=_dilemma_task_violation,
            mode="tunable",
            priority=3,
            tune_preset="standard",
            transform="standard",
        ),
    ),
)


def _constraint_specs_by_name(names):
    lookup = dict(EGO_CONSTRAINT_SPECS)
    return [lookup[name] for name in names]


_ego_base_cost = build(
    _ego_objective,
    [spec for _name, spec in EGO_CONSTRAINT_SPECS],
    k_inner=0.1,
    penalize_only_soft=True,
    jit_cost=False,
    obj_transform="standard",
)


def ego_cost(joint_sample_flat, context_arr):
    return _ego_base_cost(joint_sample_flat, _shared_context(joint_sample_flat, context_arr))

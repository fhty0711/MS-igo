"""Constraint-DSL cost profile for the signalized intersection dilemma."""

import jax.numpy as jnp

from config import DT_C
from decision_layout import BlockDecoder
from scenarios import get_scenario


_SCENARIO = get_scenario("signalized_intersection")
_DECODER = BlockDecoder(_SCENARIO)
_SOLVER_SPEC = _SCENARIO.solver_spec
_SOLVER_WIDTH = max(_SOLVER_SPEC.block_dims)
_STATE_DIM = _SCENARIO.state_dim
_CTX_STATE_DIM = _SCENARIO.context_state_dim
_EGO = _SCENARIO.agents[0]
_GEOM = _SCENARIO.vehicle_geometry

STOP_LINE_X = 34.0
INTERSECTION_ENTRY_X = 36.0
INTERSECTION_EXIT_X = 50.0
CROSS_LANE_X = 42.0
CROSS_START_Y = -18.0
CROSS_CLEAR_Y = 18.0
CROSS_ROAD_HALF_WIDTH = 4.0
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
    penetration = jnp.where(stopped_in_box, 1.0 + INTERSECTION_EXIT_X - x, -1.0)
    return jnp.max(penetration)


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
    obey_stop_y = -0.5 * _GEOM.length - 1.5
    obey_y = jnp.minimum(y, obey_stop_y)
    y = jnp.where(mode == MODE_OBEY, obey_y, y)
    x = jnp.full_like(y, CROSS_LANE_X + lateral_offset)
    v = jnp.full_like(y, base_speed)
    psi = jnp.full_like(y, jnp.pi / 2.0)
    zeros = jnp.zeros_like(y)
    return jnp.stack([x, y, v, psi, zeros, zeros], axis=1)

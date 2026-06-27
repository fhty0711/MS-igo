"""Signalized-intersection ablation with one deterministic cross-traffic mode."""

import jax.numpy as jnp

from .constraint_dsl import Deterministic, build
from .signalized_intersection import (
    MODE_YELLOW_RUSH,
    _cross_traffic_risk_violation,
    _dilemma_task_violation,
    _ego_objective,
    _ego_red_light_violation,
    _ego_road_boundary_violation,
    _no_blocking_intersection_violation,
    _shared_context,
)


_YELLOW_RUSH_XI = jnp.array(
    [
        float(MODE_YELLOW_RUSH),
        0.15,
        1.12,
        0.0,
    ],
    dtype=jnp.float32,
)


def _single_mode_cross_traffic_violation(x, ctx):
    return _cross_traffic_risk_violation(x, _YELLOW_RUSH_XI, ctx)


_ego_base_cost = build(
    _ego_objective,
    [
        Deterministic(g_fn=_ego_red_light_violation, mode="hard", priority=1),
        Deterministic(g_fn=_ego_road_boundary_violation, mode="hard", priority=1),
        Deterministic(g_fn=_no_blocking_intersection_violation, mode="hard", priority=1),
        Deterministic(
            g_fn=_single_mode_cross_traffic_violation,
            mode="tunable",
            priority=2,
            delta_soft=2.0,
            beta=5.0,
        ),
        Deterministic(g_fn=_dilemma_task_violation, mode="tunable", priority=3),
    ],
    k_inner=0.1,
    penalize_only_soft=True,
    jit_cost=False,
)


def ego_cost(joint_sample_flat, context_arr):
    return _ego_base_cost(joint_sample_flat, _shared_context(joint_sample_flat, context_arr))

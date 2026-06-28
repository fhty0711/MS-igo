"""Signalized-intersection ablation with one deterministic cross-traffic mode."""

import jax.numpy as jnp

from .constraint_dsl import Deterministic, build
from .signalized_intersection import (
    MODE_YELLOW_RUSH,
    _constraint_specs_by_name,
    _cross_traffic_risk_violation,
    _ego_objective,
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
        *_constraint_specs_by_name(
            (
                "red_light",
                "road_boundary",
                "no_blocking_intersection",
            )
        ),
        Deterministic(
            g_fn=_single_mode_cross_traffic_violation,
            mode="tunable",
            priority=2,
            tune_preset="firm",
            transform="standard",
        ),
        *_constraint_specs_by_name(("dilemma_task",)),
    ],
    k_inner=0.1,
    penalize_only_soft=True,
    jit_cost=False,
    obj_transform="standard",
)


def ego_cost(joint_sample_flat, context_arr):
    return _ego_base_cost(joint_sample_flat, _shared_context(joint_sample_flat, context_arr))

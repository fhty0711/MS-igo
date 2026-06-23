"""Matched hand-written cost for borrow-lane overtaking.

This profile intentionally mirrors ``borrow_overtake``'s constraint-DSL
transformation without calling ``build``. It is used to test whether the
observed behavior comes from the mathematical transformation itself or from the
wrapper API.
"""

import jax.numpy as jnp

from . import borrow_overtake as src
from .constraint_dsl import log_transform, sigma_k


def _hard_layer(inner, g_raw, delta):
    """Manual copy of constraint_dsl hard-layer semantics."""
    return sigma_k(jnp.where(g_raw > 0.0, log_transform(g_raw) + delta, inner))


def _tunable_layer(inner, g_raw, delta_soft=2.0, beta=5.0):
    """Manual copy of constraint_dsl tunable-layer semantics."""
    t_val = jnp.maximum(0.0, log_transform(g_raw))
    return sigma_k(delta_soft * sigma_k(beta * t_val) + inner)


def _matched_wrapper_cost(objective_fn, hard_layers, tunable_layers, x, ctx):
    """Assemble layers in the same order and scale as constraint_dsl.build."""
    inner = sigma_k(log_transform(objective_fn(x, ctx)), k=0.1)

    # build() sorts by priority descending. Here we write that order explicitly:
    # priority 3 tunable first, then priority 2 hard, then priority 1 hard.
    for violation_fn in tunable_layers:
        inner = _tunable_layer(inner, violation_fn(x, ctx))

    for violation_fn in hard_layers["priority_2"]:
        inner = _hard_layer(inner, violation_fn(x, ctx), delta=3.0)

    for violation_fn in hard_layers["priority_1"]:
        inner = _hard_layer(inner, violation_fn(x, ctx), delta=1.5)

    return inner


def ego_cost(joint_sample_flat, context_arr):
    """Matched hand-written ego cost."""
    ctx = src._shared_context(joint_sample_flat, context_arr)
    return _matched_wrapper_cost(
        src._ego_objective,
        {
            "priority_2": (
                src._ego_completion_violation,
                src._ego_return_lane_violation,
                src._ego_oncoming_lane_dwell_violation,
                src._ego_centerline_clearance_violation,
                src._ego_oncoming_lane_occupancy_violation,
                src._ego_oncoming_gap_violation,
                src._ego_stl_road_boundary_violation,
            ),
            "priority_1": (
                src._ego_stl_footprint_collision_violation,
            ),
        },
        (src._ego_following_gap_violation,),
        joint_sample_flat,
        ctx,
    )


def slow_lead_cost(joint_sample_flat, context_arr):
    """Matched hand-written slow-lead cost."""
    ctx = src._shared_context(joint_sample_flat, context_arr)
    return _matched_wrapper_cost(
        src._slow_objective,
        {
            "priority_2": (),
            "priority_1": (src._slow_collision_violation,),
        },
        (),
        joint_sample_flat,
        ctx,
    )


def oncoming_cost(joint_sample_flat, context_arr):
    """Matched hand-written oncoming cost."""
    ctx = src._shared_context(joint_sample_flat, context_arr)
    return _matched_wrapper_cost(
        src._oncoming_objective,
        {
            "priority_2": (),
            "priority_1": (src._oncoming_collision_violation,),
        },
        (),
        joint_sample_flat,
        ctx,
    )

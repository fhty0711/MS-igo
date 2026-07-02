"""Rollout helpers for trajectory-parameterized scenarios."""

from __future__ import annotations


def bspline_ego_rollout(gen, ctrl_s_free, ctrl_d_free, ctx):
    """Evaluate one ego Frenet B-spline and return compressed [T, 6] states."""
    _frenet, _full_states, compressed, _xy = gen.evaluate_plan(
        ctrl_s_free,
        ctrl_d_free,
        ctx,
    )
    return compressed

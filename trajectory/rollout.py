"""Rollout helpers for trajectory-parameterized scenarios."""

from __future__ import annotations

import jax.numpy as jnp


def bspline_context_from_state(state):
    """Build Frenet initial conditions from compressed [x, y, v, psi, acc, steer]."""
    state = jnp.asarray(state)
    return {
        "s0": state[0],
        "s_dot0": state[2] * jnp.cos(state[3]),
        "s_ddot0": state[4] * jnp.cos(state[3]),
        "d0": state[1],
        "d_dot0": state[2] * jnp.sin(state[3]),
        "d_ddot0": state[4] * jnp.sin(state[3]),
    }


def bspline_ego_rollout(gen, ctrl_s_free, ctrl_d_free, ctx):
    """Evaluate one ego Frenet B-spline and return compressed [T, 6] states."""
    _frenet, _full_states, compressed, _xy = gen.evaluate_plan(
        ctrl_s_free,
        ctrl_d_free,
        ctx,
    )
    return compressed

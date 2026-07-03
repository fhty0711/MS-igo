"""B-spline cost profile for the signalized intersection benchmark."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from config import MAX_ACC, MAX_SPEED, MAX_STEER
from scenarios import get_scenario
from trajectory.frenet_bspline import FrenetBSplineTrajectory
from trajectory.reference_path import StraightReference
from trajectory.rollout import bspline_context_from_state, bspline_ego_rollout

from . import signalized_intersection as base
from .constraint_dsl import Chance, Deterministic, build


_SCENARIO = get_scenario("signalized_intersection_bspline")
_SOLVER_SPEC = _SCENARIO.solver_spec
_SOLVER_WIDTH = max(_SOLVER_SPEC.block_dims)
_STATE_DIM = _SCENARIO.state_dim
_CTX_STATE_DIM = _SCENARIO.context_state_dim
_BASIS_PATH = (
    Path(__file__).resolve().parents[1]
    / "trajectory"
    / "assets"
    / "bspline_basis.npz"
)
_GEN = FrenetBSplineTrajectory(
    _BASIS_PATH,
    StraightReference(),
)


def _decode(joint_sample_flat):
    blocks = joint_sample_flat.reshape((_SOLVER_SPEC.n_blocks, _SOLVER_WIDTH))
    return {
        "ego_ctrl_s": blocks[0, : _SOLVER_SPEC.block_dims[0]],
        "ego_ctrl_d": blocks[1, : _SOLVER_SPEC.block_dims[1]],
    }


def _bspline_context_from_state(context_arr):
    current_states = context_arr[:_CTX_STATE_DIM].reshape(_SCENARIO.n_agents, _STATE_DIM)
    return bspline_context_from_state(current_states[0])


def _shared_context(joint_sample_flat, context_arr):
    decisions = _decode(joint_sample_flat)
    ego_traj = bspline_ego_rollout(
        _GEN,
        decisions["ego_ctrl_s"],
        decisions["ego_ctrl_d"],
        _bspline_context_from_state(context_arr),
    )
    return {
        "decisions": decisions,
        "ego_traj": ego_traj,
        "context_arr": context_arr,
    }


def _ego_objective(x, ctx):
    del x
    decisions = ctx["decisions"]
    spline_smoothness = (
        jnp.sum(jnp.diff(decisions["ego_ctrl_s"], n=2) ** 2)
        + jnp.sum(jnp.diff(decisions["ego_ctrl_d"], n=2) ** 2)
    )
    return base._ego_objective_from_traj(
        ctx["ego_traj"],
        ctx["context_arr"],
        control_smoothness=0.05 * spline_smoothness,
        dt=_GEN.dt,
    )


def _cross_traffic_risk_violation(x, xi, ctx):
    del x
    return base._cross_traffic_risk_violation_from_traj(ctx["ego_traj"], xi, dt=_GEN.dt)


def _ego_road_boundary_violation(x, ctx):
    del x
    return base._ego_road_boundary_violation_from_traj(ctx["ego_traj"])


def _ego_red_light_violation(x, ctx):
    del x
    return base._ego_red_light_violation_from_traj(
        ctx["ego_traj"],
        ctx["context_arr"],
        dt=_GEN.dt,
    )


def _no_blocking_intersection_violation(x, ctx):
    del x
    return base._no_blocking_intersection_violation_from_traj(ctx["ego_traj"])


def _dilemma_task_violation(x, ctx):
    del x
    return base._dilemma_task_violation_from_traj(ctx["ego_traj"])


def _bspline_physical_feasibility_violation(x, ctx):
    del x
    return _bspline_physical_feasibility_violation_from_traj(ctx["ego_traj"])


def _bspline_physical_feasibility_violation_from_traj(ego):
    speed_violation = jnp.max(ego[:, 2] - MAX_SPEED)
    reverse_violation = jnp.max(-jnp.diff(ego[:, 0]) + 0.05)
    heading_violation = jnp.max(jnp.abs(ego[:, 3]) - 0.75)
    acc_violation = jnp.max(jnp.abs(ego[:, 4]) - MAX_ACC)
    steer_violation = jnp.max(jnp.abs(ego[:, 5]) - MAX_STEER)
    return jnp.max(
        jnp.array(
            [
                speed_violation,
                reverse_violation,
                heading_violation,
                acc_violation,
                steer_violation,
            ],
            dtype=jnp.float32,
        )
    )


EGO_CONSTRAINT_SPECS = (
    (
        "bspline_physical_feasibility",
        Deterministic(
            g_fn=_bspline_physical_feasibility_violation,
            mode="hard",
            priority=1,
            transform="sharp",
        ),
    ),
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
            noise_fn=base._cross_traffic_noise,
            alpha=0.1,
            n_samples=base.DEV_N_SAMPLES,
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


_ego_base_cost = build(
    _ego_objective,
    [spec for _name, spec in EGO_CONSTRAINT_SPECS],
    k_inner=0.1,
    penalize_only_soft=True,
    jit_cost=False,
    obj_transform="standard",
)


def ego_cost(joint_sample_flat, context_arr):
    return _ego_base_cost(
        joint_sample_flat,
        _shared_context(joint_sample_flat, context_arr),
    )

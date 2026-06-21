"""Shared cost helpers."""

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap

from config import (
    DT_C,
    MAX_ACC,
    MAX_SPEED,
    SUB_STEPS,
    TAU_ACC,
    SAT_SCALE,
    VEH_L,
    VEH_W,
    SAFE_GAP,
)
from models import pairwise_footprint_overlap_cost, stable_single_car_step


def sigma(x: jax.Array) -> jax.Array:
    """sigma(x) = x / sqrt(1 + (x / SAT_SCALE)^2)."""
    return x / jnp.sqrt(1.0 + (x / SAT_SCALE) ** 2)


def hierarchical_cost(base: jax.Array, *penalty_layers: jax.Array) -> jax.Array:
    """sigma(...sigma(sigma(base) + Phi_1) + ...) + Phi_M."""
    cost = base
    for penalty in penalty_layers:
        cost = sigma(cost) + penalty
    return cost


@jit
def dense_horizon_pair_collision_cost(
    traj_a,
    traj_b,
    length=VEH_L,
    width=VEH_W,
    safe_gap=SAFE_GAP,
):
    """Sum pairwise footprint overlap costs over a dense prediction horizon."""
    poses_a = traj_a[:, :2]
    poses_b = traj_b[:, :2]
    psis_a = traj_a[:, 3]
    psis_b = traj_b[:, 3]
    return jnp.sum(
        vmap(
            pairwise_footprint_overlap_cost,
            in_axes=(0, 0, 0, 0, None, None, None),
        )(
            poses_a,
            poses_b,
            psis_a,
            psis_b,
            length,
            width,
            safe_gap,
        )
    )


@jit
def stable_lon_heading_step(state, target_raw_acc, dt):
    """Longitudinal step along the state's current heading."""
    x, y, v, psi, curr_acc, curr_steer = (
        state[0], state[1], state[2], state[3], state[4], state[5]
    )
    t_acc = MAX_ACC * jnp.tanh(target_raw_acc)
    next_acc = curr_acc + (t_acc - curr_acc) / TAU_ACC * dt
    next_acc = jnp.clip(next_acc, -MAX_ACC, MAX_ACC)
    next_v = jnp.clip(v + next_acc * dt, 0.0, MAX_SPEED)
    next_x = x + next_v * jnp.cos(psi) * dt
    next_y = y + next_v * jnp.sin(psi) * dt
    return jnp.array([next_x, next_y, next_v, psi, next_acc, curr_steer])


def _agent_control_arrays(scenario, decision_sequences):
    controls = {}
    zeros = None
    for decision in scenario.decisions:
        seq = decision_sequences[decision.name]
        if zeros is None:
            zeros = jnp.zeros_like(seq)
        controls.setdefault(decision.agent_name, {})[decision.kind] = seq
    if zeros is None:
        zeros = jnp.zeros((scenario.control_horizon,), dtype=jnp.float32)
    return {
        agent.name: (
            controls.get(agent.name, {}).get("acc", zeros),
            controls.get(agent.name, {}).get("steer", zeros),
        )
        for agent in scenario.agents
    }


def dense_rollout_from_decisions(scenario, current_states, decision_sequences):
    """JAX dense rollout for scenario agents using decoded decision sequences."""
    controls = _agent_control_arrays(scenario, decision_sequences)
    ordered_agents = tuple(sorted(scenario.agents, key=lambda agent: agent.state_index))
    acc_dense = jnp.stack([
        jnp.repeat(jnp.asarray(controls[agent.name][0]), SUB_STEPS)
        for agent in ordered_agents
    ])
    steer_dense = jnp.stack([
        jnp.repeat(jnp.asarray(controls[agent.name][1]), SUB_STEPS)
        for agent in ordered_agents
    ])
    dynamics_codes = jnp.array([
        0 if agent.dynamics == "bicycle" else 1
        for agent in ordered_agents
    ])
    total_micro_steps = acc_dense.shape[1]

    def micro_step_fn(carry, idx):
        next_ordered = []
        for agent_pos, _agent in enumerate(ordered_agents):
            acc = acc_dense[agent_pos, idx]
            steer = steer_dense[agent_pos, idx]
            next_ordered.append(
                lax.switch(
                    dynamics_codes[agent_pos],
                    (
                        lambda s: stable_single_car_step(s, acc, steer, DT_C),
                        lambda s: stable_lon_heading_step(s, acc, DT_C),
                    ),
                    carry[agent_pos],
                )
            )
        next_states = jnp.stack(next_ordered)
        return next_states, next_states

    _, dense_trajectory = lax.scan(
        micro_step_fn,
        jnp.asarray(current_states),
        jnp.arange(total_micro_steps),
    )
    return dense_trajectory

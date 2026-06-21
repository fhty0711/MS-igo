"""Runtime helpers that execute decoded scenario decisions.

The planner returns named control sequences. This module turns those sequences
into physical state rollouts for closed-loop execution and visualization.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from config import DT_C, SUB_STEPS
from config import MAX_ACC, MAX_SPEED, TAU_ACC
from models import np_stable_single_car_step


@dataclass(frozen=True)
class AgentControlSequences:
    """Acceleration and steering sequences for one agent.

    Missing controls are filled with zero sequences. For example, a longitudinal
    vehicle usually has acc only and receives steer = 0.
    """

    acc: object
    steer: object


def controls_by_agent(scenario, decision_sequences) -> Dict[str, AgentControlSequences]:
    """Group decoded decisions into per-agent acceleration/steering sequences."""
    by_agent = {
        agent.name: {"acc": None, "steer": None}
        for agent in scenario.agents
    }
    for decision in scenario.decisions:
        if decision.name not in decision_sequences:
            raise KeyError(f"Missing decoded decision {decision.name!r}")
        by_agent[decision.agent_name][decision.kind] = decision_sequences[decision.name]

    controls = {}
    horizon = scenario.control_horizon
    zeros = np.zeros((horizon,), dtype=np.float64)
    for agent in scenario.agents:
        acc = by_agent[agent.name]["acc"]
        steer = by_agent[agent.name]["steer"]
        if acc is None:
            acc = zeros
        if steer is None:
            steer = zeros
        controls[agent.name] = AgentControlSequences(acc=acc, steer=steer)
    return controls


def _step_agent(agent, state, control, dt):
    """Advance one agent by one micro integration step."""
    if agent.dynamics == "bicycle":
        return np_stable_single_car_step(state, control.acc, control.steer, dt)
    if agent.dynamics == "longitudinal":
        return _step_longitudinal_heading(state, control.acc, dt)
    raise ValueError(f"Unsupported dynamics profile {agent.dynamics!r}")


def _step_longitudinal_heading(state, target_raw_acc, dt):
    """Advance a non-steering vehicle along its current heading."""
    x, y, v, psi, curr_acc, curr_steer = (
        float(state[0]), float(state[1]), float(state[2]),
        float(state[3]), float(state[4]), float(state[5]),
    )
    t_acc = MAX_ACC * np.tanh(float(target_raw_acc))
    next_acc = curr_acc + (t_acc - curr_acc) / TAU_ACC * dt
    next_acc = float(np.clip(next_acc, -MAX_ACC, MAX_ACC))
    next_v = v + next_acc * dt
    next_v = float(np.clip(next_v, 0.0, MAX_SPEED))
    next_x = x + next_v * np.cos(psi) * dt
    next_y = y + next_v * np.sin(psi) * dt
    return np.array(
        [next_x, next_y, next_v, psi, next_acc, curr_steer],
        dtype=np.float64,
    )


def advance_one_macro_step(scenario, current_states, decision_sequences):
    """Execute only the first MPC control over one macro step.

    The macro step is split into SUB_STEPS micro integration steps. This is the
    closed-loop state update used after each MGIGO planning call.
    """
    controls = controls_by_agent(scenario, decision_sequences)
    next_states = [state.copy() for state in current_states]

    for _ in range(SUB_STEPS):
        stepped = list(next_states)
        for agent in scenario.agents:
            ctrl = controls[agent.name]
            scalar_control = AgentControlSequences(acc=ctrl.acc[0], steer=ctrl.steer[0])
            stepped[agent.state_index] = _step_agent(
                agent, next_states[agent.state_index], scalar_control, DT_C
            )
        next_states = stepped

    return np.stack(next_states)


def prediction_trajs(scenario, current_states, decision_sequences):
    """Roll out the full prediction horizon for every agent."""
    controls = controls_by_agent(scenario, decision_sequences)
    total_micro_steps = scenario.control_horizon * SUB_STEPS
    trajectories = {}

    for agent in scenario.agents:
        traj = np.empty((total_micro_steps + 1, scenario.state_dim), dtype=np.float64)
        cur = np.asarray(current_states[agent.state_index], dtype=np.float64)
        traj[0] = cur
        ctrl = controls[agent.name]
        acc_dense = np.repeat(np.asarray(ctrl.acc), SUB_STEPS)
        steer_dense = np.repeat(np.asarray(ctrl.steer), SUB_STEPS)
        for i in range(total_micro_steps):
            cur = _step_agent(
                agent,
                cur,
                AgentControlSequences(acc=acc_dense[i], steer=steer_dense[i]),
                DT_C,
            )
            traj[i + 1] = cur
        trajectories[agent.name] = traj

    return trajectories


def dense_rollout_np(scenario, current_states, decision_sequences):
    """Return dense rollout as ``(time, agent, state_dim)`` NumPy array."""
    predictions = prediction_trajs(scenario, current_states, decision_sequences)
    ordered = [
        predictions[agent.name][1:]
        for agent in scenario.agents
    ]
    return np.stack(ordered, axis=1)

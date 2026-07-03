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
from trajectory.frenet_bspline import FrenetBSplineTrajectory
from trajectory.reference_path import StraightReference
from trajectory.rollout import bspline_context_from_state


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


_BSPLINE_GENERATOR = None


def _bspline_generator():
    global _BSPLINE_GENERATOR
    if _BSPLINE_GENERATOR is None:
        from pathlib import Path

        basis_path = Path(__file__).resolve().parent / "trajectory" / "assets" / "bspline_basis.npz"
        _BSPLINE_GENERATOR = FrenetBSplineTrajectory(basis_path, StraightReference())
    return _BSPLINE_GENERATOR


def _bspline_ctx_from_state(state):
    return bspline_context_from_state(np.asarray(state, dtype=np.float64))


def _bspline_ego_prediction(scenario, current_states, decision_sequences):
    if scenario.n_agents != 1 or scenario.agent_names != ("ego",):
        raise ValueError("frenet_bspline runtime currently supports single ego scenarios")
    for name in ("ego_ctrl_s", "ego_ctrl_d"):
        if name not in decision_sequences:
            raise KeyError(f"Missing decoded decision {name!r}")

    gen = _bspline_generator()
    _frenet, _full_states, compressed, _xy = gen.evaluate_plan(
        decision_sequences["ego_ctrl_s"],
        decision_sequences["ego_ctrl_d"],
        _bspline_ctx_from_state(current_states[0]),
    )
    ego = np.asarray(compressed, dtype=np.float64)
    ego[0] = np.asarray(current_states[0], dtype=np.float64)
    return ego


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
    if scenario.trajectory_model == "frenet_bspline":
        ego = _bspline_ego_prediction(scenario, current_states, decision_sequences)
        macro_idx = int(round((SUB_STEPS * DT_C) / _bspline_generator().dt))
        macro_idx = min(max(macro_idx, 1), ego.shape[0] - 1)
        return ego[macro_idx:macro_idx + 1]

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
    if scenario.trajectory_model == "frenet_bspline":
        ego = _bspline_ego_prediction(scenario, current_states, decision_sequences)
        return {"ego": ego}

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

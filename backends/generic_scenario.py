"""Generic scenario backend for registered MGIGO experiments."""

from dataclasses import dataclass, field

import numpy as np

from planner import plan
from viz_utils import generic_legend_handles, render_agents_panel


@dataclass
class GenericScenarioBackend:
    """Closed-loop runner state for one ScenarioSpec.

    The backend owns mutable simulation state, history buffers, and rendering
    adapters. The optimizer itself stays in planner.py.
    """

    scenario: object
    cost_profile: str
    current_states: np.ndarray = field(init=False)
    v_refs: np.ndarray = field(init=False)
    warm: object = field(default=None, init=False)
    agent_names: tuple = field(init=False)
    state_history_by_agent: dict = field(default_factory=dict, init=False)
    trajectory_history_by_agent: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Initialize mutable closed-loop state from the immutable scenario."""
        self.agent_names = self.scenario.agent_names
        self.current_states = self.scenario.initial_states.copy()
        self.v_refs = self.scenario.v_refs.copy()
        self.state_history_by_agent = {name: [] for name in self.agent_names}
        self.trajectory_history_by_agent = {name: [] for name in self.agent_names}

    def _states_by_agent(self):
        """Return current states keyed by agent name for history/rendering."""
        return {
            agent.name: self.current_states[agent.state_index].copy()
            for agent in self.scenario.agents
        }

    def describe_start(self, n_steps):
        """Human-readable run header."""
        return (
            f"开始博弈仿真（共 {n_steps} 步，agents={list(self.agent_names)}, "
            f"v_ref={self.v_refs.tolist()} m/s）"
        )

    def step(self, key, step_idx):
        """Run one MPC planning step and advance the closed-loop simulation."""
        result = plan(
            key,
            self.current_states,
            self.v_refs,
            self.warm,
            cost_profile=self.cost_profile,
            scenario=self.scenario,
            solver_spec=self.scenario.solver_spec,
        )

        states_by_agent = self._states_by_agent()
        for name, state in states_by_agent.items():
            self.state_history_by_agent[name].append(state)

        for name, trajs in result["trajectories_by_agent"].items():
            self.trajectory_history_by_agent[name].append(trajs)

        self.current_states = result["next_states"]
        self.warm = result["warm"]
        return result

    def progress_line(self, step_idx, result):
        """Compact progress line focused on the first agent."""
        ego = self.current_states[self.scenario.agents[0].state_index]
        return (
            f"  第{step_idx:2d}步  x={ego[0]:.1f}m  y={ego[1]:.2f}m  "
            f"v={ego[2]:.1f}m/s  psi={ego[3]:.3f}rad  "
            f"k={result['best_block_ks']}"
        )

    def final_summary(self):
        """Compact final-state summary focused on the first agent."""
        ego = self.current_states[self.scenario.agents[0].state_index]
        return (
            f"\n完成。主车终点：x={ego[0]:.1f}m  "
            f"y={ego[1]:.2f}m  v={ego[2]:.1f}m/s"
            f"（目标 y={self.scenario.target_y:.2f}m）"
        )

    def render_panel(self, ax, idx, title, x_win=44.0, show_step=3):
        """Render one snapshot panel from recorded state/trajectory history."""
        states_by_agent = {
            agent.name: self.state_history_by_agent[agent.name][idx]
            for agent in self.scenario.agents
        }
        trajectories_by_agent = {
            agent.name: self.trajectory_history_by_agent[agent.name][idx]
            for agent in self.scenario.agents
        }
        render_agents_panel(
            ax,
            self.scenario,
            states_by_agent=states_by_agent,
            trajectories_by_agent=trajectories_by_agent,
            history_by_agent=self.state_history_by_agent,
            focus_agent=self.scenario.agent_names[0],
            x_win=x_win,
            title=title,
            show_step=show_step,
        )

    def legend_handles(self):
        """Legend entries matching the generic agent renderer."""
        return generic_legend_handles(self.scenario)

    def animation_title(self, idx):
        """Title used for each animation frame."""
        n_steps = self.scenario.n_mpc_steps or "default"
        return f"{self.scenario.title}  [step {idx:02d}/{n_steps}]"

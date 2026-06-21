"""Shared scenario data structures.

The planner is intentionally scenario-agnostic. A scenario file describes the
experiment by filling these specs:

- agents: physical participants and their state/reference indices.
- decisions: named control sequences such as ego acceleration or steering.
- blocks: MGIGO optimization blocks, each owning one or more decisions.

From those fields, ScenarioSpec derives the solver layout used by planner.py.
"""

from dataclasses import dataclass
from math import prod
from typing import Literal, Tuple

import numpy as np


DynamicsProfile = Literal["bicycle", "longitudinal"]
ControlKind = Literal["acc", "steer"]


@dataclass(frozen=True)
class AgentSpec:
    """One physical participant in the scenario.

    state_index points to the row in the joint state array. reference_index
    points to the entry in v_refs used by the agent's cost.
    """

    name: str
    role: str
    dynamics: DynamicsProfile
    state_index: int
    reference_index: int


@dataclass(frozen=True)
class ControlBlockSpec:
    """Compatibility view for one-decision blocks."""

    name: str
    agent_name: str
    kind: ControlKind
    block_index: int


@dataclass(frozen=True)
class DecisionSpec:
    """A named optimization decision sequence decoded from an MGIGO block."""

    name: str
    agent_name: str
    kind: ControlKind
    shape: Tuple[int, ...]

    @property
    def dim(self) -> int:
        return int(prod(self.shape))


@dataclass(frozen=True)
class BlockSpec:
    """One MGIGO block and the decisions packed inside it.

    owner_agent decides which agent-level cost ranks samples for this block.
    decision_names controls how the block vector is split by BlockDecoder.
    """

    name: str
    owner_agent: str
    decision_names: Tuple[str, ...]
    block_index: int


@dataclass(frozen=True)
class SolverSpec:
    """Concrete block layout passed to the MGIGO solver."""

    m_agent: int
    n_blocks: int
    block_to_agent: Tuple[int, ...]
    block_dims: Tuple[int, ...]
    control_horizon: int


@dataclass(frozen=True)
class RoadSpec:
    """Lane geometry used by costs and visualization."""

    lane_width: float
    lane_centers: Tuple[float, ...]

    @property
    def n_lanes(self) -> int:
        return len(self.lane_centers)

    @property
    def road_min_y(self) -> float:
        return min(self.lane_centers) - 0.5 * self.lane_width

    @property
    def road_max_y(self) -> float:
        return max(self.lane_centers) + 0.5 * self.lane_width


@dataclass(frozen=True)
class VehicleGeometrySpec:
    """Vehicle footprint and collision-envelope settings."""

    length: float
    width: float
    safe_gap: float


@dataclass(frozen=True)
class ScenarioSpec:
    """Complete scenario contract consumed by runner, planner, runtime, and viz."""

    name: str
    title: str
    description: str
    output_prefix: str
    cost_profile: str
    initial_states: np.ndarray
    v_refs: np.ndarray
    target_y: float
    lane_roles: Tuple[str, ...]
    agent_roles: Tuple[str, ...]
    agents: Tuple[AgentSpec, ...]
    decisions: Tuple[DecisionSpec, ...]
    blocks: Tuple[BlockSpec, ...]
    snap_labels: Tuple[str, str, str]
    backend: str = "generic_scenario"
    state_dim: int = 6
    control_horizon: int = 12
    road: RoadSpec = RoadSpec(3.5, (0.0, 3.5))
    vehicle_geometry: VehicleGeometrySpec = VehicleGeometrySpec(5.0, 2.0, 3.0)
    notes: Tuple[str, ...] = ()

    def __post_init__(self):
        """Validate that the scenario can be decoded into a solver layout."""
        agent_names = tuple(agent.name for agent in self.agents)
        decision_names = tuple(decision.name for decision in self.decisions)
        block_names = tuple(block.name for block in self.blocks)

        if len(set(agent_names)) != len(agent_names):
            raise ValueError(f"Duplicate agent names in scenario {self.name!r}: {agent_names}")
        if len(set(decision_names)) != len(decision_names):
            raise ValueError(
                f"Duplicate decision names in scenario {self.name!r}: {decision_names}"
            )
        if len(set(block_names)) != len(block_names):
            raise ValueError(f"Duplicate block names in scenario {self.name!r}: {block_names}")

        state_indices = tuple(agent.state_index for agent in self.agents)
        expected_agent_indices = tuple(range(len(self.agents)))
        if tuple(sorted(state_indices)) != expected_agent_indices:
            raise ValueError(
                f"Scenario {self.name!r} agent state_index values must cover "
                f"{expected_agent_indices}, got {state_indices}"
            )
        reference_indices = tuple(agent.reference_index for agent in self.agents)
        if len(set(reference_indices)) != len(reference_indices):
            raise ValueError(
                f"Duplicate reference_index values in scenario {self.name!r}: "
                f"{reference_indices}"
            )
        if any(idx < 0 for idx in reference_indices):
            raise ValueError(
                f"Scenario {self.name!r} reference_index values must be nonnegative: "
                f"{reference_indices}"
            )

        block_indices = tuple(block.block_index for block in sorted(
            self.blocks, key=lambda item: item.block_index
        ))
        expected_indices = tuple(range(len(self.blocks)))
        if block_indices != expected_indices:
            raise ValueError(
                f"Scenario {self.name!r} block indices must be contiguous and zero-based: "
                f"got {block_indices}, expected {expected_indices}"
            )

        known_agents = set(agent_names)
        known_decisions = set(decision_names)
        for decision in self.decisions:
            if decision.agent_name not in known_agents:
                raise ValueError(
                    f"Decision {decision.name!r} references unknown agent "
                    f"{decision.agent_name!r}"
                )
            if decision.kind not in ("acc", "steer"):
                raise ValueError(
                    f"Decision {decision.name!r} has unsupported kind "
                    f"{decision.kind!r}"
                )
            if decision.shape != (self.control_horizon,):
                raise ValueError(
                    f"Scalar control decision {decision.name!r} shape must be "
                    f"({self.control_horizon},), got {decision.shape}"
                )
        for block in self.blocks:
            if block.owner_agent not in known_agents:
                raise ValueError(
                    f"Block {block.name!r} references unknown owner "
                    f"{block.owner_agent!r}"
                )
            for decision_name in block.decision_names:
                if decision_name not in known_decisions:
                    raise ValueError(
                        f"Block {block.name!r} references unknown decision "
                        f"{decision_name!r}"
                    )

        initial_states = np.asarray(self.initial_states)
        expected_state_shape = (len(self.agents), self.state_dim)
        if initial_states.shape != expected_state_shape:
            raise ValueError(
                f"Scenario {self.name!r} initial_states shape must be "
                f"{expected_state_shape}, got {initial_states.shape}"
            )
        if len(self.v_refs) != len(self.agents):
            raise ValueError(
                f"Scenario {self.name!r} v_refs length must match agents: "
                f"{len(self.v_refs)} != {len(self.agents)}"
            )
        if max(reference_indices, default=-1) >= len(self.v_refs):
            raise ValueError(
                f"Scenario {self.name!r} reference_index values {reference_indices} "
                f"exceed v_refs length {len(self.v_refs)}"
            )
        if len(self.snap_labels) != 3:
            raise ValueError(
                f"Scenario {self.name!r} must provide exactly 3 snap labels, "
                f"got {len(self.snap_labels)}"
            )

    @property
    def agent_names(self) -> Tuple[str, ...]:
        return tuple(agent.name for agent in self.agents)

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    @property
    def n_control_blocks(self) -> int:
        return len(self.blocks)

    @property
    def control_blocks(self) -> Tuple[ControlBlockSpec, ...]:
        """Compact view for blocks that contain exactly one decision."""
        decisions_by_name = {decision.name: decision for decision in self.decisions}
        blocks = []
        for block in sorted(self.blocks, key=lambda item: item.block_index):
            if len(block.decision_names) != 1:
                raise ValueError(
                    "control_blocks view requires one decision per block; "
                    f"block {block.name!r} has {block.decision_names}"
                )
            decision = decisions_by_name[block.decision_names[0]]
            blocks.append(
                ControlBlockSpec(
                    name=block.name,
                    agent_name=block.owner_agent,
                    kind=decision.kind,
                    block_index=block.block_index,
                )
            )
        return tuple(blocks)

    @property
    def block_to_agent(self) -> Tuple[int, ...]:
        """Map each block index to the owning agent index expected by MGIGO."""
        agent_to_idx = {agent.name: idx for idx, agent in enumerate(self.agents)}
        ordered_blocks = sorted(self.blocks, key=lambda block: block.block_index)
        return tuple(agent_to_idx[block.owner_agent] for block in ordered_blocks)

    @property
    def block_dims(self) -> Tuple[int, ...]:
        """Flat decision dimension for each MGIGO block."""
        decisions_by_name = {decision.name: decision for decision in self.decisions}
        dims = []
        for block in sorted(self.blocks, key=lambda item: item.block_index):
            dims.append(sum(decisions_by_name[name].dim for name in block.decision_names))
        return tuple(dims)

    @property
    def context_state_dim(self) -> int:
        return self.n_agents * self.state_dim

    @property
    def context_ref_dim(self) -> int:
        return len(self.v_refs)

    @property
    def context_total_dim(self) -> int:
        return self.context_state_dim + self.context_ref_dim

    @property
    def solver_spec(self) -> SolverSpec:
        """Derived solver layout; scenario files should not duplicate this."""
        return SolverSpec(
            m_agent=self.n_agents,
            n_blocks=self.n_control_blocks,
            block_to_agent=self.block_to_agent,
            block_dims=self.block_dims,
            control_horizon=self.control_horizon,
        )


def state(x, y, v, psi=0.0, curr_acc=0.0, curr_steer=0.0):
    """Create the standard 6D vehicle state [x, y, v, psi, acc, steer]."""
    return np.array([x, y, v, psi, curr_acc, curr_steer], dtype=np.float64)

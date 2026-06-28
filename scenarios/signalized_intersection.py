"""Signalized intersection dilemma scenario.

Ego is the only optimizing agent. Cross traffic is an exogenous probabilistic
behavior model handled by the cost and visualization.
"""

import numpy as np

from config import CONTROL_HORIZON, LANE_W, SAFE_GAP, VEH_L, VEH_W
from .spec import (
    AgentSpec,
    BlockSpec,
    DecisionSpec,
    RoadSpec,
    ScenarioSpec,
    VehicleGeometrySpec,
    state,
)


EGO_ROAD_LANE_CENTERS = (
    -1.5 * LANE_W,
    -0.5 * LANE_W,
    0.5 * LANE_W,
    1.5 * LANE_W,
)
EGO_LANE_Y = -0.5 * LANE_W
INTERSECTION_CENTER_X = 42.0
CROSS_ROAD_LANE_CENTERS = (
    INTERSECTION_CENTER_X - 1.5 * LANE_W,
    INTERSECTION_CENTER_X - 0.5 * LANE_W,
    INTERSECTION_CENTER_X + 0.5 * LANE_W,
    INTERSECTION_CENTER_X + 1.5 * LANE_W,
)
CROSS_LANE_X = INTERSECTION_CENTER_X - 0.5 * LANE_W
INTERSECTION_ENTRY_X = INTERSECTION_CENTER_X - 2.0 * LANE_W
INTERSECTION_EXIT_X = INTERSECTION_CENTER_X + 2.0 * LANE_W
STOP_LINE_X = INTERSECTION_ENTRY_X - 2.0
CROSS_ROAD_HALF_WIDTH = 2.0 * LANE_W

EGO_X0 = 0.0
EGO_Y0 = EGO_LANE_Y
EGO_V0 = 14.0
EGO_REF_V = 13.0

YELLOW_START_S = 0.6
YELLOW_DURATION_S = 2.4
RED_START_S = YELLOW_START_S + YELLOW_DURATION_S


def _make_scenario(
    *,
    name,
    title_suffix,
    ego_x0,
    ego_v0,
    yellow_start_s,
    yellow_duration_s,
    snap_frames=(0, 14, 29),
    n_mpc_steps=30,
) -> ScenarioSpec:
    ego0 = state(ego_x0, EGO_Y0, ego_v0)
    red_start_s = yellow_start_s + yellow_duration_s

    return ScenarioSpec(
        name=name,
        title=f"MGIGO Signalized Intersection Dilemma - {title_suffix}",
        description=(
            f"{title_suffix}: ego approaches a yellow-light intersection with "
            "probabilistic cross traffic"
        ),
        output_prefix=f"mgigo_{name}",
        cost_profile="signalized_intersection",
        initial_states=np.stack([ego0]),
        v_refs=np.array([EGO_REF_V], dtype=np.float64),
        target_y=EGO_LANE_Y,
        lane_roles=("ego_approach",),
        agent_roles=("ego",),
        agents=(
            AgentSpec("ego", "signalized_intersection_ego", "bicycle", 0, 0),
        ),
        decisions=(
            DecisionSpec("ego_acc", "ego", "acc", (CONTROL_HORIZON,)),
            DecisionSpec("ego_steer", "ego", "steer", (CONTROL_HORIZON,)),
        ),
        blocks=(
            BlockSpec("ego_acc_block", "ego", ("ego_acc",), 0),
            BlockSpec("ego_steer_block", "ego", ("ego_steer",), 1),
        ),
        snap_labels=(
            r"$t\!=\!0$  (approach)",
            r"$t\!=\!t_{\mathrm{mid}}$  (dilemma)",
            r"$t\!=\!t_{\mathrm{end}}$  (stop/pass)",
        ),
        backend="generic_scenario",
        control_horizon=CONTROL_HORIZON,
        n_mpc_steps=n_mpc_steps,
        snap_frames=snap_frames,
        road=RoadSpec(LANE_W, EGO_ROAD_LANE_CENTERS),
        vehicle_geometry=VehicleGeometrySpec(VEH_L, VEH_W, SAFE_GAP),
        context_values=(yellow_start_s, red_start_s),
        notes=(
            "cross_traffic_exogenous",
            f"stop_line_x={STOP_LINE_X}",
            f"intersection_entry_x={INTERSECTION_ENTRY_X}",
            f"intersection_exit_x={INTERSECTION_EXIT_X}",
            f"cross_lane_x={CROSS_LANE_X}",
            f"yellow_start_s={yellow_start_s}",
            f"red_start_s={red_start_s}",
        ),
    )


def make_easy_pass_scenario() -> ScenarioSpec:
    return _make_scenario(
        name="signalized_intersection_easy_pass",
        title_suffix="Easy Pass",
        ego_x0=0.0,
        ego_v0=15.0,
        yellow_start_s=0.8,
        yellow_duration_s=3.2,
    )


def make_must_stop_scenario() -> ScenarioSpec:
    return _make_scenario(
        name="signalized_intersection_must_stop",
        title_suffix="Must Stop",
        ego_x0=4.0,
        ego_v0=11.0,
        yellow_start_s=0.2,
        yellow_duration_s=1.4,
    )


def make_critical_scenario() -> ScenarioSpec:
    return _make_scenario(
        name="signalized_intersection_critical",
        title_suffix="Critical Dilemma",
        ego_x0=0.0,
        ego_v0=14.0,
        yellow_start_s=0.6,
        yellow_duration_s=2.4,
    )


def make_scenario() -> ScenarioSpec:
    return _make_scenario(
        name="signalized_intersection",
        title_suffix="Critical Dilemma",
        ego_x0=0.0,
        ego_v0=14.0,
        yellow_start_s=0.6,
        yellow_duration_s=2.4,
    )

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


EGO_LANE_Y = 0.0
CROSS_LANE_X = 42.0
STOP_LINE_X = 34.0
INTERSECTION_ENTRY_X = 36.0
INTERSECTION_EXIT_X = 50.0
CROSS_ROAD_HALF_WIDTH = 4.0

EGO_X0 = 0.0
EGO_Y0 = EGO_LANE_Y
EGO_V0 = 14.0
EGO_REF_V = 13.0

YELLOW_START_S = 0.6
YELLOW_DURATION_S = 2.4
RED_START_S = YELLOW_START_S + YELLOW_DURATION_S


def make_scenario() -> ScenarioSpec:
    ego0 = state(EGO_X0, EGO_Y0, EGO_V0)

    return ScenarioSpec(
        name="signalized_intersection",
        title="MGIGO Signalized Intersection Dilemma",
        description=(
            "ego approaches a yellow-light intersection with probabilistic "
            "cross traffic"
        ),
        output_prefix="mgigo_signalized_intersection",
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
        n_mpc_steps=18,
        snap_frames=(0, 8, 17),
        road=RoadSpec(LANE_W, (EGO_LANE_Y,)),
        vehicle_geometry=VehicleGeometrySpec(VEH_L, VEH_W, SAFE_GAP),
        notes=(
            "cross_traffic_exogenous",
            f"stop_line_x={STOP_LINE_X}",
            f"intersection_entry_x={INTERSECTION_ENTRY_X}",
            f"intersection_exit_x={INTERSECTION_EXIT_X}",
            f"cross_lane_x={CROSS_LANE_X}",
            f"yellow_start_s={YELLOW_START_S}",
            f"red_start_s={RED_START_S}",
        ),
    )

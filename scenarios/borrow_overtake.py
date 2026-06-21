"""Borrow-lane overtaking scenario.

This file only defines the scenario data. The matching cost and oncoming-vehicle
dynamics will be wired in a separate planner refactor.
"""

import numpy as np

from config import (
    CONTROL_HORIZON,
    LANE_W,
    LOWER_LANE_Y,
    SAFE_GAP,
    UPPER_LANE_Y,
    VEH_L,
    VEH_W,
)
from .spec import (
    AgentSpec,
    BlockSpec,
    DecisionSpec,
    RoadSpec,
    ScenarioSpec,
    VehicleGeometrySpec,
    state,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Scenario Parameters
# ══════════════════════════════════════════════════════════════════════════════
# Road convention:
#   right lane (y=0.0): ego's normal lane, positive x direction
#   left lane  (y=3.5): oncoming lane, negative x direction
#
# Difficulty is mainly controlled by:
#   SLOW_GAP     = SLOW_X0 - EGO_X0
#   ONCOMING_TTC = (ONCOMING_X0 - EGO_X0) / (EGO_V0 + ONCOMING_V_ABS)
#
# Recommended sweeps later:
#   easy:     ONCOMING_X0 >= 150m
#   critical: ONCOMING_X0 around 95-120m
#   hard:     ONCOMING_X0 <= 80m
EGO_X0, EGO_Y0, EGO_V0 = 0.0, LOWER_LANE_Y, 17.0
SLOW_X0, SLOW_Y0, SLOW_V0 = 34.0, LOWER_LANE_Y, 8.0
ONCOMING_X0, ONCOMING_Y0, ONCOMING_V_ABS = 115.0, UPPER_LANE_Y, 18.0

EGO_REF_V = 17.0
SLOW_REF_V = 8.0
ONCOMING_REF_V = 18.0


def make_scenario() -> ScenarioSpec:
    ego0 = state(EGO_X0, EGO_Y0, EGO_V0, psi=0.0)
    slow0 = state(SLOW_X0, SLOW_Y0, SLOW_V0, psi=0.0)
    oncoming0 = state(ONCOMING_X0, ONCOMING_Y0, ONCOMING_V_ABS, psi=np.pi)
    oncoming_ttc = (ONCOMING_X0 - EGO_X0) / (EGO_V0 + ONCOMING_V_ABS)

    return ScenarioSpec(
        name="borrow_overtake",
        title="MGIGO Borrow-Lane Overtaking",
        description=(
            "ego follows a slow vehicle in the right lane and may borrow "
            f"the left oncoming lane; initial oncoming TTC={oncoming_ttc:.1f}s"
        ),
        output_prefix="mgigo_borrow_overtake",
        cost_profile="borrow_overtake",
        initial_states=np.stack([ego0, slow0, oncoming0]),
        v_refs=np.array([EGO_REF_V, SLOW_REF_V, ONCOMING_REF_V], dtype=np.float64),
        target_y=LOWER_LANE_Y,
        lane_roles=("ego_lane_positive_x", "oncoming_lane_negative_x"),
        agent_roles=("ego", "slow_lead_vehicle", "oncoming_vehicle"),
        agents=(
            AgentSpec("ego", "overtaking_vehicle", "bicycle", 0, 0),
            AgentSpec("slow_lead", "slow_lead_vehicle", "longitudinal", 1, 1),
            AgentSpec("oncoming", "oncoming_vehicle", "longitudinal", 2, 2),
        ),
        decisions=(
            DecisionSpec("ego_acc", "ego", "acc", (CONTROL_HORIZON,)),
            DecisionSpec("ego_steer", "ego", "steer", (CONTROL_HORIZON,)),
            DecisionSpec("slow_lead_acc", "slow_lead", "acc", (CONTROL_HORIZON,)),
            DecisionSpec("oncoming_acc", "oncoming", "acc", (CONTROL_HORIZON,)),
        ),
        blocks=(
            BlockSpec("ego_acc_block", "ego", ("ego_acc",), 0),
            BlockSpec("ego_steer_block", "ego", ("ego_steer",), 1),
            BlockSpec("slow_lead_acc_block", "slow_lead", ("slow_lead_acc",), 2),
            BlockSpec("oncoming_acc_block", "oncoming", ("oncoming_acc",), 3),
        ),
        snap_labels=(
            r"$t\!=\!0$  (approach)",
            r"$t\!=\!t_{\mathrm{mid}}$  (borrow lane)",
            r"$t\!=\!t_{\mathrm{end}}$  (return/check)",
        ),
        backend="generic_scenario",
        control_horizon=CONTROL_HORIZON,
        road=RoadSpec(LANE_W, (LOWER_LANE_Y, UPPER_LANE_Y)),
        vehicle_geometry=VehicleGeometrySpec(VEH_L, VEH_W, SAFE_GAP),
        notes=(
            "This scenario is registered for upcoming cost/dynamics refactor.",
            "Do not use this profile as a valid benchmark until borrow_overtake cost is wired.",
        ),
    )

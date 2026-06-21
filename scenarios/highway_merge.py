"""Highway merge scenario."""

import numpy as np

from config import (
    CONTROL_HORIZON,
    LANE_W, SAFE_GAP, VEH_L, VEH_W,
    LOWER_LANE_Y, TARGET_Y, UPPER_LANE_Y,
    V_EGO_DESIRED, V_FRONT_DESIRED, V_REAR_DESIRED,
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
# ego starts in the lower lane and merges into the upper lane.
# front/rear are same-direction vehicles in the target lane.
EGO_X0, EGO_Y0, EGO_V0 = 15.0, LOWER_LANE_Y, 17.0
FRONT_X0, FRONT_Y0, FRONT_V0 = 20.0, UPPER_LANE_Y, 20.0
REAR_X0, REAR_Y0, REAR_V0 = 10.0, UPPER_LANE_Y, 17.0

EGO_REF_V = V_EGO_DESIRED
FRONT_REF_V = V_FRONT_DESIRED
REAR_REF_V = V_REAR_DESIRED


def make_scenario() -> ScenarioSpec:
    ego0 = state(EGO_X0, EGO_Y0, EGO_V0)
    front0 = state(FRONT_X0, FRONT_Y0, FRONT_V0)
    rear0 = state(REAR_X0, REAR_Y0, REAR_V0)

    return ScenarioSpec(
        name="highway_merge",
        title="MGIGO Highway Merging",
        description=(
            f"ego merges from lane 0 (y={LOWER_LANE_Y:.1f}m) "
            f"to lane 1 (y={TARGET_Y:.1f}m)"
        ),
        output_prefix="mgigo_highway",
        cost_profile="highway_merge",
        initial_states=np.stack([ego0, front0, rear0]),
        v_refs=np.array([EGO_REF_V, FRONT_REF_V, REAR_REF_V], dtype=np.float64),
        target_y=TARGET_Y,
        lane_roles=("source_lane", "target_lane"),
        agent_roles=("ego", "front_same_direction", "rear_same_direction"),
        agents=(
            AgentSpec("ego", "merging_vehicle", "bicycle", 0, 0),
            AgentSpec("front", "front_same_direction", "longitudinal", 1, 1),
            AgentSpec("rear", "rear_same_direction", "longitudinal", 2, 2),
        ),
        decisions=(
            DecisionSpec("ego_acc", "ego", "acc", (CONTROL_HORIZON,)),
            DecisionSpec("ego_steer", "ego", "steer", (CONTROL_HORIZON,)),
            DecisionSpec("front_acc", "front", "acc", (CONTROL_HORIZON,)),
            DecisionSpec("rear_acc", "rear", "acc", (CONTROL_HORIZON,)),
        ),
        blocks=(
            BlockSpec("ego_acc_block", "ego", ("ego_acc",), 0),
            BlockSpec("ego_steer_block", "ego", ("ego_steer",), 1),
            BlockSpec("front_acc_block", "front", ("front_acc",), 2),
            BlockSpec("rear_acc_block", "rear", ("rear_acc",), 3),
        ),
        snap_labels=(
            r"$t\!=\!0$  (initial)",
            r"$t\!=\!t_{\mathrm{mid}}$  (merging)",
            r"$t\!=\!t_{\mathrm{end}}$  (merged)",
        ),
        backend="generic_scenario",
        control_horizon=CONTROL_HORIZON,
        road=RoadSpec(LANE_W, (LOWER_LANE_Y, UPPER_LANE_Y)),
        vehicle_geometry=VehicleGeometrySpec(VEH_L, VEH_W, SAFE_GAP),
        notes=(
            "Default cost uses the constraint DSL profile.",
            "All three vehicles move in the positive x direction.",
        ),
    )

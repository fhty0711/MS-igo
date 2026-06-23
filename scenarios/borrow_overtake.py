"""Borrow-lane overtaking scenario family.

The road is a two-lane, two-way rural/arterial road with no center divider:

- right lane (y=0.0): ego's normal lane, positive x direction
- left lane  (y=3.5): borrowed oncoming lane, negative x direction

The scenario variants below intentionally share the same agent/decision/block
layout. Only initial positions and speeds change, so a single future
``borrow_overtake`` cost profile can be A/B tested across safe, blocked, and
critical overtaking opportunities.
"""

from dataclasses import dataclass

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


# ODD reference ranges used to define the initial cases:
# - ego speed:        13.9-16.7 m/s  (50-60 km/h)
# - slow vehicle:      8.3-11.1 m/s  (30-40 km/h)
# - oncoming vehicle: 13.9-19.4 m/s  (50-70 km/h)
# - ego-to-slow gap:  15-25 m
# - oncoming range:  220-500 m, selected by the case difficulty.
#
# A rough critical-distance check is:
#   D_crit = T_overtake * (v_overtake + v_oncoming) + safety_margin
# where T_overtake includes acceleration, one lane change out, passing, and one
# lane change back. With v_overtake around 18-20 m/s, 2.5 s lane changes, and a
# 30-50 m safety margin, the critical distance is usually around 270-330 m for
# a passenger-car lead vehicle. This is why the critical case uses 340 m.
EGO_LANE_Y = LOWER_LANE_Y
ONCOMING_LANE_Y = UPPER_LANE_Y


@dataclass(frozen=True)
class BorrowOvertakeCase:
    """Initial conditions for one borrow-lane overtaking difficulty level."""

    name: str
    title_suffix: str
    output_suffix: str
    ego_speed: float
    slow_speed: float
    oncoming_speed: float
    slow_gap: float
    oncoming_distance: float
    ego_ref_speed: float
    slow_ref_speed: float
    oncoming_ref_speed: float
    expected_behavior: str
    notes: tuple[str, ...]


CASES = {
    "safe": BorrowOvertakeCase(
        name="safe",
        title_suffix="Safe Gap",
        output_suffix="safe",
        ego_speed=15.0,
        slow_speed=9.0,
        oncoming_speed=15.0,
        slow_gap=20.0,
        oncoming_distance=450.0,
        ego_ref_speed=18.0,
        slow_ref_speed=9.0,
        oncoming_ref_speed=15.0,
        expected_behavior="ego should find a safe opportunity to pass and return",
        notes=(
            "Safe case: oncoming vehicle starts far beyond the rough critical distance.",
            "Useful for checking whether the planner is overly conservative.",
        ),
    ),
    "blocked": BorrowOvertakeCase(
        name="blocked",
        title_suffix="Blocked Gap",
        output_suffix="blocked",
        ego_speed=15.0,
        slow_speed=10.0,
        oncoming_speed=17.0,
        slow_gap=18.0,
        oncoming_distance=220.0,
        ego_ref_speed=17.0,
        slow_ref_speed=10.0,
        oncoming_ref_speed=17.0,
        expected_behavior="ego should reject borrowing the oncoming lane",
        notes=(
            "Blocked case: oncoming vehicle starts below the rough critical distance.",
            "Useful as a safety-floor test; entering the oncoming lane should be costly.",
        ),
    ),
    "critical": BorrowOvertakeCase(
        name="critical",
        title_suffix="Critical Gap",
        output_suffix="critical",
        ego_speed=16.0,
        slow_speed=9.0,
        oncoming_speed=16.0,
        slow_gap=20.0,
        oncoming_distance=340.0,
        ego_ref_speed=18.0,
        slow_ref_speed=9.0,
        oncoming_ref_speed=16.0,
        expected_behavior="behavior depends on safety margin and cost aggressiveness",
        notes=(
            "Critical case: oncoming distance is near the closed-form threshold.",
            "Recommended sweep later: 300-420 m in 20 m increments.",
        ),
    ),
}


def _make_scenario(case: BorrowOvertakeCase, scenario_name: str) -> ScenarioSpec:
    """Build one scenario variant from the shared borrow-overtake layout."""
    ego_x0 = 0.0
    slow_x0 = ego_x0 + case.slow_gap
    oncoming_x0 = ego_x0 + case.oncoming_distance

    ego0 = state(ego_x0, EGO_LANE_Y, case.ego_speed, psi=0.0)
    slow0 = state(slow_x0, EGO_LANE_Y, case.slow_speed, psi=0.0)
    oncoming0 = state(
        oncoming_x0,
        ONCOMING_LANE_Y,
        case.oncoming_speed,
        psi=np.pi,
    )
    oncoming_ttc = case.oncoming_distance / (case.ego_speed + case.oncoming_speed)

    return ScenarioSpec(
        name=scenario_name,
        title=f"MGIGO Borrow-Lane Overtaking ({case.title_suffix})",
        description=(
            "ego follows a slow vehicle in the right lane and may borrow "
            f"the left oncoming lane; oncoming distance={case.oncoming_distance:.0f}m, "
            f"initial TTC={oncoming_ttc:.1f}s"
        ),
        output_prefix=f"mgigo_borrow_overtake_{case.output_suffix}",
        cost_profile="borrow_overtake",
        initial_states=np.stack([ego0, slow0, oncoming0]),
        v_refs=np.array(
            [case.ego_ref_speed, case.slow_ref_speed, case.oncoming_ref_speed],
            dtype=np.float64,
        ),
        target_y=EGO_LANE_Y,
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
            r"$t\!=\!t_{\mathrm{mid}}$  (decision)",
            r"$t\!=\!t_{\mathrm{end}}$  (return/check)",
        ),
        backend="generic_scenario",
        control_horizon=CONTROL_HORIZON,
        n_mpc_steps=35,
        snap_frames=(0, 18, 34),
        road=RoadSpec(LANE_W, (EGO_LANE_Y, ONCOMING_LANE_Y)),
        vehicle_geometry=VehicleGeometrySpec(VEH_L, VEH_W, SAFE_GAP),
        notes=(
            f"Expected behavior: {case.expected_behavior}.",
            *case.notes,
            "The matching borrow_overtake cost profile is intentionally added separately.",
        ),
    )


def make_scenario(case_name: str = "critical", scenario_name: str = "borrow_overtake"):
    """Return a borrow-overtake scenario; default alias points to critical."""
    try:
        case = CASES[case_name]
    except KeyError as exc:
        available = ", ".join(sorted(CASES))
        raise ValueError(f"Unknown borrow-overtake case {case_name!r}: {available}") from exc
    return _make_scenario(case, scenario_name)


def make_safe_scenario() -> ScenarioSpec:
    return make_scenario("safe", "borrow_overtake_safe")


def make_blocked_scenario() -> ScenarioSpec:
    return make_scenario("blocked", "borrow_overtake_blocked")


def make_critical_scenario() -> ScenarioSpec:
    return make_scenario("critical", "borrow_overtake_critical")

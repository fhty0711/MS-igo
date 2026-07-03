"""B-spline variant of the signalized intersection benchmark."""

from __future__ import annotations

from pathlib import Path

from trajectory.frenet_bspline import FrenetBSplineTrajectory
from trajectory.reference_path import StraightReference
from trajectory.warmstart import initial_component_means

from .signalized_intersection import EGO_V0, _make_scenario
from .spec import BlockSpec, DecisionSpec, ScenarioSpec


def make_scenario() -> ScenarioSpec:
    """Return a critical signalized dilemma with Frenet B-spline decisions."""
    base = _make_scenario(
        name="signalized_intersection_bspline",
        title_suffix="B-Spline Critical Dilemma",
        ego_x0=0.0,
        ego_v0=EGO_V0,
        yellow_start_s=0.6,
        yellow_duration_s=2.7,
        n_mpc_steps=30,
    )
    basis_path = (
        Path(__file__).resolve().parents[1]
        / "trajectory"
        / "assets"
        / "bspline_basis.npz"
    )
    gen = FrenetBSplineTrajectory(
        basis_path,
        StraightReference(),
    )
    means = initial_component_means(
        gen,
        s0=float(base.initial_states[0, 0]),
        s_dot0=float(base.initial_states[0, 2]),
        d0=float(base.initial_states[0, 1]),
        n_components=3,
    )
    return ScenarioSpec(
        name="signalized_intersection_bspline",
        title="MGIGO Signalized Intersection Dilemma - B-Spline Critical",
        description=(
            "B-spline critical dilemma: ego optimizes Frenet spline control "
            "points while cross traffic remains an exogenous probabilistic model"
        ),
        output_prefix="mgigo_signalized_intersection_bspline",
        cost_profile="signalized_intersection_bspline",
        initial_states=base.initial_states,
        v_refs=base.v_refs,
        target_y=base.target_y,
        lane_roles=base.lane_roles,
        agent_roles=base.agent_roles,
        agents=base.agents,
        decisions=(
            DecisionSpec("ego_ctrl_s", "ego", "ctrl_s", (gen.n_free,)),
            DecisionSpec("ego_ctrl_d", "ego", "ctrl_d", (gen.n_free,)),
        ),
        blocks=(
            BlockSpec("ego_ctrl_s_block", "ego", ("ego_ctrl_s",), 0),
            BlockSpec("ego_ctrl_d_block", "ego", ("ego_ctrl_d",), 1),
        ),
        snap_labels=base.snap_labels,
        backend=base.backend,
        state_dim=base.state_dim,
        control_horizon=gen.n_free,
        n_mpc_steps=base.n_mpc_steps,
        snap_frames=base.snap_frames,
        road=base.road,
        vehicle_geometry=base.vehicle_geometry,
        context_values=base.context_values,
        notes=base.notes + ("trajectory_model=frenet_bspline",),
        exec_mode=base.exec_mode,
        trajectory_model="frenet_bspline",
        initial_component_means=means,
    )

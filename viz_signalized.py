"""Project adapters for the standalone signalized-intersection renderer."""

from __future__ import annotations

from config import SUB_STEPS
from visualization.signalized_renderer import (
    RoadGeometry,
    SignalTiming,
    SignalizedScene,
    VehicleGeometry,
    draw_cross_traffic_cloud as _draw_cross_traffic_cloud,
    draw_intersection_metrics as _draw_intersection_metrics,
    draw_signalized_scene as _draw_signalized_scene,
    estimate_visual_metrics as _estimate_visual_metrics,
    semantic_layer_summary as _semantic_layer_summary,
)


def scene_from_scenario(scenario) -> SignalizedScene:
    """Convert the project ScenarioSpec into a standalone renderer scene."""
    from costs import signalized_intersection as si
    from scenarios import signalized_intersection as sig

    yellow_start_s, red_start_s = scenario.context_values[:2]
    return SignalizedScene(
        road=RoadGeometry(
            lane_width=float(scenario.road.lane_width),
            horizontal_lane_centers=tuple(float(v) for v in scenario.road.lane_centers),
            vertical_lane_centers=tuple(float(v) for v in sig.CROSS_ROAD_LANE_CENTERS),
        ),
        vehicle=VehicleGeometry(
            length=float(scenario.vehicle_geometry.length),
            width=float(scenario.vehicle_geometry.width),
            safe_gap=float(scenario.vehicle_geometry.safe_gap),
        ),
        signal=SignalTiming(
            yellow_start_s=float(yellow_start_s),
            red_start_s=float(red_start_s),
        ),
        intersection_center_x=float(sig.INTERSECTION_CENTER_X),
        intersection_entry_x=float(si.INTERSECTION_ENTRY_X),
        intersection_exit_x=float(si.INTERSECTION_EXIT_X),
        stop_line_x=float(si.STOP_LINE_X),
        cross_lane_x=float(sig.CROSS_LANE_X),
        cross_start_y=float(si.CROSS_START_Y),
        cross_obey_stop_buffer=0.5 * float(scenario.vehicle_geometry.safe_gap),
    )


def semantic_layer_summary(scenario):
    """Return stable semantic-layer counts for tests and report diagnostics."""
    from costs import signalized_intersection as si

    return _semantic_layer_summary(
        scene_from_scenario(scenario),
        risk_cloud_samples=si.DEV_N_SAMPLES,
    )


def draw_signalized_scene(ax, scenario, x_min, x_max, elapsed_time_s=0.0):
    """Draw a top-down semantic intersection map from a ScenarioSpec."""
    return _draw_signalized_scene(
        ax,
        scene_from_scenario(scenario),
        x_min,
        x_max,
        elapsed_time_s=elapsed_time_s,
    )


def draw_cross_traffic_cloud(ax, scenario, ego_traj=None):
    """Draw exogenous cross-traffic samples from a ScenarioSpec adapter."""
    from costs import signalized_intersection as si

    return _draw_cross_traffic_cloud(
        ax,
        scene_from_scenario(scenario),
        ego_traj=ego_traj,
        n_steps=scenario.control_horizon * SUB_STEPS,
        dt=si.DT_C,
    )


def draw_intersection_metrics(
    ax,
    ego_traj,
    *,
    red_start_s=None,
    dt=None,
    time_offset_s=0.0,
):
    """Draw compact display-only metrics using the project default scene."""
    from costs import signalized_intersection as si
    from scenarios import get_scenario

    scenario = get_scenario("signalized_intersection")
    scene = scene_from_scenario(scenario)
    if red_start_s is not None:
        scene = SignalizedScene(
            road=scene.road,
            vehicle=scene.vehicle,
            signal=SignalTiming(
                yellow_start_s=scene.signal.yellow_start_s,
                red_start_s=float(red_start_s),
            ),
            intersection_center_x=scene.intersection_center_x,
            intersection_entry_x=scene.intersection_entry_x,
            intersection_exit_x=scene.intersection_exit_x,
            stop_line_x=scene.stop_line_x,
            cross_lane_x=scene.cross_lane_x,
            cross_start_y=scene.cross_start_y,
            cross_obey_stop_buffer=scene.cross_obey_stop_buffer,
        )
    return _draw_intersection_metrics(
        ax,
        scene,
        ego_traj,
        dt=si.DT_C if dt is None else dt,
        time_offset_s=time_offset_s,
    )


def estimate_visual_metrics(
    ego_traj_np,
    n_samples=60,
    *,
    red_start_s=None,
    dt=None,
    time_offset_s=0.0,
):
    """Compatibility wrapper for display-only intersection metrics."""
    from costs import signalized_intersection as si
    from scenarios import get_scenario

    scenario = get_scenario("signalized_intersection")
    scene = scene_from_scenario(scenario)
    if red_start_s is not None:
        scene = SignalizedScene(
            road=scene.road,
            vehicle=scene.vehicle,
            signal=SignalTiming(
                yellow_start_s=scene.signal.yellow_start_s,
                red_start_s=float(red_start_s),
            ),
            intersection_center_x=scene.intersection_center_x,
            intersection_entry_x=scene.intersection_entry_x,
            intersection_exit_x=scene.intersection_exit_x,
            stop_line_x=scene.stop_line_x,
            cross_lane_x=scene.cross_lane_x,
            cross_start_y=scene.cross_start_y,
            cross_obey_stop_buffer=scene.cross_obey_stop_buffer,
        )
    return _estimate_visual_metrics(
        scene,
        ego_traj_np,
        n_samples=n_samples,
        dt=si.DT_C if dt is None else dt,
        time_offset_s=time_offset_s,
    )

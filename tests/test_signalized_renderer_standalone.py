"""Tests for the standalone signalized intersection renderer."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_signalized_renderer_imports_without_project_scenario_dependencies():
    import visualization.signalized_renderer as renderer

    module_names = set(sys.modules)
    forbidden = {
        "costs.signalized_intersection",
        "scenarios.signalized_intersection",
        "scenarios.registry",
    }
    loaded_forbidden = forbidden & module_names
    if loaded_forbidden:
        raise AssertionError(loaded_forbidden)
    if renderer.SignalizedScene is None:
        raise AssertionError("renderer should expose plain scene dataclasses")


def test_signalized_renderer_draws_plain_scene_contract():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualization.signalized_renderer import (
        RoadGeometry,
        SignalTiming,
        SignalizedScene,
        VehicleGeometry,
        draw_signalized_scene,
        semantic_layer_summary,
    )

    scene = SignalizedScene(
        road=RoadGeometry(
            lane_width=3.5,
            horizontal_lane_centers=(-5.25, -1.75, 1.75, 5.25),
            vertical_lane_centers=(36.75, 40.25, 43.75, 47.25),
        ),
        vehicle=VehicleGeometry(length=5.0, width=2.0, safe_gap=3.0),
        signal=SignalTiming(yellow_start_s=0.6, red_start_s=3.3),
        intersection_center_x=42.0,
        intersection_entry_x=35.0,
        intersection_exit_x=49.0,
        stop_line_x=33.0,
        cross_lane_x=40.25,
    )

    fig, ax = plt.subplots(figsize=(5, 3))
    try:
        draw_signalized_scene(ax, scene, 20.0, 60.0, elapsed_time_s=4.0)
        gids = {patch.get_gid() for patch in ax.patches if patch.get_gid()}
    finally:
        plt.close(fig)
    for gid in ("lane_polygon", "crosswalk", "conflict_box", "traffic_signal_red_active"):
        if gid not in gids:
            raise AssertionError((gid, gids))

    layers = semantic_layer_summary(scene, risk_cloud_samples=30)
    if layers["horizontal_lanes"] != 4 or layers["vertical_lanes"] != 4:
        raise AssertionError(layers)


if __name__ == "__main__":
    test_signalized_renderer_imports_without_project_scenario_dependencies()
    test_signalized_renderer_draws_plain_scene_contract()
    print("standalone signalized renderer tests ok")

"""Tests for the shared layer-based scene renderer."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_scene_renderer_draws_basic_layers():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualization.scene_renderer import (
        CircleLayer,
        LineLayer,
        RectLayer,
        SceneRenderSpec,
        TextLayer,
        render_scene,
    )

    spec = SceneRenderSpec(
        facecolor="#000000",
        xlim=(-1.0, 5.0),
        ylim=(-2.0, 4.0),
        aspect="equal",
        patches=(
            RectLayer(
                xy=(0.0, 0.0),
                width=2.0,
                height=1.0,
                facecolor="#ffffff",
                edgecolor="#ff0000",
                gid="rect_layer",
            ),
            CircleLayer(
                center=(3.0, 1.0),
                radius=0.4,
                facecolor="#00ff00",
                gid="circle_layer",
            ),
        ),
        lines=(
            LineLayer(
                x=(0.0, 4.0),
                y=(2.0, 2.0),
                color="#abcdef",
                gid="line_layer",
            ),
        ),
        texts=(
            TextLayer(
                x=0.5,
                y=0.5,
                text="demo",
                color="#ffffff",
                gid="text_layer",
            ),
        ),
    )

    fig, ax = plt.subplots(figsize=(4, 3))
    try:
        render_scene(ax, spec)
        patch_gids = {patch.get_gid() for patch in ax.patches if patch.get_gid()}
        line_gids = {line.get_gid() for line in ax.lines if line.get_gid()}
        text_gids = {text.get_gid() for text in ax.texts if text.get_gid()}
    finally:
        plt.close(fig)

    if {"rect_layer", "circle_layer"} - patch_gids:
        raise AssertionError(patch_gids)
    if "line_layer" not in line_gids:
        raise AssertionError(line_gids)
    if "text_layer" not in text_gids:
        raise AssertionError(text_gids)


def test_scene_renderer_draws_oriented_vehicle_footprint():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualization.scene_renderer import (
        SceneRenderSpec,
        VehicleLayer,
        render_scene,
    )

    spec = SceneRenderSpec(
        vehicles=(
            VehicleLayer(
                center=(1.0, 2.0),
                heading=0.25,
                length=4.5,
                width=1.9,
                facecolor="#22cc88",
                edgecolor="#ffffff",
                gid="vehicle_layer",
            ),
        )
    )

    fig, ax = plt.subplots(figsize=(4, 3))
    try:
        render_scene(ax, spec)
        gids = {patch.get_gid() for patch in ax.patches if patch.get_gid()}
    finally:
        plt.close(fig)

    if "vehicle_layer" not in gids:
        raise AssertionError(gids)


if __name__ == "__main__":
    test_scene_renderer_draws_basic_layers()
    test_scene_renderer_draws_oriented_vehicle_footprint()
    print("unified scene renderer tests ok")

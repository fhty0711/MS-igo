"""Standalone NuPlan-style renderer for signalized-intersection scenes.

This module intentionally depends only on NumPy and Matplotlib. It does not
import the MGIGO scenario, cost, or planner modules; callers provide a plain
``SignalizedScene`` dataclass or an equivalent object.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.patches as mpatches
import numpy as np
from visualization.scene_renderer import (
    ArrowLayer,
    CircleLayer,
    LineLayer,
    RectLayer,
    SceneRenderSpec,
    VehicleLayer,
    draw_vehicle_footprint as _render_vehicle_footprint,
    render_scene,
)


MAP_BG = "#111827"
ROAD_FILL = "#253244"
LANE_FILL_A = "#2f3d52"
LANE_FILL_B = "#29384d"
LANE_EDGE = "#d7dee8"
DIVIDER = "#cbd5e1"
STOP_LINE = "#f8fafc"
CONFLICT = "#f59e0b"
CROSSWALK = "#e5e7eb"
SIGNAL_POST = "#94a3b8"
SIGNAL_GREEN = "#22c55e"
SIGNAL_YELLOW = "#facc15"
SIGNAL_RED = "#ef4444"

CROSS_FOOTPRINT_ALPHA = 0.055
CROSS_FOOTPRINT_CRITICAL_ALPHA = 0.24

MODE_OBEY = 0
MODE_YELLOW_RUSH = 1
MODE_RED_RUN = 2
XI_MODE = 0
XI_ARRIVAL_SHIFT = 1
XI_SPEED_SCALE = 2
XI_LATERAL_OFFSET = 3


@dataclass(frozen=True)
class RoadGeometry:
    """Lane-level road geometry for a top-down intersection scene."""

    lane_width: float
    horizontal_lane_centers: tuple[float, ...]
    vertical_lane_centers: tuple[float, ...]

    @property
    def horizontal_min_y(self) -> float:
        return min(self.horizontal_lane_centers) - 0.5 * self.lane_width

    @property
    def horizontal_max_y(self) -> float:
        return max(self.horizontal_lane_centers) + 0.5 * self.lane_width

    @property
    def vertical_min_x(self) -> float:
        return min(self.vertical_lane_centers) - 0.5 * self.lane_width

    @property
    def vertical_max_x(self) -> float:
        return max(self.vertical_lane_centers) + 0.5 * self.lane_width


@dataclass(frozen=True)
class VehicleGeometry:
    """Vehicle footprint and safety envelope settings."""

    length: float
    width: float
    safe_gap: float


@dataclass(frozen=True)
class SignalTiming:
    """Traffic-light timing in seconds."""

    yellow_start_s: float
    red_start_s: float


@dataclass(frozen=True)
class SignalizedScene:
    """Plain data contract consumed by the standalone signalized renderer."""

    road: RoadGeometry
    vehicle: VehicleGeometry
    signal: SignalTiming
    intersection_center_x: float
    intersection_entry_x: float
    intersection_exit_x: float
    stop_line_x: float
    cross_lane_x: float
    cross_start_y: float = -18.0
    cross_obey_stop_buffer: float = 1.5


def draw_vehicle_footprint(
    ax,
    *,
    center_x,
    center_y,
    heading,
    length,
    width,
    color,
    alpha,
    linewidth,
    zorder,
    gid,
):
    """Draw a top-down oriented vehicle footprint centered on a state."""
    return _render_vehicle_footprint(
        ax,
        VehicleLayer(
            center=(float(center_x), float(center_y)),
            heading=float(heading),
            length=float(length),
            width=float(width),
            facecolor=color,
            edgecolor=color,
            linewidth=float(linewidth),
            alpha=float(alpha),
            zorder=float(zorder),
            gid=gid,
        ),
    )


def _draw_cross_traffic_legend(ax, mode_colors):
    handles = [
        mpatches.Patch(facecolor=mode_colors["obey"], edgecolor="none", label="obey"),
        mpatches.Patch(
            facecolor=mode_colors["yellow_rush"],
            edgecolor="none",
            label="yellow-rush",
        ),
        mpatches.Patch(
            facecolor=mode_colors["red_run"],
            edgecolor="none",
            label="red-run",
        ),
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=6.5,
        frameon=True,
        framealpha=0.78,
        facecolor="#020617",
        edgecolor="#475569",
        labelcolor="#e5e7eb",
        borderpad=0.35,
        handlelength=1.1,
        handletextpad=0.45,
    )
    legend.set_zorder(21)


def semantic_layer_summary(scene: SignalizedScene, risk_cloud_samples: int = 40):
    """Return stable semantic-layer counts for tests and diagnostics."""
    return {
        "horizontal_lanes": len(scene.road.horizontal_lane_centers),
        "vertical_lanes": len(scene.road.vertical_lane_centers),
        "crosswalks": 2,
        "stop_lines": 1,
        "direction_arrows": (
            len(scene.road.horizontal_lane_centers)
            + len(scene.road.vertical_lane_centers)
        ),
        "risk_cloud_samples": int(risk_cloud_samples),
        "traffic_signals": 1,
    }


def signal_phase(scene: SignalizedScene, elapsed_time_s):
    """Return active signal phase from scene timing."""
    elapsed = float(elapsed_time_s)
    if elapsed < float(scene.signal.yellow_start_s):
        return "green"
    if elapsed < float(scene.signal.red_start_s):
        return "yellow"
    return "red"


def draw_signalized_scene(ax, scene: SignalizedScene, x_min, x_max, elapsed_time_s=0.0):
    """Draw a top-down semantic intersection map."""
    road_min = scene.road.horizontal_min_y
    road_max = scene.road.horizontal_max_y
    lane_w = scene.road.lane_width
    cross_min = -22.0
    cross_max = 22.0
    cross_road_min = scene.road.vertical_min_x
    cross_road_max = scene.road.vertical_max_x

    patches = [
        RectLayer(
            xy=(float(x_min), float(road_min)),
            width=float(x_max - x_min),
            height=float(road_max - road_min),
            facecolor=ROAD_FILL,
            edgecolor=LANE_EDGE,
            linewidth=1.3,
            alpha=1.0,
            zorder=1,
            gid="road_surface",
        ),
        RectLayer(
            xy=(float(cross_road_min), float(cross_min)),
            width=float(cross_road_max - cross_road_min),
            height=float(cross_max - cross_min),
            facecolor=ROAD_FILL,
            edgecolor=LANE_EDGE,
            linewidth=1.3,
            alpha=1.0,
            zorder=1,
            gid="road_surface",
        ),
    ]
    for idx, y in enumerate(scene.road.horizontal_lane_centers):
        patches.append(
            RectLayer(
                xy=(float(x_min), float(y - 0.5 * lane_w)),
                width=float(x_max - x_min),
                height=float(lane_w),
                facecolor=LANE_FILL_A if idx % 2 == 0 else LANE_FILL_B,
                edgecolor="none",
                alpha=0.62,
                zorder=2,
                gid="lane_polygon",
            )
        )
    for idx, x in enumerate(scene.road.vertical_lane_centers):
        patches.append(
            RectLayer(
                xy=(float(x - 0.5 * lane_w), float(cross_min)),
                width=float(lane_w),
                height=float(cross_max - cross_min),
                facecolor=LANE_FILL_B if idx % 2 == 0 else LANE_FILL_A,
                edgecolor="none",
                alpha=0.55,
                zorder=2,
                gid="lane_polygon",
            )
        )

    patches.append(
        RectLayer(
            xy=(float(scene.intersection_entry_x), float(road_min)),
            width=float(scene.intersection_exit_x - scene.intersection_entry_x),
            height=float(road_max - road_min),
            facecolor=CONFLICT,
            edgecolor="#fde68a",
            linewidth=1.1,
            alpha=0.22,
            zorder=4,
            gid="conflict_box",
        )
    )

    lines = []
    for y0, y1 in zip(
        scene.road.horizontal_lane_centers[:-1],
        scene.road.horizontal_lane_centers[1:],
    ):
        lane_mid = 0.5 * (y0 + y1)
        lines.append(
            LineLayer(
                x=(float(x_min), float(x_max)),
                y=(float(lane_mid), float(lane_mid)),
                color=DIVIDER,
                linewidth=0.9,
                linestyle=(0, (6, 6)),
                alpha=0.72,
                zorder=5,
                gid="lane_divider",
            )
        )
    for x0, x1 in zip(
        scene.road.vertical_lane_centers[:-1],
        scene.road.vertical_lane_centers[1:],
    ):
        lane_mid = 0.5 * (x0 + x1)
        lines.append(
            LineLayer(
                x=(float(lane_mid), float(lane_mid)),
                y=(float(cross_min), float(cross_max)),
                color=DIVIDER,
                linewidth=0.9,
                linestyle=(0, (6, 6)),
                alpha=0.72,
                zorder=5,
                gid="lane_divider",
            )
        )

    lines.append(
        LineLayer(
            x=(float(scene.stop_line_x), float(scene.stop_line_x)),
            y=(float(road_min), float(road_max)),
            color=STOP_LINE,
            linewidth=3.0,
            zorder=8,
            gid="stop_line",
        )
    )

    width = road_max - road_min
    stripe_h = 0.55
    for x0 in (scene.stop_line_x + 0.8, scene.intersection_exit_x - 1.6):
        for idx in range(5):
            y = road_min + 0.6 + idx * (width - 1.2) / 5.0
            patches.append(
                RectLayer(
                    xy=(float(x0), float(y)),
                    width=1.0,
                    height=stripe_h,
                    facecolor=CROSSWALK,
                    edgecolor="none",
                    alpha=0.82,
                    zorder=7,
                    gid="crosswalk",
                )
            )

    arrows = []
    x_arrow = x_min + 8.0
    for y in scene.road.horizontal_lane_centers:
        direction = 1.0 if y < 0.0 else -1.0
        arrows.append(
            ArrowLayer(
                x=float(x_arrow if direction > 0 else x_max - 8.0),
                y=float(y),
                dx=float(5.0 * direction),
                dy=0.0,
                color="#dbeafe",
                width=0.08,
                head_width=0.7,
                head_length=1.0,
                alpha=0.55,
                zorder=5,
                gid="lane_direction_arrow",
            )
        )
    y_arrow = cross_min + 7.0
    for x in scene.road.vertical_lane_centers:
        direction = 1.0 if x < scene.intersection_center_x else -1.0
        arrows.append(
            ArrowLayer(
                x=float(x),
                y=float(y_arrow if direction > 0 else cross_max - 7.0),
                dx=0.0,
                dy=float(5.0 * direction),
                color="#dbeafe",
                width=0.08,
                head_width=0.7,
                head_length=1.0,
                alpha=0.55,
                zorder=5,
                gid="lane_direction_arrow",
            )
        )

    lamp_x = scene.stop_line_x + 2.2
    lamp_y = road_max + 1.1
    phase = signal_phase(scene, elapsed_time_s)
    lines.append(
        LineLayer(
            x=(float(lamp_x), float(lamp_x)),
            y=(float(road_max + 0.1), float(lamp_y - 0.85)),
            color=SIGNAL_POST,
            linewidth=1.7,
            zorder=10,
            gid="traffic_signal_post",
        )
    )
    patches.append(
        RectLayer(
            xy=(float(lamp_x - 0.58), float(lamp_y - 0.92)),
            width=1.16,
            height=1.84,
            boxstyle="round,pad=0.08,rounding_size=0.18",
            facecolor="#0f172a",
            edgecolor=LANE_EDGE,
            linewidth=0.75,
            zorder=11,
            gid="traffic_signal",
        )
    )
    lamp_specs = (
        ("red", SIGNAL_RED, lamp_y + 0.55),
        ("yellow", SIGNAL_YELLOW, lamp_y),
        ("green", SIGNAL_GREEN, lamp_y - 0.55),
    )
    for name, color, y in lamp_specs:
        active = name == phase
        patches.append(
            CircleLayer(
                center=(float(lamp_x), float(y)),
                radius=0.27 if active else 0.21,
                facecolor=color,
                edgecolor="#f8fafc" if active else "none",
                linewidth=0.65 if active else 0.0,
                alpha=0.96 if active else 0.22,
                zorder=12 if active else 11,
                gid=f"traffic_signal_{name}_{'active' if active else 'inactive'}",
            )
        )

    render_scene(
        ax,
        SceneRenderSpec(
            facecolor=MAP_BG,
            ylim=(float(cross_min), float(cross_max)),
            patches=tuple(patches),
            lines=tuple(lines),
            arrows=tuple(arrows),
        )
    )


def cross_traffic_samples(n=30):
    """Deterministic multimodal cross-traffic behavior table."""
    idx = np.arange(int(n))
    mode = np.where(
        idx < int(0.60 * n),
        MODE_OBEY,
        np.where(idx < int(0.90 * n), MODE_YELLOW_RUSH, MODE_RED_RUN),
    )
    phase = (idx.astype(float) + 0.5) / max(float(n), 1.0)
    arrival_shift = np.where(
        mode == MODE_OBEY,
        2.2 + 0.8 * phase,
        np.where(mode == MODE_YELLOW_RUSH, -0.2 + 0.7 * phase, -0.8 + 0.5 * phase),
    )
    speed_scale = np.where(
        mode == MODE_OBEY,
        0.8 + 0.1 * phase,
        np.where(mode == MODE_YELLOW_RUSH, 1.05 + 0.15 * phase, 1.15 + 0.2 * phase),
    )
    lateral_offset = (phase - 0.5) * 0.7
    return np.stack(
        [
            mode.astype(float),
            arrival_shift.astype(float),
            speed_scale.astype(float),
            lateral_offset.astype(float),
        ],
        axis=1,
    )


def cross_traj_for_sample(scene: SignalizedScene, sample, n_steps, dt=0.05):
    """Roll out one exogenous cross-traffic behavior sample."""
    xi = np.asarray(sample, dtype=float)
    t = np.arange(int(n_steps), dtype=float) * float(dt)
    mode = int(xi[XI_MODE])
    arrival_shift = float(xi[XI_ARRIVAL_SHIFT])
    speed_scale = float(xi[XI_SPEED_SCALE])
    lateral_offset = float(xi[XI_LATERAL_OFFSET])
    base_speed = 7.5 * speed_scale
    y = scene.cross_start_y + base_speed * (t - arrival_shift)
    if mode == MODE_OBEY:
        obey_stop_y = (
            scene.road.horizontal_min_y
            - 0.5 * scene.vehicle.length
            - scene.cross_obey_stop_buffer
        )
        y = np.minimum(y, obey_stop_y)
    x = np.full_like(y, scene.cross_lane_x + lateral_offset)
    v = np.full_like(y, base_speed)
    psi = np.full_like(y, np.pi / 2.0)
    zeros = np.zeros_like(y)
    return np.stack([x, y, v, psi, zeros, zeros], axis=1)


def _pair_clearance(scene: SignalizedScene, a_traj, b_traj):
    a = np.asarray(a_traj, dtype=float)
    b = np.asarray(b_traj, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return float("inf")
    dx = np.abs(a[:n, 0] - b[:n, 0])
    dy = np.abs(a[:n, 1] - b[:n, 1])
    min_dx = scene.vehicle.length + scene.vehicle.safe_gap
    min_dy = scene.vehicle.width + 0.5 * scene.vehicle.safe_gap
    penetration = np.minimum(min_dx - dx, min_dy - dy)
    return float(-np.max(penetration))


def classify_ego_mode(scene: SignalizedScene, ego_traj):
    """Classify a displayed ego trajectory as stop, pass, or undecided."""
    ego = np.asarray(ego_traj, dtype=float)
    if ego.size == 0:
        return "undecided"
    x = ego[:, 0]
    v = ego[:, 2]
    stopped_before = np.any((x <= scene.stop_line_x) & (v <= 1.0))
    cleared = np.max(x) >= scene.intersection_exit_x
    blocking = np.any(
        (x > scene.intersection_entry_x)
        & (x < scene.intersection_exit_x)
        & (v <= 1.0)
    )
    if stopped_before and not cleared:
        return "stop"
    if cleared and not blocking:
        return "pass"
    return "undecided"


def red_legal_for_history(scene: SignalizedScene, ego_traj, *, dt, time_offset_s=0.0):
    """Evaluate red-light legality for a displayed history sampled at dt."""
    ego = np.asarray(ego_traj, dtype=float)
    if len(ego) == 0:
        return True
    t = float(time_offset_s) + np.arange(len(ego), dtype=float) * float(dt)
    red = t >= float(scene.signal.red_start_s)
    legal = (ego[:, 0] <= scene.stop_line_x) | (ego[:, 0] >= scene.intersection_exit_x)
    return bool(np.all(~red | legal))


def no_blocking(scene: SignalizedScene, ego_traj):
    """Return whether the displayed ego trajectory avoids stopped blocking."""
    ego = np.asarray(ego_traj, dtype=float)
    if len(ego) == 0:
        return True
    x = ego[:, 0]
    v = ego[:, 2]
    in_box = (x > scene.intersection_entry_x) & (x < scene.intersection_exit_x)
    stopped_in_box = in_box & (v < 1.0)
    terminal_in_box = bool(in_box[-1])
    return bool(not np.any(stopped_in_box) and not terminal_in_box)


def estimate_visual_metrics(
    scene: SignalizedScene,
    ego_traj,
    n_samples=60,
    *,
    dt=0.05,
    time_offset_s=0.0,
):
    """Compute display-only intersection metrics from one ego trajectory."""
    ego = np.asarray(ego_traj, dtype=float)
    samples = cross_traffic_samples(n_samples)
    if len(ego) == 0:
        clearances = np.array([float("inf")])
        critical = samples[0] if len(samples) else np.zeros(4)
    else:
        clearances = np.array(
            [
                _pair_clearance(
                    scene,
                    ego,
                    cross_traj_for_sample(scene, xi, len(ego), dt=dt),
                )
                for xi in samples
            ],
            dtype=float,
        )
        critical = samples[int(np.argmin(clearances))]
    return {
        "mode": classify_ego_mode(scene, ego),
        "min_clearance": float(np.min(clearances)),
        "risk_quantile": float(np.quantile(-clearances, 0.9)),
        "red_legal": red_legal_for_history(
            scene,
            ego,
            dt=dt,
            time_offset_s=time_offset_s,
        ),
        "no_blocking": no_blocking(scene, ego),
        "critical_sample": critical,
    }


def draw_cross_traffic_cloud(
    ax,
    scene: SignalizedScene,
    *,
    ego_traj=None,
    n_steps=None,
    dt=0.05,
    samples=None,
):
    """Draw exogenous cross-traffic samples with mode-coded probability paths."""
    samples = cross_traffic_samples(30) if samples is None else np.asarray(samples)
    n_steps = 120 if n_steps is None else int(n_steps)
    critical_xi = None
    if ego_traj is not None:
        metrics = estimate_visual_metrics(
            scene,
            ego_traj,
            n_samples=30,
            dt=dt,
        )
        critical_xi = np.asarray(metrics["critical_sample"], dtype=float)
    mode_colors = {
        "obey": "#38bdf8",
        "yellow_rush": "#facc15",
        "red_run": "#fb7185",
    }
    geom = scene.vehicle
    for xi in samples:
        mode = int(xi[XI_MODE])
        color = {
            MODE_OBEY: mode_colors["obey"],
            MODE_YELLOW_RUSH: mode_colors["yellow_rush"],
            MODE_RED_RUN: mode_colors["red_run"],
        }.get(mode, "#ffffff")
        tr = cross_traj_for_sample(scene, xi, n_steps, dt=dt)
        is_critical = critical_xi is not None and np.allclose(xi, critical_xi)
        ax.plot(
            tr[:, 0],
            tr[:, 1],
            color=color,
            lw=2.4 if is_critical else 0.9,
            alpha=0.78 if is_critical else 0.16,
            zorder=9 if is_critical else 6,
        )
        footprint_stride = max(len(tr) // (7 if is_critical else 5), 1)
        footprint_idx = range(0, len(tr), footprint_stride)
        for idx in footprint_idx:
            draw_vehicle_footprint(
                ax,
                center_x=float(tr[idx, 0]),
                center_y=float(tr[idx, 1]),
                heading=float(tr[idx, 3]),
                length=geom.length,
                width=geom.width,
                color=color,
                alpha=(
                    CROSS_FOOTPRINT_CRITICAL_ALPHA
                    if is_critical
                    else CROSS_FOOTPRINT_ALPHA
                ),
                linewidth=1.0 if is_critical else 0.25,
                zorder=11 if is_critical else 7,
                gid="cross_vehicle_critical"
                if is_critical
                else "cross_vehicle_sample",
            )
        if is_critical:
            stride = max(len(tr) // 5, 1)
            ax.scatter(
                tr[::stride, 0],
                tr[::stride, 1],
                s=10,
                color=color,
                alpha=0.9,
                zorder=10,
            )
    _draw_cross_traffic_legend(ax, mode_colors)


def draw_intersection_metrics(
    ax,
    scene: SignalizedScene,
    ego_traj,
    *,
    dt=0.05,
    time_offset_s=0.0,
):
    """Draw compact display-only signalized-intersection metrics."""
    metrics = estimate_visual_metrics(
        scene,
        ego_traj,
        n_samples=40,
        dt=dt,
        time_offset_s=time_offset_s,
    )
    text = (
        f"mode {metrics['mode']}\n"
        f"clear {metrics['min_clearance']:.1f} m\n"
        f"risk q90 {metrics['risk_quantile']:.1f}\n"
        f"red legal {metrics['red_legal']}\n"
        f"no block {metrics['no_blocking']}"
    )
    ax.text(
        0.014,
        0.965,
        text,
        transform=ax.transAxes,
        color="#e5e7eb",
        fontsize=7.5,
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "#020617",
            "edgecolor": "#64748b",
            "alpha": 0.84,
        },
        zorder=20,
    )

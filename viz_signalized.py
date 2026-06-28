"""NuPlan-style semantic rendering for signalized-intersection scenarios."""

import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import numpy as np

from config import SUB_STEPS


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


def _add_patch(ax, patch, gid, zorder):
    patch.set_gid(gid)
    patch.set_zorder(zorder)
    ax.add_patch(patch)
    return patch


def _draw_vehicle_footprint(
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
    """Draw a top-down oriented vehicle footprint centered on a trajectory state."""
    patch = mpatches.Rectangle(
        (center_x - 0.5 * length, center_y - 0.5 * width),
        length,
        width,
        facecolor=color,
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha,
    )
    transform = (
        mtransforms.Affine2D().rotate_around(center_x, center_y, heading)
        + ax.transData
    )
    patch.set_transform(transform)
    return _add_patch(ax, patch, gid, zorder)


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


def semantic_layer_summary(scenario):
    """Return stable semantic-layer counts for tests and report diagnostics."""
    from costs import signalized_intersection as si
    from scenarios import signalized_intersection as sig

    return {
        "horizontal_lanes": len(scenario.road.lane_centers),
        "vertical_lanes": len(sig.CROSS_ROAD_LANE_CENTERS),
        "crosswalks": 2,
        "stop_lines": 1,
        "direction_arrows": len(scenario.road.lane_centers)
        + len(sig.CROSS_ROAD_LANE_CENTERS),
        "risk_cloud_samples": int(si.DEV_N_SAMPLES),
        "traffic_signals": 1,
    }


def _draw_lane_arrows(ax, scenario, x_min, x_max, cross_min, cross_max):
    from scenarios import signalized_intersection as sig

    arrow_kw = {
        "width": 0.08,
        "head_width": 0.7,
        "head_length": 1.0,
        "length_includes_head": True,
        "color": "#dbeafe",
        "alpha": 0.55,
        "zorder": 5,
    }
    x_arrow = x_min + 8.0
    for y in scenario.road.lane_centers:
        direction = 1.0 if y < 0.0 else -1.0
        ax.arrow(
            x_arrow if direction > 0 else x_max - 8.0,
            y,
            5.0 * direction,
            0.0,
            **arrow_kw,
        )
    y_arrow = cross_min + 7.0
    for x in sig.CROSS_ROAD_LANE_CENTERS:
        direction = 1.0 if x < sig.INTERSECTION_CENTER_X else -1.0
        ax.arrow(
            x,
            y_arrow if direction > 0 else cross_max - 7.0,
            0.0,
            5.0 * direction,
            **arrow_kw,
        )


def _draw_crosswalks(ax, scenario):
    from costs import signalized_intersection as si

    road_min = scenario.road.road_min_y
    road_max = scenario.road.road_max_y
    width = road_max - road_min
    stripe_h = 0.55
    for x0 in (si.STOP_LINE_X + 0.8, si.INTERSECTION_EXIT_X - 1.6):
        for idx in range(5):
            y = road_min + 0.6 + idx * (width - 1.2) / 5.0
            _add_patch(
                ax,
                mpatches.Rectangle(
                    (x0, y),
                    1.0,
                    stripe_h,
                    facecolor=CROSSWALK,
                    edgecolor="none",
                    alpha=0.82,
                ),
                "crosswalk",
                7,
            )


def _signal_phase(scenario, elapsed_time_s):
    """Return active signal phase from scenario context timing."""
    if len(scenario.context_values) < 2:
        return "green"
    yellow_start_s, red_start_s = scenario.context_values[:2]
    elapsed = float(elapsed_time_s)
    if elapsed < float(yellow_start_s):
        return "green"
    if elapsed < float(red_start_s):
        return "yellow"
    return "red"


def draw_signalized_scene(ax, scenario, x_min, x_max, elapsed_time_s=0.0):
    """Draw a top-down semantic intersection map."""
    from costs import signalized_intersection as si
    from scenarios import signalized_intersection as sig

    road_min = scenario.road.road_min_y
    road_max = scenario.road.road_max_y
    lane_w = scenario.road.lane_width
    cross_min = -22.0
    cross_max = 22.0
    cross_road_min = min(sig.CROSS_ROAD_LANE_CENTERS) - 0.5 * lane_w
    cross_road_max = max(sig.CROSS_ROAD_LANE_CENTERS) + 0.5 * lane_w

    ax.set_facecolor(MAP_BG)
    ax.set_ylim(cross_min, cross_max)

    _add_patch(
        ax,
        mpatches.Rectangle(
            (x_min, road_min),
            x_max - x_min,
            road_max - road_min,
            facecolor=ROAD_FILL,
            edgecolor=LANE_EDGE,
            linewidth=1.3,
            alpha=1.0,
        ),
        "road_surface",
        1,
    )
    _add_patch(
        ax,
        mpatches.Rectangle(
            (cross_road_min, cross_min),
            cross_road_max - cross_road_min,
            cross_max - cross_min,
            facecolor=ROAD_FILL,
            edgecolor=LANE_EDGE,
            linewidth=1.3,
            alpha=1.0,
        ),
        "road_surface",
        1,
    )

    for idx, y in enumerate(scenario.road.lane_centers):
        _add_patch(
            ax,
            mpatches.Rectangle(
                (x_min, y - 0.5 * lane_w),
                x_max - x_min,
                lane_w,
                facecolor=LANE_FILL_A if idx % 2 == 0 else LANE_FILL_B,
                edgecolor="none",
                alpha=0.62,
            ),
            "lane_polygon",
            2,
        )
    for idx, x in enumerate(sig.CROSS_ROAD_LANE_CENTERS):
        _add_patch(
            ax,
            mpatches.Rectangle(
                (x - 0.5 * lane_w, cross_min),
                lane_w,
                cross_max - cross_min,
                facecolor=LANE_FILL_B if idx % 2 == 0 else LANE_FILL_A,
                edgecolor="none",
                alpha=0.55,
            ),
            "lane_polygon",
            2,
        )

    _add_patch(
        ax,
        mpatches.Rectangle(
            (si.INTERSECTION_ENTRY_X, road_min),
            si.INTERSECTION_EXIT_X - si.INTERSECTION_ENTRY_X,
            road_max - road_min,
            facecolor=CONFLICT,
            edgecolor="#fde68a",
            linewidth=1.1,
            alpha=0.22,
        ),
        "conflict_box",
        4,
    )

    for y0, y1 in zip(scenario.road.lane_centers[:-1], scenario.road.lane_centers[1:]):
        lane_mid = 0.5 * (y0 + y1)
        ax.plot(
            [x_min, x_max],
            [lane_mid, lane_mid],
            color=DIVIDER,
            lw=0.9,
            ls=(0, (6, 6)),
            alpha=0.72,
            zorder=5,
        )
    for x0, x1 in zip(sig.CROSS_ROAD_LANE_CENTERS[:-1], sig.CROSS_ROAD_LANE_CENTERS[1:]):
        lane_mid = 0.5 * (x0 + x1)
        ax.plot(
            [lane_mid, lane_mid],
            [cross_min, cross_max],
            color=DIVIDER,
            lw=0.9,
            ls=(0, (6, 6)),
            alpha=0.72,
            zorder=5,
        )

    ax.plot(
        [si.STOP_LINE_X, si.STOP_LINE_X],
        [road_min, road_max],
        color=STOP_LINE,
        lw=3.0,
        zorder=8,
    )
    _draw_crosswalks(ax, scenario)
    _draw_lane_arrows(ax, scenario, x_min, x_max, cross_min, cross_max)

    lamp_x = si.STOP_LINE_X + 2.2
    lamp_y = road_max + 1.1
    phase = _signal_phase(scenario, elapsed_time_s)
    ax.plot(
        [lamp_x, lamp_x],
        [road_max + 0.1, lamp_y - 0.85],
        color=SIGNAL_POST,
        lw=1.7,
        zorder=10,
    )
    _add_patch(
        ax,
        mpatches.FancyBboxPatch(
            (lamp_x - 0.58, lamp_y - 0.92),
            1.16,
            1.84,
            boxstyle="round,pad=0.08,rounding_size=0.18",
            facecolor="#0f172a",
            edgecolor=LANE_EDGE,
            linewidth=0.75,
        ),
        "traffic_signal",
        11,
    )
    lamp_specs = (
        ("red", SIGNAL_RED, lamp_y + 0.55),
        ("yellow", SIGNAL_YELLOW, lamp_y),
        ("green", SIGNAL_GREEN, lamp_y - 0.55),
    )
    for name, color, y in lamp_specs:
        active = name == phase
        _add_patch(
            ax,
            mpatches.Circle(
                (lamp_x, y),
                0.27 if active else 0.21,
                facecolor=color,
                edgecolor="#f8fafc" if active else "none",
                linewidth=0.65 if active else 0.0,
                alpha=0.96 if active else 0.22,
            ),
            f"traffic_signal_{name}_{'active' if active else 'inactive'}",
            12 if active else 11,
        )


def draw_cross_traffic_cloud(ax, scenario, ego_traj=None):
    """Draw exogenous cross-traffic samples with mode-coded probability paths."""
    from costs import signalized_intersection as si
    import jax.numpy as jnp

    samples = np.asarray(si._cross_traffic_noise(None, (30,)))
    n_steps = scenario.control_horizon * SUB_STEPS
    critical_xi = None
    if ego_traj is not None:
        metrics = si.estimate_visual_metrics(ego_traj, n_samples=30)
        critical_xi = np.asarray(metrics["critical_sample"], dtype=float)
    mode_colors = {
        "obey": "#38bdf8",
        "yellow_rush": "#facc15",
        "red_run": "#fb7185",
    }
    geom = scenario.vehicle_geometry
    for xi in samples:
        mode = int(xi[si.XI_MODE])
        color = {
            si.MODE_OBEY: mode_colors["obey"],
            si.MODE_YELLOW_RUSH: mode_colors["yellow_rush"],
            si.MODE_RED_RUN: mode_colors["red_run"],
        }.get(mode, "#ffffff")
        tr = np.asarray(si._cross_traj_for_xi(jnp.asarray(xi), n_steps))
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
            _draw_vehicle_footprint(
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
    ego_traj,
    *,
    red_start_s=None,
    dt=None,
    time_offset_s=0.0,
):
    """Draw compact display-only signalized-intersection metrics."""
    from costs import signalized_intersection as si

    kwargs = {}
    if red_start_s is not None:
        kwargs["red_start_s"] = red_start_s
    if dt is not None:
        kwargs["dt"] = dt
    metrics = si.estimate_visual_metrics(
        ego_traj,
        n_samples=40,
        time_offset_s=time_offset_s,
        **kwargs,
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

"""
viz_utils.py — 公路场景可视化工具

提供：
  ROAD_BG, EGO_CLR, EGO_DARK, FRONT_CLR, REAR_CLR, HIST_CLR   配色常量
  _draw_rect(ax, cx, cy, theta, length, width, ...)           绘制旋转矩形车辆
  _draw_road(ax, x_min, x_max)                                 绘制公路背景
  render_agents_panel(...)                                     渲染通用 agent 场景面板
"""

import matplotlib.patches as mpatches
import numpy as np

from config import LANE_W, N_LANES, SUB_STEPS
from visualization.scene_renderer import (
    LineLayer,
    SceneRenderSpec,
    VehicleLayer,
    draw_vehicle_footprint,
    render_scene,
)


# ══════════════════════════════════════════════════════════════════════════════
#  配色
# ══════════════════════════════════════════════════════════════════════════════
ROAD_BG   = "#10192b"
EGO_CLR   = "#2ecc71"
EGO_DARK  = "#1a8a4a"
FRONT_CLR = "#1f77b4"
FRONT_DRK = "#13496e"
REAR_CLR  = "#d1495b"
REAR_DRK  = "#8a2e38"
HIST_CLR  = "#ffd166"
AGENT_COLORS = (
    (EGO_CLR, EGO_DARK),
    (FRONT_CLR, FRONT_DRK),
    (REAR_CLR, REAR_DRK),
    ("#f4a261", "#9d5f28"),
    ("#b388eb", "#6d4ca2"),
    ("#2ec4b6", "#15766e"),
)


# ══════════════════════════════════════════════════════════════════════════════
#  基础绘图工具
# ══════════════════════════════════════════════════════════════════════════════

def _draw_rect(ax, cx, cy, theta, length, width,
               color, alpha=1.0, zorder=5, edge_lw=0.8):
    """在 ax 上绘制一个以 (cx, cy) 为中心、偏航角 theta（rad）的车辆矩形。"""
    return draw_vehicle_footprint(
        ax,
        VehicleLayer(
            center=(float(cx), float(cy)),
            heading=float(theta),
            length=float(length),
            width=float(width),
            facecolor=color,
            edgecolor="white" if edge_lw > 0 else "none",
            linewidth=float(edge_lw),
            alpha=float(alpha),
            zorder=float(zorder),
        ),
    )


def _draw_road(ax, x_min, x_max, road=None):
    """绘制双车道公路背景（深色背景 + 白色道路边界与虚线路中线）。"""
    if road is None:
        lane_centers = tuple(i * LANE_W for i in range(N_LANES))
        road_min = -0.5 * LANE_W
        road_max = (N_LANES - 0.5) * LANE_W
    else:
        lane_centers = road.lane_centers
        road_min = road.road_min_y
        road_max = road.road_max_y

    lines = [
        LineLayer(
            x=(float(x_min), float(x_max)),
            y=(float(road_min), float(road_min)),
            color="white",
            linewidth=2.0,
            zorder=1,
            gid="road_boundary",
        ),
        LineLayer(
            x=(float(x_min), float(x_max)),
            y=(float(road_max), float(road_max)),
            color="white",
            linewidth=2.0,
            zorder=1,
            gid="road_boundary",
        ),
    ]
    for y0, y1 in zip(lane_centers[:-1], lane_centers[1:]):
        lane_mid = 0.5 * (y0 + y1)
        x = x_min
        while x < x_max:
            lines.append(
                LineLayer(
                    x=(float(x), float(x + 4.5)),
                    y=(float(lane_mid), float(lane_mid)),
                    color="white",
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.7,
                    zorder=1,
                    gid="lane_divider",
                )
            )
            x += 9.0
    render_scene(
        ax,
        SceneRenderSpec(
            facecolor=ROAD_BG,
            xlim=(float(x_min), float(x_max)),
            ylim=(float(road_min - 0.8), float(road_max + 0.8)),
            aspect="equal",
            lines=tuple(lines),
            hide_ticks=True,
            hide_spines=True,
        ),
    )


def _smooth_xy(points, samples_per_segment=8):
    """Catmull-Rom smoothing for display only; endpoints are preserved."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 4:
        return pts
    padded = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        for u in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
            u2 = u * u
            u3 = u2 * u
            out.append(0.5 * (
                (2.0 * p1) +
                (-p0 + p2) * u +
                (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2 +
                (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
            ))
    out.append(pts[-1])
    return np.asarray(out)


# ══════════════════════════════════════════════════════════════════════════════
#  场景面板渲染
# ══════════════════════════════════════════════════════════════════════════════

def render_agents_panel(
    ax,
    scenario,
    states_by_agent,
    trajectories_by_agent=None,
    history_by_agent=None,
    focus_agent=None,
    x_win=58.0,
    title="",
    show_step=3,
    elapsed_time_s=0.0,
):
    """Render a road panel for any scenario described by AgentSpec."""
    focus_agent = focus_agent or scenario.agent_names[0]
    focus_state = states_by_agent[focus_agent]
    geometry = scenario.vehicle_geometry
    ego_x = float(focus_state[0])
    if scenario.name.startswith("signalized_intersection"):
        from costs import signalized_intersection as si

        x_win = max(x_win, 76.0)
        center = 0.5 * (si.STOP_LINE_X + si.INTERSECTION_EXIT_X)
        x_min = center - 0.5 * x_win
        x_max = center + 0.5 * x_win
    else:
        x_min = ego_x - 8.0
        x_max = x_min + x_win
    _draw_road(ax, x_min, x_max, scenario.road)
    if scenario.name.startswith("signalized_intersection"):
        import viz_signalized

        viz_signalized.draw_signalized_scene(
            ax,
            scenario,
            x_min,
            x_max,
            elapsed_time_s=elapsed_time_s,
        )

    micro_per_macro = SUB_STEPS
    color_by_agent = {
        agent.name: AGENT_COLORS[idx % len(AGENT_COLORS)]
        for idx, agent in enumerate(scenario.agents)
    }

    if history_by_agent is not None:
        hist = history_by_agent.get(focus_agent, [])
        if len(hist) >= 2:
            hist_arr = np.asarray(hist)
            hist_xy = _smooth_xy(hist_arr[:, :2], samples_per_segment=6)
            ax.plot(hist_xy[:, 0], hist_xy[:, 1], color=HIST_CLR,
                    lw=2.0, alpha=0.95, zorder=3)
            ax.scatter(hist_arr[:, 0], hist_arr[:, 1], s=9, color=HIST_CLR,
                       edgecolors="none", alpha=0.85, zorder=3)

    if trajectories_by_agent is not None:
        for agent in scenario.agents:
            name = agent.name
            _, dark = color_by_agent[name]
            trajs = trajectories_by_agent.get(name)
            if trajs is None:
                continue
            for tr in trajs:
                ax.plot(tr[:, 0], tr[:, 1], color=dark, lw=0.75,
                        ls="-" if name == focus_agent else "--",
                        alpha=0.10 if name == focus_agent else 0.32,
                        zorder=2)
                for h in range(2, scenario.control_horizon + 1, show_step):
                    mi = min(h * micro_per_macro, len(tr) - 1)
                    s = tr[mi]
                    if x_min - 6 < s[0] < x_max + 6:
                        _draw_rect(
                            ax, float(s[0]), float(s[1]), float(s[3]),
                            geometry.length * 0.85, geometry.width * 0.85,
                            dark, alpha=0.08, zorder=2, edge_lw=0.0,
                        )

        best_focus = trajectories_by_agent.get(focus_agent, [])
        if best_focus:
            best = best_focus[0]
            best_xy = _smooth_xy(best[:, :2], samples_per_segment=8)
            ax.plot(best_xy[:, 0], best_xy[:, 1], color="#9ff3bd",
                    lw=2.2, alpha=0.95, zorder=4)
            if scenario.name.startswith("signalized_intersection"):
                import viz_signalized

                viz_signalized.draw_cross_traffic_cloud(ax, scenario, ego_traj=best)
                viz_signalized.draw_intersection_metrics(
                    ax,
                    best,
                    red_start_s=float(scenario.context_values[1]),
                    time_offset_s=elapsed_time_s,
                )
    elif scenario.name.startswith("signalized_intersection"):
        import viz_signalized

        viz_signalized.draw_cross_traffic_cloud(ax, scenario)

    for idx, agent in enumerate(scenario.agents):
        state = states_by_agent[agent.name]
        color, _ = color_by_agent[agent.name]
        x = float(state[0])
        if x_min - 6 < x < x_max + 6:
            _draw_rect(
                ax,
                x,
                float(state[1]),
                float(state[3]),
                geometry.length,
                geometry.width,
                color,
                zorder=7 if agent.name == focus_agent else 6,
                edge_lw=1.2 if agent.name == focus_agent else 0.8,
            )

    if title:
        ax.set_title(title, color="white", fontsize=10, pad=5, fontweight="bold")


def generic_legend_handles(scenario):
    handles = []
    for idx, agent in enumerate(scenario.agents):
        color, _ = AGENT_COLORS[idx % len(AGENT_COLORS)]
        handles.append(
            mpatches.Patch(
                facecolor=color,
                edgecolor="white",
                lw=0.8,
                label=f"{agent.name} ({agent.role})",
            )
        )
    handles.append(
        mpatches.Patch(
            facecolor=EGO_DARK,
            alpha=0.55,
            label="MGIGO trajectory samples",
        )
    )
    handles.append(
        mpatches.Patch(
            facecolor=HIST_CLR,
            alpha=0.85,
            label="Executed ego path",
        )
    )
    return handles

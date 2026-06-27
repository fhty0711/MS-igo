"""
viz_utils.py — 公路场景可视化工具

提供：
  ROAD_BG, EGO_CLR, EGO_DARK, FRONT_CLR, REAR_CLR, HIST_CLR   配色常量
  _draw_rect(ax, cx, cy, theta, length, width, ...)           绘制旋转矩形车辆
  _draw_road(ax, x_min, x_max)                                 绘制公路背景
  render_agents_panel(...)                                     渲染通用 agent 场景面板
"""

import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import numpy as np

from config import LANE_W, N_LANES, SUB_STEPS


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
    rect = mpatches.Rectangle(
        (-length / 2, -width / 2), length, width,
        facecolor=color,
        edgecolor="white" if edge_lw > 0 else "none",
        linewidth=edge_lw, alpha=alpha,
    )
    t_ = transforms.Affine2D().rotate(theta).translate(cx, cy) + ax.transData
    rect.set_transform(t_)
    rect.set_zorder(zorder)
    ax.add_patch(rect)


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

    ax.set_facecolor(ROAD_BG)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(road_min - 0.8, road_max + 0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(road_min, color="white", lw=2.0, zorder=1)
    ax.axhline(road_max, color="white", lw=2.0, zorder=1)
    for y0, y1 in zip(lane_centers[:-1], lane_centers[1:]):
        lane_mid = 0.5 * (y0 + y1)
        x = x_min
        while x < x_max:
            ax.plot([x, x + 4.5], [lane_mid, lane_mid],
                    color="white", lw=1.2, ls="--", alpha=0.7, zorder=1)
            x += 9.0
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _draw_signalized_intersection(ax, scenario, x_min, x_max):
    """Draw intersection-specific geometry over the generic ego approach road."""
    from costs import signalized_intersection as si

    road_min = scenario.road.road_min_y
    road_max = scenario.road.road_max_y
    cross_min = -18.0
    cross_max = 18.0
    ax.set_ylim(cross_min, cross_max)
    ax.add_patch(
        mpatches.Rectangle(
            (si.INTERSECTION_ENTRY_X, cross_min),
            si.INTERSECTION_EXIT_X - si.INTERSECTION_ENTRY_X,
            cross_max - cross_min,
            facecolor="#26384f",
            edgecolor="#ffffff",
            linewidth=1.0,
            alpha=0.45,
            zorder=1,
        )
    )
    ax.plot(
        [si.STOP_LINE_X, si.STOP_LINE_X],
        [road_min - 0.5, road_max + 0.5],
        color="#ffdf6e",
        lw=2.4,
        zorder=4,
    )
    ax.plot(
        [si.CROSS_LANE_X, si.CROSS_LANE_X],
        [cross_min, cross_max],
        color="#d9e6f2",
        lw=1.0,
        ls="--",
        alpha=0.65,
        zorder=1,
    )
    lamp_x = min(max(si.STOP_LINE_X + 2.5, x_min + 4.0), x_max - 4.0)
    lamp_y = road_max + 0.45
    ax.scatter([lamp_x], [lamp_y], s=80, color="#ffd166", edgecolors="white", zorder=9)
    ax.text(
        lamp_x + 1.0,
        lamp_y,
        "yellow -> red",
        color="white",
        fontsize=7,
        va="center",
        zorder=9,
    )


def _draw_cross_traffic_cloud(ax, scenario, ego_traj=None):
    """Draw exogenous cross-traffic behavior samples for intersection scenarios."""
    from costs import signalized_intersection as si
    import jax.numpy as jnp

    samples = np.asarray(si._cross_traffic_noise(None, (30,)))
    n_steps = scenario.control_horizon * SUB_STEPS
    critical_xi = None
    if ego_traj is not None:
        metrics = si.estimate_visual_metrics(ego_traj, n_samples=30)
        critical_xi = np.asarray(metrics["critical_sample"], dtype=float)
    for xi in samples:
        mode = int(xi[si.XI_MODE])
        color = {
            si.MODE_OBEY: "#4cc9f0",
            si.MODE_YELLOW_RUSH: "#ffd166",
            si.MODE_RED_RUN: "#ef476f",
        }.get(mode, "#ffffff")
        tr = np.asarray(si._cross_traj_for_xi(jnp.asarray(xi), n_steps))
        is_critical = critical_xi is not None and np.allclose(xi, critical_xi)
        ax.plot(
            tr[:, 0],
            tr[:, 1],
            color=color,
            lw=2.0 if is_critical else 0.8,
            alpha=0.65 if is_critical else 0.16,
            zorder=5 if is_critical else 2,
        )


def _draw_intersection_metrics(ax, ego_traj):
    """Draw display-only signalized-intersection metrics."""
    from costs import signalized_intersection as si

    metrics = si.estimate_visual_metrics(ego_traj, n_samples=40)
    text = (
        f"mode: {metrics['mode']}\n"
        f"min clearance: {metrics['min_clearance']:.1f} m\n"
        f"risk q90: {metrics['risk_quantile']:.1f}\n"
        f"red legal: {metrics['red_legal']}\n"
        f"no blocking: {metrics['no_blocking']}"
    )
    ax.text(
        0.012,
        0.965,
        text,
        transform=ax.transAxes,
        color="white",
        fontsize=7.5,
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#0a1120",
            "edgecolor": "#445566",
            "alpha": 0.82,
        },
        zorder=12,
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
        _draw_signalized_intersection(ax, scenario, x_min, x_max)

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
                _draw_cross_traffic_cloud(ax, scenario, ego_traj=best)
                _draw_intersection_metrics(ax, best)
    elif scenario.name.startswith("signalized_intersection"):
        _draw_cross_traffic_cloud(ax, scenario)

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

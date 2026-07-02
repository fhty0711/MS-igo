"""Warm-start helpers for Frenet B-spline control points."""

from __future__ import annotations

import numpy as np


def tangent_control_points(gen, s0: float, s_dot0: float, d0: float):
    """Free control points for straight constant-speed motion."""
    ctrl_s = np.asarray(s0 + s_dot0 * np.asarray(gen.greville[3:]), dtype=np.float32)
    ctrl_d = np.full((gen.n_free,), float(d0), dtype=np.float32)
    return ctrl_s, ctrl_d


def initial_component_means(gen, s0: float, s_dot0: float, d0: float, n_components: int):
    """Return per-block initial component means for ctrl_s and ctrl_d blocks."""
    ctrl_s, ctrl_d = tangent_control_points(gen, s0, s_dot0, d0)
    return (
        tuple(tuple(float(v) for v in ctrl_s) for _ in range(n_components)),
        tuple(tuple(float(v) for v in ctrl_d) for _ in range(n_components)),
    )

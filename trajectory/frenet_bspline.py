"""Frenet-frame quintic B-spline trajectory generator."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import lax

from config import MAX_SPEED
from .reference_path import ReferencePath


class FrenetBSplineTrajectory:
    """Clamped quintic B-spline in Frenet (s, d) coordinates."""

    def __init__(self, basis_path: Path | str, ref_path: ReferencePath):
        data = np.load(str(basis_path))
        self.B = jnp.array(data["B"])
        self.dB = jnp.array(data["dB"])
        self.d2B = jnp.array(data["d2B"])
        self.d3B = jnp.array(data["d3B"])
        self.d4B = jnp.array(data["d4B"])
        self.greville = jnp.array(data["greville"])
        self.T = int(self.B.shape[0])
        self.n_ctrl = int(self.B.shape[1])
        self.dt = float(data["dt"])
        self.total_time = float(data["total_time"])
        self.degree = int(data["degree"])
        self.dt_knot = float(data["dt_knot"])
        self.n_free = self.n_ctrl - 3
        self.ref_path = ref_path

    def _clamped_3pts(self, x0, v0, a0):
        p0 = x0
        p1 = p0 + (self.dt_knot / self.degree) * v0
        p2 = 3.0 * p1 - 2.0 * p0 + (self.dt_knot**2 / 10.0) * a0
        return p0, p1, p2

    def _speed_limited_ctrl_s_free(self, ctrl_s_free, p2_s, max_speed=MAX_SPEED):
        """Project raw longitudinal control points to a monotone speed-bounded polygon."""
        times = self.greville[3:]
        prev_times = jnp.concatenate([self.greville[2:3], times[:-1]])
        max_steps = float(max_speed) * jnp.maximum(times - prev_times, 1e-3)

        def body(prev_s, item):
            raw_s, max_step = item
            next_s = jnp.clip(raw_s, prev_s, prev_s + max_step)
            return next_s, next_s

        _last, limited = lax.scan(body, p2_s, (ctrl_s_free, max_steps))
        return limited

    def evaluate(
        self,
        ctrl_s_free,
        ctrl_d_free,
        s0,
        s_dot0,
        s_ddot0,
        d0,
        d_dot0,
        d_ddot0,
    ):
        p0_s, p1_s, p2_s = self._clamped_3pts(s0, s_dot0, s_ddot0)
        p0_d, p1_d, p2_d = self._clamped_3pts(d0, d_dot0, d_ddot0)
        ctrl_s_free = self._speed_limited_ctrl_s_free(ctrl_s_free, p2_s)
        ctrl_s = jnp.concatenate(
            [jnp.array([p0_s]), jnp.array([p1_s]), jnp.array([p2_s]), ctrl_s_free],
            axis=0,
        )
        ctrl_d = jnp.concatenate(
            [jnp.array([p0_d]), jnp.array([p1_d]), jnp.array([p2_d]), ctrl_d_free],
            axis=0,
        )
        return (
            jnp.dot(self.B, ctrl_s),
            jnp.dot(self.B, ctrl_d),
            jnp.dot(self.dB, ctrl_s),
            jnp.dot(self.dB, ctrl_d),
            jnp.dot(self.d2B, ctrl_s),
            jnp.dot(self.d2B, ctrl_d),
            jnp.dot(self.d3B, ctrl_s),
            jnp.dot(self.d3B, ctrl_d),
        )

    def to_cartesian(self, s, d):
        return self.ref_path.frenet_to_cartesian(s, d)

    def to_vehicle_states(
        self,
        s,
        d,
        s_dot,
        d_dot,
        s_ddot,
        d_ddot,
        s_dddot,
        d_dddot,
        wheel_base=2.8,
    ):
        _x_ref, _y_ref, theta_r, kappa_r = self.ref_path.evaluate(s)
        vt = (1.0 - d * kappa_r) * s_dot
        vn = d_dot
        v2 = vt**2 + vn**2
        v = jnp.sqrt(v2)
        vs = v + 1e-6
        dpsi = jnp.arctan2(vn, vt)
        psi = theta_r + dpsi
        cos_dpsi = vt / vs
        sin_dpsi = vn / vs

        vt_dot = (1.0 - d * kappa_r) * s_ddot - kappa_r * s_dot * d_dot
        a_t = vt_dot - vn * kappa_r * s_dot
        a_n = d_ddot + kappa_r * vt * s_dot
        a_long = a_t * cos_dpsi + a_n * sin_dpsi
        a_lat = -a_t * sin_dpsi + a_n * cos_dpsi
        j_long = s_dddot * cos_dpsi + d_dddot * sin_dpsi
        j_lat = -s_dddot * sin_dpsi + d_dddot * cos_dpsi

        ddpsi_dt = (vt * d_ddot - vn * vt_dot) / jnp.maximum(v2, 1e-6)
        dpsi_dt = kappa_r * s_dot + ddpsi_dt
        curvature = dpsi_dt / vs
        steer = jnp.arctan(curvature * wheel_base)
        x, y = self.to_cartesian(s, d)
        return jnp.stack(
            [x, y, v, psi, a_long, a_lat, j_long, j_lat, steer],
            axis=-1,
        )

    def to_compressed_states(self, vehicle_states):
        """Return [x, y, v, psi, a_long, steer] for existing igo costs/viz."""
        return jnp.stack(
            [
                vehicle_states[:, 0],
                vehicle_states[:, 1],
                vehicle_states[:, 2],
                vehicle_states[:, 3],
                vehicle_states[:, 4],
                vehicle_states[:, 8],
            ],
            axis=-1,
        )

    def evaluate_plan(self, ctrl_s_free, ctrl_d_free, ctx):
        frenet = self.evaluate(
            ctrl_s_free,
            ctrl_d_free,
            ctx["s0"],
            ctx["s_dot0"],
            ctx["s_ddot0"],
            ctx["d0"],
            ctx["d_dot0"],
            ctx["d_ddot0"],
        )
        st = self.to_vehicle_states(*frenet)
        x, y = self.to_cartesian(frenet[0], frenet[1])
        return frenet, st, self.to_compressed_states(st), (x, y)

"""Reference-path helpers for Frenet trajectory parameterizations."""

from __future__ import annotations

import jax.numpy as jnp


class ReferencePath:
    """Base reference path interface."""

    def evaluate(self, s):
        raise NotImplementedError

    def frenet_to_cartesian(self, s, d):
        x_ref, y_ref, theta, _kappa = self.evaluate(s)
        nx = -jnp.sin(theta)
        ny = jnp.cos(theta)
        return x_ref + d * nx, y_ref + d * ny


class StraightReference(ReferencePath):
    """Straight reference path along +x with zero curvature."""

    def __init__(self, y0: float = 0.0):
        self.y0 = float(y0)

    def evaluate(self, s):
        s = jnp.asarray(s)
        x = s
        y = jnp.zeros_like(s) + self.y0
        theta = jnp.zeros_like(s)
        kappa = jnp.zeros_like(s)
        return x, y, theta, kappa

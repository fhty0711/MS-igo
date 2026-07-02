"""Tests for Frenet B-spline trajectory helpers."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax.numpy as jnp
import numpy as np


def test_straight_reference_path_maps_frenet_to_cartesian():
    from trajectory.reference_path import StraightReference

    ref = StraightReference()
    s = jnp.array([0.0, 10.0, 25.0])
    d = jnp.array([-1.75, 0.0, 2.0])
    x, y = ref.frenet_to_cartesian(s, d)

    if not np.allclose(np.asarray(x), np.asarray(s)):
        raise AssertionError((x, s))
    if not np.allclose(np.asarray(y), np.asarray(d)):
        raise AssertionError((y, d))


def test_frenet_bspline_loads_basis_asset():
    from trajectory.frenet_bspline import FrenetBSplineTrajectory
    from trajectory.reference_path import StraightReference

    basis = Path("trajectory/assets/bspline_basis.npz")
    gen = FrenetBSplineTrajectory(basis, StraightReference())

    if gen.n_ctrl != 12:
        raise AssertionError(gen.n_ctrl)
    if gen.n_free != 9:
        raise AssertionError(gen.n_free)
    if gen.T != 100:
        raise AssertionError(gen.T)


if __name__ == "__main__":
    test_straight_reference_path_maps_frenet_to_cartesian()
    test_frenet_bspline_loads_basis_asset()

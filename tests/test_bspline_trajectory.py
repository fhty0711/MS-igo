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


def test_bspline_rollout_returns_dense_ego_states():
    from trajectory.frenet_bspline import FrenetBSplineTrajectory
    from trajectory.reference_path import StraightReference
    from trajectory.rollout import bspline_ego_rollout
    from trajectory.warmstart import tangent_control_points

    gen = FrenetBSplineTrajectory("trajectory/assets/bspline_basis.npz", StraightReference())
    ctrl_s, ctrl_d = tangent_control_points(gen, s0=5.0, s_dot0=8.0, d0=0.5)
    ctx = {
        "s0": 5.0,
        "s_dot0": 8.0,
        "s_ddot0": 0.0,
        "d0": 0.5,
        "d_dot0": 0.0,
        "d_ddot0": 0.0,
    }

    traj = bspline_ego_rollout(gen, jnp.asarray(ctrl_s), jnp.asarray(ctrl_d), ctx)

    if traj.shape != (gen.T, 6):
        raise AssertionError(traj.shape)
    if not np.all(np.isfinite(np.asarray(traj))):
        raise AssertionError(traj)
    if not np.allclose(np.asarray(traj[0, :2]), np.array([5.0, 0.5]), atol=1e-5):
        raise AssertionError(traj[0, :2])


if __name__ == "__main__":
    test_straight_reference_path_maps_frenet_to_cartesian()
    test_frenet_bspline_loads_basis_asset()
    test_bspline_rollout_returns_dense_ego_states()

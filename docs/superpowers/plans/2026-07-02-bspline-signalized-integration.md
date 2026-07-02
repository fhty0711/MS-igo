# B-Spline Signalized Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a complete B-spline trajectory-parameterized signalized-intersection path alongside the existing acc/steer path, reusing the current `igo` scenario/planner/cost/report architecture and the current `costs/constraint_dsl.py` prioritized cost layering.

**Architecture:** Keep the existing `signalized_intersection` acc/steer benchmark unchanged as the validated baseline. Add a new trajectory module copied and adapted from `MGIGO/Cartest`, then add a new `signalized_intersection_bspline` scenario and cost profile that decode MGIGO blocks as Frenet B-spline control points. The B-spline path returns the same compressed vehicle state shape `[x, y, v, psi, acc, steer]` expected by existing signalized constraints and visualization.

**Tech Stack:** Python, JAX, NumPy, SciPy-generated B-spline basis (`bspline_basis.npz`), current `igo.MPC_G_MS.mmog_igo_rne_blocks_solver`, current `costs.constraint_dsl`, Matplotlib, WSL2 Ubuntu + CUDA for full MGIGO runs.

---

## File Structure

- Create `trajectory/__init__.py`: public exports for B-spline helpers.
- Create `trajectory/reference_path.py`: straight reference path API adapted from Cartest, with `evaluate()` and `frenet_to_cartesian()`.
- Create `trajectory/frenet_bspline.py`: Frenet quintic B-spline evaluator adapted from Cartest, returning Frenet arrays, Cartesian positions, full vehicle states, and compressed 6D states.
- Create `trajectory/warmstart.py`: Greville/tangent initialization helpers for B-spline component means.
- Add `trajectory/assets/bspline_basis.npz`: copied from `MGIGO/Cartest/bspline_basis.npz`.
- Modify `scenarios/spec.py`: extend `DecisionSpec.kind` validation to support B-spline decisions and add an optional `trajectory_model` field.
- Create `scenarios/signalized_intersection_bspline.py`: single-ego B-spline variant of the signalized benchmark.
- Modify `scenarios/registry.py`: register `signalized_intersection_bspline`.
- Create `costs/signalized_intersection_bspline.py`: B-spline cost profile that reuses signalized constraints but sources ego trajectory from B-spline rollout.
- Modify `costs/registry.py`: register `signalized_intersection_bspline`.
- Modify `planner.py`: seed initial component means for B-spline blocks from scenario-provided arrays; do not change old acc/steer scenarios.
- Modify `scenario_runtime.py`: add B-spline execution path for `trajectory_model == "frenet_bspline"`.
- Modify `viz_utils.py` only if needed to render B-spline dense trajectory with existing signalized renderer.
- Extend `tests/test_signalized_intersection_helpers.py`: contract tests for B-spline scenario/cost registration and rollout semantics.
- Create `tests/test_bspline_trajectory.py`: low-level trajectory tests.
- Update `README_signalized_intersection_CN.md`: document the B-spline path after it runs.

---

## Task 1: Add B-Spline Trajectory Module

**Files:**
- Create: `trajectory/__init__.py`
- Create: `trajectory/reference_path.py`
- Create: `trajectory/frenet_bspline.py`
- Create: `trajectory/warmstart.py`
- Copy asset: `trajectory/assets/bspline_basis.npz`
- Test: `tests/test_bspline_trajectory.py`

- [ ] **Step 1: Write failing tests for reference path and B-spline basis loading**

Create `tests/test_bspline_trajectory.py` with:

```python
"""Tests for Frenet B-spline trajectory helpers."""

from pathlib import Path

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python tests/test_bspline_trajectory.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'trajectory'`.

- [ ] **Step 3: Copy the B-spline asset**

Create directory and copy:

```powershell
New-Item -ItemType Directory -Force -Path trajectory\assets | Out-Null
Copy-Item -LiteralPath MGIGO\Cartest\bspline_basis.npz -Destination trajectory\assets\bspline_basis.npz -Force
```

- [ ] **Step 4: Implement `trajectory/reference_path.py`**

Use this implementation:

```python
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
```

- [ ] **Step 5: Implement `trajectory/frenet_bspline.py`**

Adapt `MGIGO/Cartest/frenet_traj.py`, but use local imports and add `to_compressed_states()`:

```python
"""Frenet-frame quintic B-spline trajectory generator."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

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
        p2 = 3.0 * p1 - 2.0 * p0 + (self.dt_knot ** 2 / 10.0) * a0
        return p0, p1, p2

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
        v2 = vt ** 2 + vn ** 2
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
```

- [ ] **Step 6: Implement `trajectory/warmstart.py`**

Use this implementation:

```python
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
```

- [ ] **Step 7: Export module symbols**

Create `trajectory/__init__.py`:

```python
"""Trajectory parameterization helpers."""

from .frenet_bspline import FrenetBSplineTrajectory
from .reference_path import ReferencePath, StraightReference

__all__ = ["FrenetBSplineTrajectory", "ReferencePath", "StraightReference"]
```

- [ ] **Step 8: Run tests to verify Task 1 passes**

Run:

```bash
python tests/test_bspline_trajectory.py
```

Expected: exit code 0, no traceback.

- [ ] **Step 9: Commit Task 1**

```bash
git add trajectory tests/test_bspline_trajectory.py
git commit -m "feat: add frenet bspline trajectory module"
```

---

## Task 2: Extend Scenario Contracts for B-Spline Decisions

**Files:**
- Modify: `scenarios/spec.py`
- Test: `tests/test_signalized_intersection_helpers.py`

- [ ] **Step 1: Write failing contract test for B-spline decision kinds**

Append this test to `tests/test_signalized_intersection_helpers.py`:

```python
def test_scenario_spec_accepts_bspline_decision_kinds():
    import numpy as np

    from scenarios.spec import (
        AgentSpec,
        BlockSpec,
        DecisionSpec,
        ScenarioSpec,
        state,
    )

    scenario = ScenarioSpec(
        name="unit_bspline",
        title="unit",
        description="unit",
        output_prefix="unit",
        cost_profile="unit",
        initial_states=np.stack([state(0.0, 0.0, 10.0)]),
        v_refs=np.array([10.0]),
        target_y=0.0,
        lane_roles=("ego_lane",),
        agent_roles=("ego",),
        agents=(AgentSpec("ego", "ego", "bicycle", 0, 0),),
        decisions=(
            DecisionSpec("ego_ctrl_s", "ego", "ctrl_s", (9,)),
            DecisionSpec("ego_ctrl_d", "ego", "ctrl_d", (9,)),
        ),
        blocks=(
            BlockSpec("ego_ctrl_s_block", "ego", ("ego_ctrl_s",), 0),
            BlockSpec("ego_ctrl_d_block", "ego", ("ego_ctrl_d",), 1),
        ),
        snap_labels=("a", "b", "c"),
        control_horizon=9,
        trajectory_model="frenet_bspline",
    )

    if scenario.block_dims != (9, 9):
        raise AssertionError(scenario.block_dims)
    if scenario.trajectory_model != "frenet_bspline":
        raise AssertionError(scenario.trajectory_model)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python tests/test_signalized_intersection_helpers.py
```

Expected: FAIL because `DecisionSpec.kind` rejects `ctrl_s` / `ctrl_d` or `ScenarioSpec` has no `trajectory_model` field.

- [ ] **Step 3: Update `scenarios/spec.py` types and validation**

Change:

```python
ControlKind = Literal["acc", "steer"]
```

to:

```python
ControlKind = Literal["acc", "steer", "ctrl_s", "ctrl_d"]
```

Add field to `ScenarioSpec`:

```python
trajectory_model: str = "control_sequence"
```

Replace the decision validation block with:

```python
            if decision.kind not in ("acc", "steer", "ctrl_s", "ctrl_d"):
                raise ValueError(
                    f"Decision {decision.name!r} has unsupported kind "
                    f"{decision.kind!r}"
                )
            if decision.kind in ("acc", "steer") and decision.shape != (self.control_horizon,):
                raise ValueError(
                    f"Scalar control decision {decision.name!r} shape must be "
                    f"({self.control_horizon},), got {decision.shape}"
                )
            if decision.kind in ("ctrl_s", "ctrl_d") and len(decision.shape) != 1:
                raise ValueError(
                    f"B-spline decision {decision.name!r} must be a flat vector, "
                    f"got {decision.shape}"
                )
```

Add trajectory model validation after `context_values` validation:

```python
        if self.trajectory_model not in ("control_sequence", "frenet_bspline"):
            raise ValueError(
                f"Scenario {self.name!r} has unsupported trajectory_model "
                f"{self.trajectory_model!r}"
            )
```

- [ ] **Step 4: Run tests**

Run:

```bash
python tests/test_signalized_intersection_helpers.py
```

Expected: exit code 0.

- [ ] **Step 5: Commit Task 2**

```bash
git add scenarios/spec.py tests/test_signalized_intersection_helpers.py
git commit -m "feat: allow bspline scenario decisions"
```

---

## Task 3: Add B-Spline Rollout Helper

**Files:**
- Create: `trajectory/rollout.py`
- Test: `tests/test_bspline_trajectory.py`

- [ ] **Step 1: Write failing rollout test**

Append to `tests/test_bspline_trajectory.py`:

```python
def test_bspline_rollout_returns_dense_ego_states():
    import jax.numpy as jnp
    import numpy as np

    from trajectory.frenet_bspline import FrenetBSplineTrajectory
    from trajectory.reference_path import StraightReference
    from trajectory.rollout import bspline_ego_rollout
    from trajectory.warmstart import tangent_control_points

    gen = FrenetBSplineTrajectory("trajectory/assets/bspline_basis.npz", StraightReference())
    ctrl_s, ctrl_d = tangent_control_points(gen, s0=0.0, s_dot0=12.0, d0=-1.75)
    ctx = {
        "s0": 0.0,
        "s_dot0": 12.0,
        "s_ddot0": 0.0,
        "d0": -1.75,
        "d_dot0": 0.0,
        "d_ddot0": 0.0,
    }
    traj = bspline_ego_rollout(gen, jnp.asarray(ctrl_s), jnp.asarray(ctrl_d), ctx)

    if traj.shape != (gen.T, 6):
        raise AssertionError(traj.shape)
    if not np.all(np.isfinite(np.asarray(traj))):
        raise AssertionError(traj)
    if abs(float(traj[0, 0])) > 1e-5:
        raise AssertionError(traj[0])
    if abs(float(traj[0, 1]) + 1.75) > 1e-5:
        raise AssertionError(traj[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python tests/test_bspline_trajectory.py
```

Expected: FAIL with `ModuleNotFoundError` or missing `trajectory.rollout`.

- [ ] **Step 3: Implement `trajectory/rollout.py`**

Use:

```python
"""Rollout helpers for trajectory-parameterized scenarios."""

from __future__ import annotations


def bspline_ego_rollout(gen, ctrl_s_free, ctrl_d_free, ctx):
    """Evaluate one ego Frenet B-spline and return compressed [T, 6] states."""
    _frenet, _full_states, compressed, _xy = gen.evaluate_plan(ctrl_s_free, ctrl_d_free, ctx)
    return compressed
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python tests/test_bspline_trajectory.py
```

Expected: exit code 0.

- [ ] **Step 5: Commit Task 3**

```bash
git add trajectory/rollout.py tests/test_bspline_trajectory.py
git commit -m "feat: add bspline rollout helper"
```

---

## Task 4: Add `signalized_intersection_bspline` Scenario

**Files:**
- Create: `scenarios/signalized_intersection_bspline.py`
- Modify: `scenarios/registry.py`
- Test: `tests/test_signalized_intersection_helpers.py`

- [ ] **Step 1: Write failing scenario registration test**

Append:

```python
def test_signalized_intersection_bspline_scenario_contract():
    from scenarios import get_scenario

    scenario = get_scenario("signalized_intersection_bspline")
    if scenario.name != "signalized_intersection_bspline":
        raise AssertionError(scenario.name)
    if scenario.cost_profile != "signalized_intersection_bspline":
        raise AssertionError(scenario.cost_profile)
    if scenario.trajectory_model != "frenet_bspline":
        raise AssertionError(scenario.trajectory_model)
    if scenario.block_dims != (9, 9):
        raise AssertionError(scenario.block_dims)
    kinds = tuple(decision.kind for decision in scenario.decisions)
    if kinds != ("ctrl_s", "ctrl_d"):
        raise AssertionError(kinds)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python tests/test_signalized_intersection_helpers.py
```

Expected: FAIL with unknown scenario.

- [ ] **Step 3: Implement scenario file**

Create `scenarios/signalized_intersection_bspline.py`:

```python
"""B-spline variant of the signalized intersection benchmark."""

from __future__ import annotations

import numpy as np

from .signalized_intersection import (
    EGO_INITIAL_Y,
    EGO_SPEED,
    _make_scenario,
)
from .spec import BlockSpec, DecisionSpec
from trajectory.frenet_bspline import FrenetBSplineTrajectory
from trajectory.reference_path import StraightReference
from trajectory.warmstart import initial_component_means


def make_scenario():
    base = _make_scenario(
        name="signalized_intersection_bspline",
        title_suffix="B-Spline Critical",
        ego_speed=EGO_SPEED,
        yellow_start_s=0.6,
        yellow_duration_s=2.7,
        n_mpc_steps=30,
    )
    gen = FrenetBSplineTrajectory("trajectory/assets/bspline_basis.npz", StraightReference())
    component_means = initial_component_means(
        gen,
        s0=float(base.initial_states[0, 0]),
        s_dot0=float(base.initial_states[0, 2]),
        d0=EGO_INITIAL_Y,
        n_components=3,
    )
    return base.__class__(
        name="signalized_intersection_bspline",
        title="B-Spline Critical: ego approaches a yellow-light intersection with probabilistic cross traffic",
        description=base.description,
        output_prefix="mgigo_signalized_intersection_bspline",
        cost_profile="signalized_intersection_bspline",
        initial_states=base.initial_states,
        v_refs=base.v_refs,
        target_y=base.target_y,
        lane_roles=base.lane_roles,
        agent_roles=base.agent_roles,
        agents=base.agents,
        decisions=(
            DecisionSpec("ego_ctrl_s", "ego", "ctrl_s", (gen.n_free,)),
            DecisionSpec("ego_ctrl_d", "ego", "ctrl_d", (gen.n_free,)),
        ),
        blocks=(
            BlockSpec("ego_ctrl_s_block", "ego", ("ego_ctrl_s",), 0),
            BlockSpec("ego_ctrl_d_block", "ego", ("ego_ctrl_d",), 1),
        ),
        snap_labels=base.snap_labels,
        backend=base.backend,
        state_dim=base.state_dim,
        control_horizon=gen.n_free,
        n_mpc_steps=base.n_mpc_steps,
        snap_frames=base.snap_frames,
        road=base.road,
        vehicle_geometry=base.vehicle_geometry,
        context_values=base.context_values,
        notes=base.notes + ("trajectory_model=frenet_bspline",),
        exec_mode="cost_select",
        initial_component_means=component_means,
        trajectory_model="frenet_bspline",
    )
```

- [ ] **Step 4: Register scenario**

Modify `scenarios/registry.py`:

```python
from .signalized_intersection_bspline import make_scenario as make_signalized_intersection_bspline
```

Add to `SCENARIO_FACTORIES`:

```python
    "signalized_intersection_bspline": make_signalized_intersection_bspline,
```

- [ ] **Step 5: Run test**

Run:

```bash
python tests/test_signalized_intersection_helpers.py
```

Expected: exit code 0 after cost profile registration is temporarily skipped by this test or after Task 5. If it fails due missing cost profile, proceed to Task 5 before re-running full helper suite.

- [ ] **Step 6: Commit Task 4**

```bash
git add scenarios/signalized_intersection_bspline.py scenarios/registry.py tests/test_signalized_intersection_helpers.py
git commit -m "feat: add signalized bspline scenario"
```

---

## Task 5: Add B-Spline Signalized Cost Profile

**Files:**
- Create: `costs/signalized_intersection_bspline.py`
- Modify: `costs/registry.py`
- Test: `tests/test_signalized_intersection_helpers.py`

- [ ] **Step 1: Write failing cost registration test**

Append:

```python
def test_signalized_intersection_bspline_cost_profile_is_registered_and_finite():
    import jax.numpy as jnp
    import numpy as np

    from costs import get_cost_functions
    from scenarios import get_scenario

    scenario = get_scenario("signalized_intersection_bspline")
    cost_functions = get_cost_functions("signalized_intersection_bspline")
    if len(cost_functions) != 1:
        raise AssertionError(cost_functions)
    width = max(scenario.block_dims)
    sample = np.zeros((scenario.n_control_blocks, width), dtype=np.float32)
    for block_idx, component_values in enumerate(scenario.initial_component_means):
        sample[block_idx, : scenario.block_dims[block_idx]] = np.asarray(component_values[0], dtype=np.float32)
    context_arr = jnp.concatenate(
        [
            jnp.asarray(scenario.initial_states.reshape(-1), dtype=jnp.float32),
            jnp.asarray(scenario.v_refs, dtype=jnp.float32),
            jnp.asarray(scenario.context_values, dtype=jnp.float32),
            jnp.asarray([0.0], dtype=jnp.float32),
        ]
    )
    value = cost_functions[0](jnp.asarray(sample.reshape(-1)), context_arr)
    if not np.isfinite(float(value)):
        raise AssertionError(value)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python tests/test_signalized_intersection_helpers.py
```

Expected: FAIL with unknown cost profile.

- [ ] **Step 3: Implement `costs/signalized_intersection_bspline.py`**

Use current signalized constraint functions and a local B-spline shared context:

```python
"""B-spline cost profile for signalized intersection."""

from __future__ import annotations

import jax.numpy as jnp

from scenarios import get_scenario
from trajectory.frenet_bspline import FrenetBSplineTrajectory
from trajectory.reference_path import StraightReference
from trajectory.rollout import bspline_ego_rollout

from . import signalized_intersection as base
from .constraint_dsl import build


_SCENARIO = get_scenario("signalized_intersection_bspline")
_STATE_DIM = _SCENARIO.state_dim
_CTX_STATE_DIM = _SCENARIO.context_state_dim
_GEN = FrenetBSplineTrajectory("trajectory/assets/bspline_basis.npz", StraightReference())


def _decode(joint_sample_flat):
    blocks = joint_sample_flat.reshape((_SCENARIO.n_control_blocks, max(_SCENARIO.block_dims)))
    return {
        "ego_ctrl_s": blocks[0, : _SCENARIO.block_dims[0]],
        "ego_ctrl_d": blocks[1, : _SCENARIO.block_dims[1]],
    }


def _context_dict(context_arr):
    current_states = context_arr[:_CTX_STATE_DIM].reshape(_SCENARIO.n_agents, _STATE_DIM)
    ego = current_states[0]
    return {
        "s0": ego[0],
        "s_dot0": ego[2],
        "s_ddot0": ego[4],
        "d0": ego[1],
        "d_dot0": jnp.asarray(0.0, dtype=jnp.float32),
        "d_ddot0": jnp.asarray(0.0, dtype=jnp.float32),
    }


def _shared_context(joint_sample_flat, context_arr):
    decisions = _decode(joint_sample_flat)
    ego_traj = bspline_ego_rollout(
        _GEN,
        decisions["ego_ctrl_s"],
        decisions["ego_ctrl_d"],
        _context_dict(context_arr),
    )
    return {
        "decisions": decisions,
        "ego_traj": ego_traj,
        "context_arr": context_arr,
    }


def _ego_objective(x, ctx):
    ego = ctx["ego_traj"]
    decisions = ctx["decisions"]
    v_ref = ctx["context_arr"][_CTX_STATE_DIM]
    stop_basin = (ego[-1, 0] - (base.STOP_LINE_X - 2.0)) ** 2 + 4.0 * ego[-1, 2] ** 2
    pass_basin = (ego[-1, 0] - (base.INTERSECTION_EXIT_X + 6.0)) ** 2
    mild_task = 0.05 * jnp.minimum(stop_basin, pass_basin)
    speed = 0.2 * jnp.sum((ego[:, 2] - v_ref) ** 2 * base.DT_C)
    lane = 4.0 * jnp.sum((ego[:, 1] - _SCENARIO.target_y) ** 2 * base.DT_C)
    heading = 3.0 * jnp.sum(ego[:, 3] ** 2 * base.DT_C)
    spline_shape = 0.02 * (
        jnp.sum(jnp.diff(decisions["ego_ctrl_s"], n=2) ** 2)
        + jnp.sum(jnp.diff(decisions["ego_ctrl_d"], n=2) ** 2)
    )
    return mild_task + speed + lane + heading + spline_shape


def _ego_traj_override(ctx):
    return ctx["ego_traj"]


def _patch_base_ego_traj(fn):
    def wrapped(x, ctx):
        original = base._ego_traj
        try:
            base._ego_traj = _ego_traj_override
            return fn(x, ctx)
        finally:
            base._ego_traj = original
    return wrapped


_ego_base_cost = build(
    _ego_objective,
    [
        spec.__class__(
            **{
                **spec.__dict__,
                "g_fn": _patch_base_ego_traj(spec.g_fn),
            }
        )
        if name != "cross_traffic_chance"
        else spec.__class__(
            **{
                **spec.__dict__,
                "g_fn": lambda x, xi, ctx: base._cross_traffic_risk_violation(x, xi, {"ego_traj": ctx["ego_traj"], "context_arr": ctx["context_arr"]}),
            }
        )
        for name, spec in base.EGO_CONSTRAINT_SPECS
    ],
    k_inner=0.1,
    penalize_only_soft=True,
    jit_cost=False,
    obj_transform="standard",
)


def ego_cost(joint_sample_flat, context_arr):
    return _ego_base_cost(joint_sample_flat, _shared_context(joint_sample_flat, context_arr))
```

After implementing, immediately refactor away monkey-patching if tests fail or JAX tracing rejects it. Preferred refactor is to expose trajectory-taking helper functions in `costs/signalized_intersection.py`, but keep the first implementation small.

- [ ] **Step 4: Register cost profile**

Modify `costs/registry.py`:

```python
from . import signalized_intersection_bspline
```

Add:

```python
    "signalized_intersection_bspline": (signalized_intersection_bspline.ego_cost,),
```

- [ ] **Step 5: Run tests**

Run:

```bash
python tests/test_signalized_intersection_helpers.py
```

Expected: exit code 0.

- [ ] **Step 6: Commit Task 5**

```bash
git add costs/signalized_intersection_bspline.py costs/registry.py tests/test_signalized_intersection_helpers.py
git commit -m "feat: add signalized bspline cost profile"
```

---

## Task 6: Add B-Spline Runtime Execution

**Files:**
- Modify: `scenario_runtime.py`
- Test: `tests/test_bspline_trajectory.py`

- [ ] **Step 1: Write failing execution test**

Append:

```python
def test_bspline_runtime_advances_one_macro_step():
    import numpy as np

    from scenario_runtime import advance_one_macro_step
    from scenarios import get_scenario

    scenario = get_scenario("signalized_intersection_bspline")
    decisions = {}
    for decision, component_values in zip(scenario.decisions, scenario.initial_component_means):
        decisions[decision.name] = np.asarray(component_values[0], dtype=np.float64)
    next_states = advance_one_macro_step(scenario, scenario.initial_states, decisions)

    if next_states.shape != scenario.initial_states.shape:
        raise AssertionError(next_states.shape)
    if not np.all(np.isfinite(next_states)):
        raise AssertionError(next_states)
    if not next_states[0, 0] > scenario.initial_states[0, 0]:
        raise AssertionError((scenario.initial_states[0], next_states[0]))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python tests/test_bspline_trajectory.py
```

Expected: FAIL because `controls_by_agent()` expects acc/steer controls.

- [ ] **Step 3: Add B-spline branch to `advance_one_macro_step()`**

In `scenario_runtime.py`, import:

```python
from trajectory.frenet_bspline import FrenetBSplineTrajectory
from trajectory.reference_path import StraightReference
from trajectory.rollout import bspline_ego_rollout
```

Add helper:

```python
def _advance_bspline_one_macro_step(scenario, current_states, decision_sequences):
    gen = FrenetBSplineTrajectory("trajectory/assets/bspline_basis.npz", StraightReference())
    ego = current_states[0]
    ctx = {
        "s0": ego[0],
        "s_dot0": ego[2],
        "s_ddot0": ego[4],
        "d0": ego[1],
        "d_dot0": 0.0,
        "d_ddot0": 0.0,
    }
    traj = np.asarray(
        bspline_ego_rollout(
            gen,
            decision_sequences["ego_ctrl_s"],
            decision_sequences["ego_ctrl_d"],
            ctx,
        ),
        dtype=np.float64,
    )
    macro_idx = min(SUB_STEPS, len(traj) - 1)
    next_states = np.asarray(current_states, dtype=np.float64).copy()
    next_states[0] = traj[macro_idx]
    return next_states
```

At top of `advance_one_macro_step()`:

```python
    if getattr(scenario, "trajectory_model", "control_sequence") == "frenet_bspline":
        return _advance_bspline_one_macro_step(scenario, current_states, decision_sequences)
```

- [ ] **Step 4: Add B-spline branch to `prediction_trajs()`**

At top of `prediction_trajs()`:

```python
    if getattr(scenario, "trajectory_model", "control_sequence") == "frenet_bspline":
        gen = FrenetBSplineTrajectory("trajectory/assets/bspline_basis.npz", StraightReference())
        ego = current_states[0]
        ctx = {
            "s0": ego[0],
            "s_dot0": ego[2],
            "s_ddot0": ego[4],
            "d0": ego[1],
            "d_dot0": 0.0,
            "d_ddot0": 0.0,
        }
        traj = np.asarray(
            bspline_ego_rollout(
                gen,
                decision_sequences["ego_ctrl_s"],
                decision_sequences["ego_ctrl_d"],
                ctx,
            ),
            dtype=np.float64,
        )
        return {"ego": traj}
```

- [ ] **Step 5: Run tests**

Run:

```bash
python tests/test_bspline_trajectory.py
python tests/test_signalized_intersection_helpers.py
```

Expected: both exit code 0.

- [ ] **Step 6: Commit Task 6**

```bash
git add scenario_runtime.py tests/test_bspline_trajectory.py
git commit -m "feat: execute bspline signalized trajectories"
```

---

## Task 7: Run First B-Spline Scenario and Stabilize Minimal Behavior

**Files:**
- Modify only if tests or run reveal a structural bug:
  - `costs/signalized_intersection_bspline.py`
  - `scenarios/signalized_intersection_bspline.py`
  - `trajectory/*.py`

- [ ] **Step 1: Run a cheap import/compile check**

Run:

```bash
python -m py_compile trajectory/reference_path.py trajectory/frenet_bspline.py trajectory/warmstart.py trajectory/rollout.py scenarios/signalized_intersection_bspline.py costs/signalized_intersection_bspline.py scenario_runtime.py
```

Expected: exit code 0.

- [ ] **Step 2: Run a single B-spline scenario in WSL2/CUDA**

Run from Windows PowerShell:

```powershell
wsl.exe -d Ubuntu-22.04 bash -lc 'cd /mnt/d/claude_workspace1/igo && JAX_PLATFORMS=cuda /root/.venvs/tcmgigo-jaxgpu/bin/python run_mgigo_scenario.py signalized_intersection_bspline signalized_intersection_bspline'
```

Expected:

```text
初始化 JAX...
场景：signalized_intersection_bspline
...
完成！
```

Expected outputs:

```text
figures/mgigo_signalized_intersection_bspline_snapshot.png
figures/mgigo_signalized_intersection_bspline.mp4
```

- [ ] **Step 3: If the run fails due cost helper monkey-patching, refactor trajectory-taking constraints**

Modify `costs/signalized_intersection.py` to add pure helpers:

```python
def red_light_violation_from_traj(ego, context_arr):
    ...

def road_boundary_violation_from_traj(ego):
    ...

def no_blocking_intersection_violation_from_traj(ego):
    ...

def dilemma_task_violation_from_traj(ego):
    ...

def cross_traffic_risk_violation_from_traj(ego, xi):
    ...
```

Then make both acc/steer and B-spline profiles call these pure helpers. Keep old public cost behavior unchanged.

- [ ] **Step 4: Re-run tests after any stabilization**

Run:

```bash
python tests/test_constraint_dsl_semantics.py
python tests/test_bspline_trajectory.py
python tests/test_signalized_intersection_helpers.py
python -m py_compile costs/signalized_intersection.py costs/signalized_intersection_bspline.py trajectory/frenet_bspline.py scenario_runtime.py
```

Expected: all exit code 0.

- [ ] **Step 5: Commit Task 7**

```bash
git add costs/signalized_intersection.py costs/signalized_intersection_bspline.py scenarios/signalized_intersection_bspline.py trajectory scenario_runtime.py tests
git commit -m "fix: stabilize signalized bspline benchmark"
```

---

## Task 8: Add B-Spline Comparison and Documentation

**Files:**
- Modify: `compare_signalized_intersection_profiles.py` or create `compare_signalized_bspline_profiles.py`
- Modify: `README_signalized_intersection_CN.md`
- Test: existing helper tests plus a cheap report-generation smoke if a new script is added.

- [ ] **Step 1: Decide comparison scope**

Use a new small script only if adding B-spline to the existing 3x4 comparison makes runtime too expensive. Preferred first comparison:

```text
signalized_intersection_critical / signalized_intersection
signalized_intersection_bspline / signalized_intersection_bspline
```

- [ ] **Step 2: Document B-spline path**

Add to `README_signalized_intersection_CN.md`:

```markdown
## B-spline 版本

新增 `signalized_intersection_bspline` 作为并行实现。旧版本优化 ego 的 `acc/steer` 控制序列；B-spline 版本优化 Frenet `ctrl_s/ctrl_d` 自由控制点，并由五次 B-spline 解析出 `[x, y, v, psi, acc, steer]` 轨迹。

两者共用 prioritized signalized cost 语义：hard red-light / road-boundary / no-blocking，tunable cross-traffic chance，tunable stop/pass dilemma。B-spline 版本用于验证同一 black-box cost 在更平滑、更低维轨迹参数化下是否仍能工作。
```

- [ ] **Step 3: Run documentation and tests**

Run:

```bash
python tests/test_constraint_dsl_semantics.py
python tests/test_bspline_trajectory.py
python tests/test_signalized_intersection_helpers.py
```

Expected: all exit code 0.

- [ ] **Step 4: Commit Task 8**

```bash
git add README_signalized_intersection_CN.md compare_signalized_bspline_profiles.py compare_signalized_intersection_profiles.py tests
git commit -m "docs: document signalized bspline benchmark"
```

---

## Final Verification

- [ ] **Step 1: Run lightweight local verification**

```bash
python tests/test_constraint_dsl_semantics.py
python tests/test_bspline_trajectory.py
python tests/test_signalized_intersection_helpers.py
python -m py_compile trajectory/reference_path.py trajectory/frenet_bspline.py trajectory/warmstart.py trajectory/rollout.py scenarios/signalized_intersection_bspline.py costs/signalized_intersection_bspline.py scenario_runtime.py planner.py
```

Expected: all commands exit code 0.

- [ ] **Step 2: Run WSL2/CUDA B-spline benchmark**

```powershell
wsl.exe -d Ubuntu-22.04 bash -lc 'cd /mnt/d/claude_workspace1/igo && JAX_PLATFORMS=cuda /root/.venvs/tcmgigo-jaxgpu/bin/python run_mgigo_scenario.py signalized_intersection_bspline signalized_intersection_bspline'
```

Expected: mp4 and snapshot generated under `figures/`.

- [ ] **Step 3: Compare old and new signalized outputs**

Run:

```powershell
wsl.exe -d Ubuntu-22.04 bash -lc 'cd /mnt/d/claude_workspace1/igo && JAX_PLATFORMS=cuda /root/.venvs/tcmgigo-jaxgpu/bin/python run_mgigo_scenario.py signalized_intersection_critical signalized_intersection'
```

Expected: old acc/steer scenario still runs and writes `figures/mgigo_signalized_intersection_critical.mp4`.

- [ ] **Step 4: Inspect Git state**

```bash
git status --short
```

Expected: only intentional generated files are untracked or modified. Do not add `MGIGO/Cartest` or `MGIGO-master` reference trees unless explicitly requested.

---

## Self-Review

**Spec coverage:** This plan keeps the current acc/steer benchmark, adds a complete B-spline path, reuses current `constraint_dsl.py`, and avoids introducing a second Constraintdealer runtime.

**Placeholder scan:** No task uses TBD/TODO. Each code task includes concrete file paths, concrete code, and concrete verification commands.

**Type consistency:** The plan consistently uses `trajectory_model="frenet_bspline"`, B-spline decisions `ctrl_s` and `ctrl_d`, 9 free control points per channel, and compressed `[x, y, v, psi, acc, steer]` states for compatibility.

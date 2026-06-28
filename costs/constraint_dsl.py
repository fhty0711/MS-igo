"""Constraint-to-cost helpers for MGIGO black-box objectives.

Users write objective functions and violations in ``g(x, ctx) <= 0`` form.
``build`` assembles them into a JAX-compatible scalar cost using the current
Constran-style T_alpha transforms, tunable/soft additive nesting, and priority
layers.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random, vmap


TRANSFORM_SOFT = (
    np.array([1e-2, 5e-2, 1e-1, 0.5, 1, 10, 100, 1e4, 1e6]),
    np.array([0.03, 0.08, 0.2, 0.5, 1.0, 2.5, 5.0, 9.0, 13.0]),
)
TRANSFORM_TUNABLE = (
    np.array([1e-4, 1e-3, 1e-2, 0.1, 0.5, 1, 10, 100, 1e4, 1e6]),
    np.array([0.15, 0.25, 0.4, 0.7, 1.0, 1.5, 3.0, 5.5, 9.0, 13.0]),
)
TRANSFORM_HARD = (
    np.array([1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e4, 1e6]),
    np.array([0.5, 0.7, 0.9, 1.2, 1.6, 2.5, 4.0, 7.0, 10.0, 13.0]),
)

TRANSFORM_STANDARD = TRANSFORM_TUNABLE
TRANSFORM_SHARP = TRANSFORM_HARD
TRANSFORM_TIGHT = TRANSFORM_SOFT
TRANSFORM_WIDE = TRANSFORM_SOFT

OBJ_TRANSFORM_STANDARD = (
    np.array([1e-4, 1e-2, 1e0, 1e2, 1e4, 1e8]),
    np.array([0.5, 0.7, 1.5, 3.0, 6.0, 12.0]),
)
OBJ_TRANSFORM_FLAT = (
    np.array([1e-2, 1e0, 1e2, 1e4, 1e8]),
    np.array([0.5, 1.0, 2.5, 5.0, 10.0]),
)

TRANSFORM_PRESETS = {
    "soft": TRANSFORM_SOFT,
    "tunable": TRANSFORM_TUNABLE,
    "hard": TRANSFORM_HARD,
    "tight": TRANSFORM_SOFT,
    "standard": TRANSFORM_TUNABLE,
    "sharp": TRANSFORM_HARD,
    "wide": TRANSFORM_SOFT,
    "log": None,
}
DEFAULT_TRANSFORM = {
    "soft": "soft",
    "tunable": "tunable",
    "hard": "hard",
}
OBJ_PRESETS = {
    "standard": OBJ_TRANSFORM_STANDARD,
    "flat": OBJ_TRANSFORM_FLAT,
    "log": None,
}

DELTA_TIGHT = (
    np.array([0.0, 1e-4, 1e-2, 1e-1, 1e0, 1e2, 1e4]),
    np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]),
)
DELTA_STANDARD = (
    np.array([0.0, 1e-4, 1e-2, 1e-1, 1e0, 1e2, 1e4]),
    np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]),
)
DELTA_SHARP = (
    np.array([0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e2]),
    np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]),
)
DELTA_PRESETS = {
    "tight": DELTA_TIGHT,
    "standard": DELTA_STANDARD,
    "sharp": DELTA_SHARP,
    "none": None,
}

CONSTRAINT_K = 0.2
NEAR_HARD_BETA = 1.0
TUNE_PRESETS = {
    "mild": (0.1, 1.0),
    "standard": (0.3, 1.0),
    "firm": (0.5, 1.5),
    "strong": (1.0, 1.5),
    "nearhard": (1.0, 2.0),
    "__hard__": (2.0, 1.5),
    "__tunable_default__": (0.3, 1.0),
}


@jax.jit
def sigma_k(x: jax.Array, k: float = 1.0) -> jax.Array:
    """Odd saturation: sigma_k(x) = kx / sqrt(1 + (kx)^2)."""
    kx = k * x
    return kx / jnp.sqrt(1.0 + kx**2)


@jax.jit
def log_transform(x: jax.Array) -> jax.Array:
    """Odd log compression retained for ``transform='log'`` compatibility."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


@jax.jit
def T_alpha(
    x: jax.Array,
    knots_g: jax.Array = TRANSFORM_TUNABLE[0],
    knots_T: jax.Array = TRANSFORM_TUNABLE[1],
) -> jax.Array:
    """Piecewise log-like transform with a small-violation floor."""
    ax = jnp.abs(x)
    kg = jnp.asarray(knots_g)
    kt = jnp.asarray(knots_T)
    log_knots_g = jnp.log(kg)
    log_ax = jnp.log(jnp.maximum(ax, jnp.nextafter(0.0, 1.0)))
    idx = jnp.searchsorted(log_knots_g, log_ax, side="right") - 1
    idx = jnp.clip(idx, 0, len(knots_g) - 2)
    x0 = log_knots_g[idx]
    x1 = log_knots_g[idx + 1]
    y0 = kt[idx]
    y1 = kt[idx + 1]
    t = jnp.clip((log_ax - x0) / (x1 - x0 + 1e-12), 0.0, 1.0)
    return jnp.sign(x) * (y0 + t * (y1 - y0))


def _make_delta_fn(knots_g, knots_d):
    if knots_g is None or knots_d is None:
        return None
    kg = jnp.asarray(knots_g)
    kd = jnp.asarray(knots_d)
    log_kg = jnp.log(jnp.maximum(kg, jnp.nextafter(0.0, 1.0)))

    def delta_fn(g_raw):
        ax = jnp.abs(g_raw)
        log_ax = jnp.log(jnp.maximum(ax, jnp.nextafter(0.0, 1.0)))
        idx = jnp.searchsorted(log_kg, log_ax, side="right") - 1
        idx = jnp.clip(idx, 0, len(knots_g) - 2)
        x0 = log_kg[idx]
        x1 = log_kg[idx + 1]
        y0 = kd[idx]
        y1 = kd[idx + 1]
        t = jnp.clip((log_ax - x0) / (x1 - x0 + 1e-12), 0.0, 1.0)
        return y0 + t * (y1 - y0)

    return delta_fn


@dataclass(kw_only=True)
class ConstraintSpec:
    """Base specification for one constraint layer."""

    mode: str = "soft"
    priority: int = 1
    delta: Optional[float] = None
    delta_table: str = "none"
    _delta_table_raw: Optional[Tuple] = None
    delta_soft: Optional[float] = None
    beta: Optional[float] = None
    tune_preset: str = "none"
    transform: str = ""
    _transform_table: Optional[Tuple] = None
    aggregate: str = ""

    def __post_init__(self):
        if self.priority < 1:
            raise ValueError(f"priority must be >= 1, got {self.priority}")
        if self.mode == "hard":
            self.mode = "tunable"
            if self.delta is not None and self.delta_soft is None:
                self.delta_soft = self.delta
            if self.tune_preset == "none":
                self.tune_preset = "__hard__"
            if not self.transform:
                self.transform = "hard"
        if self.mode not in ("soft", "tunable"):
            raise ValueError(
                f"mode must be 'hard', 'soft', or 'tunable', got {self.mode!r}"
            )
        if not self.transform:
            self.transform = DEFAULT_TRANSFORM.get(self.mode, "standard")
        if (
            self.mode == "tunable"
            and self.tune_preset == "none"
            and self.beta is None
            and self.delta_soft is None
        ):
            self.tune_preset = "__tunable_default__"
        if self.transform not in TRANSFORM_PRESETS and self._transform_table is None:
            raise ValueError(
                f"Unknown transform preset: {self.transform!r}. "
                f"Available: {list(TRANSFORM_PRESETS.keys())}."
            )
        if self.delta_table not in DELTA_PRESETS and self._delta_table_raw is None:
            raise ValueError(
                f"Unknown delta_table preset: {self.delta_table!r}. "
                f"Available: {list(DELTA_PRESETS.keys())}."
            )
        if (
            self.mode == "tunable"
            and self.tune_preset not in TUNE_PRESETS
            and self.tune_preset != "none"
        ):
            raise ValueError(
                f"Unknown tune_preset: {self.tune_preset!r}. "
                f"Available: {list(TUNE_PRESETS.keys())}."
            )

    def get_transform_table(self):
        if self._transform_table is not None:
            return self._transform_table
        return TRANSFORM_PRESETS[self.transform]

    def get_delta_table(self):
        if self._delta_table_raw is not None:
            return self._delta_table_raw
        return DELTA_PRESETS[self.delta_table]

    def get_tune_params(self):
        if self.tune_preset == "__hard__":
            return (
                TUNE_PRESETS["__hard__"][0]
                if self.beta is None
                else self.beta,
                TUNE_PRESETS["__hard__"][1]
                if self.delta_soft is None
                else self.delta_soft,
            )
        if self.tune_preset == "__tunable_default__":
            return TUNE_PRESETS["__tunable_default__"]
        if self.tune_preset != "none":
            return TUNE_PRESETS[self.tune_preset]
        return (
            self.beta if self.beta is not None else 0.3,
            self.delta_soft if self.delta_soft is not None else 1.0,
        )


@dataclass
class Deterministic(ConstraintSpec):
    """Deterministic constraint: g_fn(x, ctx) <= 0."""

    g_fn: Optional[Callable[[jax.Array, Any], jax.Array]] = None


@dataclass
class Chance(ConstraintSpec):
    """Chance constraint via Monte Carlo quantile of sampled violations."""

    g_fn: Optional[Callable[[jax.Array, jax.Array, Any], jax.Array]] = None
    noise_fn: Optional[Callable[[jax.Array, Tuple[int, ...]], jax.Array]] = None
    alpha: float = 0.1
    n_samples: int = 100

    def __post_init__(self):
        super().__post_init__()
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {self.n_samples}")


@dataclass
class Robust(ConstraintSpec):
    """Robust constraint: max_xi g_fn(x, xi, ctx) <= 0."""

    g_fn: Optional[Callable[[jax.Array, jax.Array, Any], jax.Array]] = None
    uncertainty_set: Union[jax.Array, Callable[[int], jax.Array], Sequence, None] = None
    n_grid: int = 40


@dataclass
class DRO(ConstraintSpec):
    """Distributionally robust chance constraint over candidate distributions."""

    g_fn: Optional[Callable[[jax.Array, jax.Array, Any], jax.Array]] = None
    ambiguity_set: Optional[List[Callable[[jax.Array, Tuple[int, ...]], jax.Array]]] = None
    alpha: float = 0.1
    n_samples_per_dist: int = 100

    def __post_init__(self):
        super().__post_init__()
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.n_samples_per_dist <= 0:
            raise ValueError(
                f"n_samples_per_dist must be positive, got {self.n_samples_per_dist}"
            )


def _aggregate_value(value, agg: str):
    if not agg or agg in ("sum", "identity", ""):
        return jnp.sum(value)
    if agg == "mean":
        return jnp.mean(value)
    if agg == "max":
        return jnp.max(value)
    if agg == "count":
        return jnp.sum(value > 0.0)
    if agg.startswith("q"):
        q = float(agg[1:]) / 100.0
        return jnp.quantile(value, q)
    raise ValueError(
        f"Unknown aggregate: {agg!r}. Use 'sum','mean','max','count','q90','q95','q99'."
    )


def _wrap_aggregate(g_fn, agg: str, sampled: bool = False):
    """Wrap scalar/vector g_fn output aggregation while preserving signatures."""
    if sampled:
        return lambda x, xi, ctx: _aggregate_value(g_fn(x, xi, ctx), agg)
    return lambda x, ctx: _aggregate_value(g_fn(x, ctx), agg)


def _make_violation_fn(spec: ConstraintSpec) -> Callable:
    """Convert a ConstraintSpec into a deterministic violation function."""
    agg = spec.aggregate

    if isinstance(spec, Deterministic):
        return _wrap_aggregate(spec.g_fn, agg or "sum")

    if isinstance(spec, Chance):
        g_fn = _wrap_aggregate(spec.g_fn, agg or "sum", sampled=True)
        noise_fn = spec.noise_fn
        alpha = spec.alpha
        n_samples = spec.n_samples

        def chance_violation(x, ctx):
            key = random.PRNGKey(0)
            xi = noise_fn(key, (n_samples,))
            samples = vmap(lambda xi_i: g_fn(x, xi_i, ctx))(xi)
            return jnp.quantile(samples, 1.0 - alpha)

        return chance_violation

    if isinstance(spec, Robust):
        g_fn = _wrap_aggregate(spec.g_fn, agg or "sum", sampled=True)
        uncertainty_set = spec.uncertainty_set(spec.n_grid) if callable(
            spec.uncertainty_set
        ) else jnp.asarray(spec.uncertainty_set)

        def robust_violation(x, ctx):
            def body(carry, xi):
                return jnp.maximum(carry, g_fn(x, xi, ctx)), None

            worst, _ = lax.scan(body, -jnp.inf, uncertainty_set)
            return worst

        return robust_violation

    if isinstance(spec, DRO):
        g_fn = _wrap_aggregate(spec.g_fn, agg or "sum", sampled=True)
        ambiguity_set = tuple(spec.ambiguity_set)
        alpha = spec.alpha
        n_samples = spec.n_samples_per_dist

        def dro_violation(x, ctx):
            worst_q = -jnp.inf
            for noise_fn in ambiguity_set:
                key = random.PRNGKey(0)
                xi = noise_fn(key, (n_samples,))
                samples = vmap(lambda xi_i: g_fn(x, xi_i, ctx))(xi)
                worst_q = jnp.maximum(worst_q, jnp.quantile(samples, 1.0 - alpha))
            return worst_q

        return dro_violation

    raise TypeError(f"Unknown constraint type: {type(spec)}")


def _make_transform_fn(table):
    """Return T(x) for a preset/custom table. None selects odd log compression."""
    if table is None:
        return log_transform
    knots_g, knots_T = table

    def transform_fn(x):
        return T_alpha(x, knots_g, knots_T)

    return transform_fn


def _assemble_nest(
    objective_fn: Callable[[jax.Array, Any], jax.Array],
    layers: List[Tuple[int, str, dict, Callable, Callable]],
    k_inner: float,
    penalize_only_soft: bool,
    obj_T_fn: Callable,
) -> Callable[[jax.Array, Any], jax.Array]:
    """Assemble objective and constraint layers into one scalar black-box cost."""

    def cost_fn(x, ctx):
        inner = sigma_k(obj_T_fn(objective_fn(x, ctx)), k=k_inner)

        for _priority, mode, params, viol_fn, T_fn in layers:
            g_raw = viol_fn(x, ctx)
            t_val = T_fn(g_raw)
            if penalize_only_soft:
                t_val = jnp.maximum(0.0, t_val)

            if mode == "tunable":
                beta = params.get("beta", 0.3)
                delta_soft = params.get("delta_soft", 1.0)
                delta_fn = params.get("delta_fn")
                if delta_fn is not None:
                    delta_soft = delta_fn(g_raw)
                contrib = delta_soft * sigma_k(beta * t_val)
                inner = sigma_k(contrib + inner, k=params.get("k_out", CONSTRAINT_K))
            else:
                inner = sigma_k(t_val + inner, k=params.get("k_out", CONSTRAINT_K))

        return inner

    return cost_fn


def build(
    objective_fn: Callable[[jax.Array, Any], jax.Array],
    constraints: Optional[Sequence[ConstraintSpec]] = None,
    *,
    k_inner: float = 0.1,
    penalize_only_soft: bool = False,
    validate: bool = True,
    jit_cost: bool = True,
    obj_transform: Union[str, Tuple] = "standard",
) -> Callable[[jax.Array, Any], jax.Array]:
    """Build a JAX-compatible scalar cost from objective and constraints."""
    constraints = [] if constraints is None else list(constraints)
    if validate:
        _validate_constraints(constraints)

    if isinstance(obj_transform, tuple):
        obj_T_fn = _make_transform_fn(obj_transform)
    elif obj_transform in OBJ_PRESETS:
        obj_T_fn = _make_transform_fn(OBJ_PRESETS[obj_transform])
    else:
        raise ValueError(
            f"Unknown obj_transform: {obj_transform!r}. "
            f"Available: {list(OBJ_PRESETS.keys())}."
        )

    specs_sorted = sorted(constraints, key=lambda spec: spec.priority, reverse=True)
    layers = []
    for spec in specs_sorted:
        viol_fn = _make_violation_fn(spec)
        params = {}
        if spec.mode == "tunable":
            beta, delta_soft = spec.get_tune_params()
            params["beta"] = beta
            params["delta_soft"] = delta_soft
            delta_table = spec.get_delta_table()
            if delta_table is not None:
                params["delta_fn"] = _make_delta_fn(*delta_table)
        layers.append(
            (
                spec.priority,
                spec.mode,
                params,
                viol_fn,
                _make_transform_fn(spec.get_transform_table()),
            )
        )

    cost_fn = _assemble_nest(
        objective_fn,
        layers,
        k_inner=k_inner,
        penalize_only_soft=penalize_only_soft,
        obj_T_fn=obj_T_fn,
    )
    return jax.jit(cost_fn) if jit_cost else cost_fn


def build_unconstrained(
    objective_fn: Callable[[jax.Array, Any], jax.Array],
    k_inner: float = 0.1,
) -> Callable[[jax.Array, Any], jax.Array]:
    return build(objective_fn, constraints=[], k_inner=k_inner, validate=False)


def build_multi_agent(
    agent_specs: Dict[int, Tuple[Callable, Optional[Sequence[ConstraintSpec]]]],
    *,
    k_inner: float = 0.1,
    penalize_only_soft: bool = True,
    validate: bool = True,
    obj_transform: Union[str, Tuple] = "standard",
) -> Dict[int, Callable]:
    """Build standalone multi-agent cost wrappers."""
    result = {}
    for agent_id, (objective_fn, constraints) in agent_specs.items():
        base_fn = build(
            objective_fn,
            constraints,
            k_inner=k_inner,
            penalize_only_soft=penalize_only_soft,
            validate=validate,
            jit_cost=False,
            obj_transform=obj_transform,
        )

        def _wrap(base):
            def agent_fn(agent_idx, joint_x, ctx):
                _ = agent_idx
                return base(joint_x, ctx)

            return agent_fn

        result[agent_id] = _wrap(base_fn)
    return result


def _validate_constraints(constraints: Sequence[ConstraintSpec]) -> None:
    for idx, spec in enumerate(constraints):
        label = f"constraint[{idx}] ({type(spec).__name__}, priority={spec.priority})"
        if isinstance(spec, Deterministic) and spec.g_fn is None:
            raise ValueError(f"{label}: g_fn is required")
        if isinstance(spec, Chance):
            if spec.g_fn is None:
                raise ValueError(f"{label}: g_fn is required")
            if spec.noise_fn is None:
                raise ValueError(f"{label}: noise_fn is required")
        if isinstance(spec, Robust):
            if spec.g_fn is None:
                raise ValueError(f"{label}: g_fn is required")
            if spec.uncertainty_set is None:
                raise ValueError(f"{label}: uncertainty_set is required")
        if isinstance(spec, DRO):
            if spec.g_fn is None:
                raise ValueError(f"{label}: g_fn is required")
            if not spec.ambiguity_set:
                raise ValueError(f"{label}: ambiguity_set is required")


def quick_check(
    cost_fn: Callable[[jax.Array, Any], jax.Array],
    x_samples: Sequence[jax.Array],
    ctx: Any = None,
) -> Dict[str, Any]:
    """Evaluate representative samples and report output distinguishability."""
    eps_f32 = 6e-8
    values = []
    for x in x_samples:
        try:
            values.append(float(cost_fn(x, ctx)))
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    output_range = (min(values), max(values))
    distinguishable = int((output_range[1] - output_range[0]) / eps_f32)
    return {
        "ok": distinguishable > 100,
        "output_range": output_range,
        "distinguishable_values": distinguishable,
        "samples": values,
    }


def autodelta(constraints: Sequence[ConstraintSpec]) -> List[ConstraintSpec]:
    """Assign default tunable-hard deltas in place and return a list copy."""
    for spec in constraints:
        if spec.mode == "tunable" and spec.tune_preset == "__hard__" and spec.delta_soft is None:
            spec.delta_soft = TUNE_PRESETS["__hard__"][1]
    return list(constraints)

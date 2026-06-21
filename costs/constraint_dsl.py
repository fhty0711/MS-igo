"""Constraint-to-cost helpers for MGIGO black-box objectives.

The module mirrors the Constran idea: users write objective functions and
violations in the form ``g(x, ctx) <= 0``. ``build`` assembles them into a
JAX-compatible scalar cost using log compression, saturation, and priority
nesting.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import jax
import jax.numpy as jnp
from jax import lax, random, vmap


@jax.jit
def sigma_k(x: jax.Array, k: float = 1.0) -> jax.Array:
    """Odd saturation: sigma_k(x) = kx / sqrt(1 + (kx)^2)."""
    kx = k * x
    return kx / jnp.sqrt(1.0 + kx ** 2)


@jax.jit
def log_transform(x: jax.Array) -> jax.Array:
    """Odd log compression that preserves the sign of raw values."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


@dataclass(kw_only=True)
class ConstraintSpec:
    """Base specification for one constraint layer.

    g_fn should return a scalar violation where ``g <= 0`` means satisfied and
    ``g > 0`` means violated. priority controls nesting order; lower-priority
    terms are saturated before higher-priority terms are applied.

    mode:
      hard    - switches to a high layer when violated.
      tunable - smooth high-priority penalty for important but adjustable goals.
      soft    - regular additive/saturated violation layer.
    """

    mode: str = "soft"
    priority: int = 1
    delta: Optional[float] = None
    delta_soft: Optional[float] = None
    beta: Optional[float] = None

    def __post_init__(self):
        if self.mode not in ("hard", "soft", "tunable"):
            raise ValueError(
                f"mode must be 'hard', 'soft', or 'tunable', got {self.mode!r}"
            )
        if self.priority < 1:
            raise ValueError(f"priority must be >= 1, got {self.priority}")


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


def _make_violation_fn(spec: ConstraintSpec) -> Callable:
    """Convert a ConstraintSpec into a deterministic violation function."""
    if isinstance(spec, Deterministic):
        return spec.g_fn

    if isinstance(spec, Chance):
        g_fn = spec.g_fn
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
        g_fn = spec.g_fn
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
        g_fn = spec.g_fn
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


def _assemble_nest(
    objective_fn: Callable[[jax.Array, Any], jax.Array],
    layers: List[Tuple[int, str, dict, Callable]],
    k_inner: float,
    penalize_only_soft: bool,
) -> Callable[[jax.Array, Any], jax.Array]:
    """Assemble objective and constraint layers into one scalar black-box cost."""
    def cost_fn(x, ctx):
        inner = sigma_k(log_transform(objective_fn(x, ctx)), k=k_inner)

        for _priority, mode, params, viol_fn in layers:
            g_raw = viol_fn(x, ctx)
            if mode == "hard":
                delta = params.get("delta", 3.0)
                inner = sigma_k(
                    jnp.where(g_raw > 0.0, log_transform(g_raw) + delta, inner)
                )
            elif mode == "tunable":
                delta_soft = params.get("delta_soft", 2.0)
                beta = params.get("beta", 5.0)
                t_val = log_transform(g_raw)
                if penalize_only_soft:
                    t_val = jnp.maximum(0.0, t_val)
                inner = sigma_k(delta_soft * sigma_k(beta * t_val) + inner)
            else:
                t_val = log_transform(g_raw)
                if penalize_only_soft:
                    t_val = jnp.maximum(0.0, t_val)
                inner = sigma_k(t_val + inner)

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
) -> Callable[[jax.Array, Any], jax.Array]:
    """Build a JAX-compatible scalar cost from objective and constraints.

    objective_fn and all constraints receive the same ``(x, ctx)`` pair. In
    this project x is the flattened joint MGIGO sample and ctx contains decoded
    rollouts or other cached values.
    """
    constraints = [] if constraints is None else list(constraints)
    if validate:
        _validate_constraints(constraints)

    specs_sorted = sorted(constraints, key=lambda spec: spec.priority, reverse=True)
    hard_specs = [spec for spec in specs_sorted if spec.mode == "hard"]
    highest_hard_priority = (
        min(spec.priority for spec in hard_specs) if hard_specs else None
    )

    layers = []
    for spec in specs_sorted:
        viol_fn = _make_violation_fn(spec)
        params = {}
        if spec.mode == "hard":
            if spec.delta is not None:
                params["delta"] = spec.delta
            elif spec.priority == highest_hard_priority:
                params["delta"] = 1.5
            else:
                params["delta"] = 3.0
        elif spec.mode == "tunable":
            params["delta_soft"] = (
                spec.delta_soft if spec.delta_soft is not None else 2.0
            )
            params["beta"] = spec.beta if spec.beta is not None else 5.0
        layers.append((spec.priority, spec.mode, params, viol_fn))

    cost_fn = _assemble_nest(
        objective_fn,
        layers,
        k_inner=k_inner,
        penalize_only_soft=penalize_only_soft,
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
) -> Dict[int, Callable]:
    """Build standalone multi-agent cost wrappers.

    The current project usually registers one cost per agent instead, but this
    helper is kept for compatibility with the Constran API.
    """
    result = {}
    for agent_id, (objective_fn, constraints) in agent_specs.items():
        base_fn = build(
            objective_fn,
            constraints,
            k_inner=k_inner,
            penalize_only_soft=penalize_only_soft,
            validate=validate,
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

    specs_by_priority = sorted(constraints, key=lambda spec: spec.priority)
    seen_soft = False
    for spec in specs_by_priority:
        if spec.mode in ("soft", "tunable"):
            seen_soft = True
        elif spec.mode == "hard" and seen_soft:
            warnings.warn(
                f"Hard constraint (priority={spec.priority}) is inside a "
                "soft/tunable layer. Hard layers should generally be outermost."
            )
            break


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
    """Assign default hard-layer deltas in place and return a list copy."""
    hard_specs = [spec for spec in constraints if spec.mode == "hard"]
    if not hard_specs:
        return list(constraints)

    highest_priority = min(spec.priority for spec in hard_specs)
    for spec in hard_specs:
        if spec.delta is None:
            spec.delta = 1.5 if spec.priority == highest_priority else 3.0
    return list(constraints)

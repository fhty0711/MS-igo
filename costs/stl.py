"""Signal Temporal Logic robustness helpers.

The cost modules use this file to define STL formulas explicitly and then pass
their signed violations to the MGIGO constraint wrapper.  A formula is satisfied
when its robustness is non-negative; ``violation`` converts robustness into the
``g(x, ctx) <= 0`` convention used by ``constraint_dsl``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from jax import lax
import jax.numpy as jnp


RobustnessFn = Callable[[object, object], jnp.ndarray]
Interval = Optional[Tuple[int, Optional[int]]]


@dataclass(frozen=True)
class Formula:
    """A JAX-compatible STL formula with quantitative robustness semantics."""

    name: str
    robustness_fn: RobustnessFn

    def robustness(self, x, ctx):
        """Return the signed robustness of this formula for one candidate."""
        return self.robustness_fn(x, ctx)


def predicate(name: str, robustness_fn: RobustnessFn) -> Formula:
    """Create an atomic STL predicate from a signed robustness function."""
    return Formula(name=name, robustness_fn=robustness_fn)


def _window(signal, interval: Interval):
    """Slice a robustness signal according to a discrete STL time interval."""
    if interval is None:
        return signal
    start, end = interval
    return signal[start:end]


def always(phi: Formula, interval: Interval = None, name: Optional[str] = None) -> Formula:
    """STL globally operator: rho(G phi) = min_t rho(phi, t)."""

    def robustness_fn(x, ctx):
        return jnp.min(_window(phi.robustness(x, ctx), interval))

    return Formula(name=name or f"G({phi.name})", robustness_fn=robustness_fn)


def eventually(phi: Formula, interval: Interval = None, name: Optional[str] = None) -> Formula:
    """STL eventually operator: rho(F phi) = max_t rho(phi, t)."""

    def robustness_fn(x, ctx):
        return jnp.max(_window(phi.robustness(x, ctx), interval))

    return Formula(name=name or f"F({phi.name})", robustness_fn=robustness_fn)


def all_of(children: Sequence[Formula], name: Optional[str] = None) -> Formula:
    """STL conjunction: rho(phi and psi) = min(rho(phi), rho(psi))."""
    children = tuple(children)

    def robustness_fn(x, ctx):
        values = jnp.broadcast_arrays(*[child.robustness(x, ctx) for child in children])
        values = jnp.stack(values, axis=0)
        return jnp.min(values, axis=0)

    label = " & ".join(child.name for child in children)
    return Formula(name=name or f"({label})", robustness_fn=robustness_fn)


def any_of(children: Sequence[Formula], name: Optional[str] = None) -> Formula:
    """STL disjunction: rho(phi or psi) = max(rho(phi), rho(psi))."""
    children = tuple(children)

    def robustness_fn(x, ctx):
        values = jnp.broadcast_arrays(*[child.robustness(x, ctx) for child in children])
        values = jnp.stack(values, axis=0)
        return jnp.max(values, axis=0)

    label = " | ".join(child.name for child in children)
    return Formula(name=name or f"({label})", robustness_fn=robustness_fn)


def negate(phi: Formula, name: Optional[str] = None) -> Formula:
    """STL negation: rho(not phi) = -rho(phi)."""

    def robustness_fn(x, ctx):
        return -phi.robustness(x, ctx)

    return Formula(name=name or f"!({phi.name})", robustness_fn=robustness_fn)


def implies(lhs: Formula, rhs: Formula, name: Optional[str] = None) -> Formula:
    """STL implication, represented as ``not lhs or rhs``."""
    return any_of((negate(lhs), rhs), name=name or f"({lhs.name} -> {rhs.name})")


def until(lhs: Formula, rhs: Formula, interval: Interval = None, name: Optional[str] = None) -> Formula:
    """Discrete STL until operator.

    rho(phi U_[a,b] psi, 0) = max_t min(rho(psi,t), min_tau<t rho(phi,tau)).
    This implementation is intentionally simple and intended for finite MPC
    horizons, where the evaluated traces are short.
    """

    def robustness_fn(x, ctx):
        lhs_trace = _window(lhs.robustness(x, ctx), interval)
        rhs_trace = _window(rhs.robustness(x, ctx), interval)
        running_lhs = lax.associative_scan(jnp.minimum, lhs_trace)
        return jnp.max(jnp.minimum(rhs_trace, running_lhs))

    return Formula(name=name or f"({lhs.name} U {rhs.name})", robustness_fn=robustness_fn)


def violation(phi: Formula, x, ctx):
    """Return the signed constraint violation for a formula.

    The result is non-positive when the STL formula is satisfied and positive
    when it is violated, matching the ``g(x, ctx) <= 0`` constraint convention.
    """
    return -phi.robustness(x, ctx)

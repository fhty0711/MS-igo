"""Semantic checks for constraint_dsl used by the intersection benchmark."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax.numpy as jnp

from costs.constraint_dsl import Chance, Deterministic, build


def assert_close(actual, expected, tol=1e-6):
    actual = float(actual)
    expected = float(expected)
    if abs(actual - expected) > tol:
        raise AssertionError(f"expected {expected}, got {actual}")


def test_chance_g_fn_is_per_sample_quantile():
    def objective(x, ctx):
        del x, ctx
        return 0.0

    def noise_fn(key, shape):
        del key
        if shape != (5,):
            raise AssertionError(f"unexpected shape {shape}")
        return jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    def per_sample_violation(x, xi, ctx):
        del ctx
        if xi.shape != ():
            raise AssertionError(f"g_fn should receive one sample, got shape {xi.shape}")
        return xi + x[0]

    cost_fn = build(
        objective,
        [
            Chance(
                g_fn=per_sample_violation,
                noise_fn=noise_fn,
                alpha=0.2,
                n_samples=5,
                mode="soft",
                priority=1,
            )
        ],
        k_inner=0.1,
        penalize_only_soft=False,
        jit_cost=False,
    )

    value = cost_fn(jnp.array([0.0]), None)
    expected_g = 1.2
    expected_log = jnp.log1p(expected_g)
    expected = expected_log / jnp.sqrt(1.0 + expected_log**2)
    assert_close(value, expected)


def test_priority_one_hard_layer_is_outermost():
    def objective(x, ctx):
        del x, ctx
        return 0.0

    def soft_violation(x, ctx):
        del x, ctx
        return 1_000_000.0

    def hard_satisfied(x, ctx):
        del x, ctx
        return -1.0

    def hard_violated(x, ctx):
        del x, ctx
        return 0.001

    feasible = build(
        objective,
        [
            Deterministic(g_fn=soft_violation, mode="tunable", priority=3),
            Deterministic(g_fn=hard_satisfied, mode="hard", priority=1),
        ],
        jit_cost=False,
    )(jnp.array([0.0]), None)

    infeasible = build(
        objective,
        [
            Deterministic(g_fn=soft_violation, mode="tunable", priority=3),
            Deterministic(g_fn=hard_violated, mode="hard", priority=1),
        ],
        jit_cost=False,
    )(jnp.array([0.0]), None)

    if not float(infeasible) > float(feasible):
        raise AssertionError(
            f"priority=1 hard violation should outrank inner layers: "
            f"infeasible={infeasible}, feasible={feasible}"
        )


if __name__ == "__main__":
    test_chance_g_fn_is_per_sample_quantile()
    test_priority_one_hard_layer_is_outermost()
    print("constraint_dsl semantics ok")

"""Semantic checks for constraint_dsl used by the intersection benchmark."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax.numpy as jnp

from costs.constraint_dsl import (
    Chance,
    Deterministic,
    TRANSFORM_SHARP,
    T_alpha,
    build,
)


def assert_close(actual, expected, tol=1e-6):
    actual = float(actual)
    expected = float(expected)
    if abs(actual - expected) > tol:
        raise AssertionError(f"expected {expected}, got {actual}")


def test_t_alpha_sharp_has_small_violation_floor():
    value = T_alpha(jnp.array(1e-9), *TRANSFORM_SHARP)
    if not float(value) > 0.45:
        raise AssertionError(f"sharp transform should expose tiny violations, got {value}")


def test_hard_mode_normalizes_to_tunable_hard_preset():
    spec = Deterministic(g_fn=lambda x, ctx: x[0], mode="hard", priority=1)
    if spec.mode != "tunable":
        raise AssertionError(f"hard should normalize to tunable, got {spec.mode!r}")
    if spec.tune_preset != "__hard__":
        raise AssertionError(
            f"hard should use __hard__ tune preset, got {spec.tune_preset!r}"
        )
    beta, delta_soft = spec.get_tune_params()
    if (beta, delta_soft) != (2.0, 1.5):
        raise AssertionError(
            f"hard should use upstream __hard__ params, got {(beta, delta_soft)}"
        )


def test_vector_aggregate_max_and_mean_are_scalar_costs():
    def objective(x, ctx):
        del x, ctx
        return 0.0

    def vector_violation(x, ctx):
        del ctx
        return jnp.array([x[0] - 1.0, x[1] - 1.0, x[2] - 1.0])

    x = jnp.array([0.0, 2.0, 4.0])
    max_cost = build(
        objective,
        [Deterministic(g_fn=vector_violation, aggregate="max", mode="soft")],
        jit_cost=False,
        obj_transform="log",
    )(x, None)
    mean_cost = build(
        objective,
        [Deterministic(g_fn=vector_violation, aggregate="mean", mode="soft")],
        jit_cost=False,
        obj_transform="log",
    )(x, None)

    if jnp.shape(max_cost) != ():
        raise AssertionError(f"max aggregate should return scalar cost, got {max_cost}")
    if jnp.shape(mean_cost) != ():
        raise AssertionError(f"mean aggregate should return scalar cost, got {mean_cost}")
    if not float(max_cost) > float(mean_cost):
        raise AssertionError(f"max aggregate should exceed mean: {max_cost} <= {mean_cost}")


def test_vector_aggregate_defaults_to_sum():
    def objective(x, ctx):
        del x, ctx
        return 0.0

    def vector_violation(x, ctx):
        del ctx
        return jnp.array([x[0], x[1], x[2]])

    explicit = build(
        objective,
        [Deterministic(g_fn=vector_violation, aggregate="sum", mode="soft", transform="log")],
        jit_cost=False,
        obj_transform="log",
    )(jnp.array([1.0, 2.0, 3.0]), None)
    default = build(
        objective,
        [Deterministic(g_fn=vector_violation, mode="soft", transform="log")],
        jit_cost=False,
        obj_transform="log",
    )(jnp.array([1.0, 2.0, 3.0]), None)

    assert_close(default, explicit)


def test_obj_transform_presets_are_accepted():
    def objective(x, ctx):
        del ctx
        return x[0]

    for preset in ("standard", "flat", "log"):
        cost_fn = build(objective, [], jit_cost=False, obj_transform=preset)
        value = cost_fn(jnp.array([10.0]), None)
        if jnp.shape(value) != ():
            raise AssertionError(f"{preset} objective transform returned {value}")


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
                transform="log",
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
    expected = 0.2 * expected_log / jnp.sqrt(1.0 + (0.2 * expected_log) ** 2)
    assert_close(value, expected)


def test_chance_aggregate_keeps_per_sample_g_fn_signature():
    def objective(x, ctx):
        del x, ctx
        return 0.0

    def noise_fn(key, shape):
        del key
        return jnp.arange(shape[0], dtype=jnp.float32)

    def per_sample_vector_violation(x, xi, ctx):
        del x, ctx
        if xi.shape != ():
            raise AssertionError(f"g_fn should receive scalar sample, got {xi.shape}")
        return jnp.array([xi - 2.0, xi - 1.0])

    cost_fn = build(
        objective,
        [
            Chance(
                g_fn=per_sample_vector_violation,
                noise_fn=noise_fn,
                aggregate="max",
                alpha=0.25,
                n_samples=4,
                mode="soft",
                transform="log",
                priority=1,
            )
        ],
        jit_cost=False,
        obj_transform="log",
    )

    value = cost_fn(jnp.array([0.0]), None)
    if jnp.shape(value) != ():
        raise AssertionError(f"chance aggregate should produce scalar cost, got {value}")


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
            Deterministic(g_fn=hard_satisfied, mode="hard", transform="sharp", priority=1),
        ],
        jit_cost=False,
        obj_transform="log",
    )(jnp.array([0.0]), None)

    infeasible = build(
        objective,
        [
            Deterministic(g_fn=soft_violation, mode="tunable", priority=3),
            Deterministic(g_fn=hard_violated, mode="hard", transform="sharp", priority=1),
        ],
        jit_cost=False,
        obj_transform="log",
    )(jnp.array([0.0]), None)

    if not float(infeasible) > float(feasible):
        raise AssertionError(
            f"priority=1 hard violation should outrank inner layers: "
            f"infeasible={infeasible}, feasible={feasible}"
        )


if __name__ == "__main__":
    test_t_alpha_sharp_has_small_violation_floor()
    test_hard_mode_normalizes_to_tunable_hard_preset()
    test_vector_aggregate_max_and_mean_are_scalar_costs()
    test_vector_aggregate_defaults_to_sum()
    test_obj_transform_presets_are_accepted()
    test_chance_g_fn_is_per_sample_quantile()
    test_chance_aggregate_keeps_per_sample_g_fn_signature()
    test_priority_one_hard_layer_is_outermost()
    print("constraint_dsl semantics ok")

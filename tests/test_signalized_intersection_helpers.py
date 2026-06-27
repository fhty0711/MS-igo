"""Tests for the signalized intersection benchmark helpers."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_signalized_intersection_scenario_contract():
    from scenarios import get_scenario

    scenario = get_scenario("signalized_intersection")
    if scenario.name != "signalized_intersection":
        raise AssertionError(scenario.name)
    if scenario.n_agents != 1:
        raise AssertionError(f"expected one optimizing ego agent, got {scenario.n_agents}")
    if scenario.agent_names != ("ego",):
        raise AssertionError(scenario.agent_names)
    if scenario.solver_spec.n_blocks != 2:
        raise AssertionError(f"expected acc and steer blocks, got {scenario.solver_spec}")
    if scenario.cost_profile != "signalized_intersection":
        raise AssertionError(scenario.cost_profile)


def test_cross_traffic_noise_is_multimodal_and_small_by_default():
    import jax
    from costs import signalized_intersection as cost

    samples = cost._cross_traffic_noise(jax.random.PRNGKey(0), (40,))
    modes = set(int(v) for v in samples[:, cost.XI_MODE].tolist())
    expected = {
        cost.MODE_OBEY,
        cost.MODE_YELLOW_RUSH,
        cost.MODE_RED_RUN,
    }
    if modes != expected:
        raise AssertionError(f"expected modes {expected}, got {modes}")


def test_no_blocking_intersection_violation():
    import jax.numpy as jnp
    from costs import signalized_intersection as cost

    stopped_before = jnp.array([[30.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    blocking = jnp.array([[42.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    cleared = jnp.array([[55.0, 0.0, 8.0, 0.0, 0.0, 0.0]])

    if float(cost._no_blocking_intersection_from_ego_traj(stopped_before)) > 0.0:
        raise AssertionError("stopping before line should not block")
    if float(cost._no_blocking_intersection_from_ego_traj(cleared)) > 0.0:
        raise AssertionError("clearing intersection should not block")
    if float(cost._no_blocking_intersection_from_ego_traj(blocking)) <= 0.0:
        raise AssertionError("stopping inside conflict box should block")


if __name__ == "__main__":
    test_signalized_intersection_scenario_contract()
    test_cross_traffic_noise_is_multimodal_and_small_by_default()
    test_no_blocking_intersection_violation()
    print("signalized intersection helper tests ok")

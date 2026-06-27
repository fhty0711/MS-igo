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


if __name__ == "__main__":
    test_signalized_intersection_scenario_contract()
    print("signalized intersection helper tests ok")

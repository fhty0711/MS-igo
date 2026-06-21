"""Scenario registry.

Add new scenarios here by importing their factory and adding one entry to
SCENARIO_FACTORIES. Keep scenario-specific parameters inside each scenario file.
"""

from typing import Callable, Dict

from .highway_merge import make_scenario as make_highway_merge
from .spec import ScenarioSpec


SCENARIO_FACTORIES: Dict[str, Callable[[], ScenarioSpec]] = {
    "highway_merge": make_highway_merge,
}

SCENARIOS: Dict[str, ScenarioSpec] = {
    name: factory() for name, factory in SCENARIO_FACTORIES.items()
}


def get_scenario(name: str = "highway_merge") -> ScenarioSpec:
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        available = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unknown scenario {name!r}. Available: {available}") from exc

"""Scenario registry.

Add new scenarios here by importing their factory and adding one entry to
SCENARIO_FACTORIES. Keep scenario-specific parameters inside each scenario file.
"""

from typing import Callable, Dict

from .borrow_overtake import (
    make_blocked_scenario as make_borrow_overtake_blocked,
    make_critical_scenario as make_borrow_overtake_critical,
    make_safe_scenario as make_borrow_overtake_safe,
    make_scenario as make_borrow_overtake,
)
from .highway_merge import make_scenario as make_highway_merge
from .spec import ScenarioSpec


SCENARIO_FACTORIES: Dict[str, Callable[[], ScenarioSpec]] = {
    "borrow_overtake": make_borrow_overtake,
    "borrow_overtake_blocked": make_borrow_overtake_blocked,
    "borrow_overtake_critical": make_borrow_overtake_critical,
    "borrow_overtake_safe": make_borrow_overtake_safe,
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

"""Cost profile registry."""

from typing import Callable, Dict, Optional, Tuple

from . import highway_merge, highway_merge_baseline

CostFunctions = Tuple[Callable, ...]
DEFAULT_COST_PROFILE = "highway_merge"


COST_PROFILES: Dict[str, CostFunctions] = {
    # Wrapper / constraint-DSL cost. This is the default profile.
    "highway_merge": (
        highway_merge.ego_cost,
        highway_merge.front_cost,
        highway_merge.rear_cost,
    ),
    # Baseline hand-written hierarchical cost, kept for A/B testing only.
    "highway_merge_baseline": (
        highway_merge_baseline.ego_cost,
        highway_merge_baseline.front_cost,
        highway_merge_baseline.rear_cost,
    ),
}


def get_cost_functions(profile: Optional[str] = None) -> CostFunctions:
    profile = profile or DEFAULT_COST_PROFILE
    try:
        return COST_PROFILES[profile]
    except KeyError as exc:
        available = ", ".join(sorted(COST_PROFILES))
        raise ValueError(f"Unknown cost profile {profile!r}. Available: {available}") from exc

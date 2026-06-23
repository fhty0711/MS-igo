"""Cost profile registry."""

from typing import Callable, Dict, Optional, Tuple

from . import (
    borrow_overtake,
    borrow_overtake_baseline,
    borrow_overtake_matched,
    highway_merge,
    highway_merge_baseline,
)

CostFunctions = Tuple[Callable, ...]
DEFAULT_COST_PROFILE = "highway_merge"


COST_PROFILES: Dict[str, CostFunctions] = {
    # Borrow-lane overtaking with STL safety layers.
    "borrow_overtake": (
        borrow_overtake.ego_cost,
        borrow_overtake.slow_lead_cost,
        borrow_overtake.oncoming_cost,
    ),
    # Same borrow-overtake terms, combined by the original hand-written hierarchy.
    "borrow_overtake_baseline": (
        borrow_overtake_baseline.ego_cost,
        borrow_overtake_baseline.slow_lead_cost,
        borrow_overtake_baseline.oncoming_cost,
    ),
    # Hand-written assembler matched to the wrapper transformation.
    "borrow_overtake_matched": (
        borrow_overtake_matched.ego_cost,
        borrow_overtake_matched.slow_lead_cost,
        borrow_overtake_matched.oncoming_cost,
    ),
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

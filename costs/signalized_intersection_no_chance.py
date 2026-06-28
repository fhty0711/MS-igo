"""Signalized-intersection ablation without probabilistic cross-traffic risk."""

from .constraint_dsl import build
from .signalized_intersection import (
    _constraint_specs_by_name,
    _ego_objective,
    _shared_context,
)


_ego_base_cost = build(
    _ego_objective,
    _constraint_specs_by_name(
        (
            "red_light",
            "road_boundary",
            "no_blocking_intersection",
            "dilemma_task",
        )
    ),
    k_inner=0.1,
    penalize_only_soft=True,
    jit_cost=False,
    obj_transform="standard",
)


def ego_cost(joint_sample_flat, context_arr):
    return _ego_base_cost(joint_sample_flat, _shared_context(joint_sample_flat, context_arr))

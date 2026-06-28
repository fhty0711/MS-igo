"""Signalized-intersection ablation with the stop/pass dilemma as soft."""

from .constraint_dsl import Deterministic, build
from .signalized_intersection import (
    _constraint_specs_by_name,
    _dilemma_task_violation,
    _ego_objective,
    _shared_context,
)


_ego_base_cost = build(
    _ego_objective,
    [
        *_constraint_specs_by_name(
            (
                "red_light",
                "road_boundary",
                "no_blocking_intersection",
                "cross_traffic_chance",
            )
        ),
        Deterministic(
            g_fn=_dilemma_task_violation,
            mode="soft",
            priority=3,
            transform="wide",
        ),
    ],
    k_inner=0.1,
    penalize_only_soft=True,
    jit_cost=False,
    obj_transform="standard",
)


def ego_cost(joint_sample_flat, context_arr):
    return _ego_base_cost(joint_sample_flat, _shared_context(joint_sample_flat, context_arr))

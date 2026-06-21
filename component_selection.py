"""Component selection utilities for blockwise MGIGO outputs."""

from itertools import product

import jax.numpy as jnp
import numpy as np

from config import HYSTERESIS_BIAS, K_COMP


SWITCH_PENALTY = 1.0


def blocks_by_agent(block_to_agent):
    groups = {}
    for block_idx, agent_idx in enumerate(block_to_agent):
        groups.setdefault(int(agent_idx), []).append(block_idx)
    return groups


def select_components_by_weight(final_pi_np, block_to_agent, prev_exec_k=None):
    """Select one mixture component per block by agent-level joint log weight."""
    log_pi = np.log(np.clip(final_pi_np, 1e-20, 1.0))
    selected = np.zeros((len(block_to_agent),), dtype=np.int32)

    for owned_blocks in blocks_by_agent(block_to_agent).values():
        best_score = -float("inf")
        best_combo = None
        for combo in product(range(K_COMP), repeat=len(owned_blocks)):
            score = 0.0
            for block_idx, comp_idx in zip(owned_blocks, combo):
                score += float(log_pi[block_idx, comp_idx])
                if prev_exec_k is not None and comp_idx == int(prev_exec_k[block_idx]):
                    score += HYSTERESIS_BIAS
            if score > best_score:
                best_score = score
                best_combo = combo

        for block_idx, comp_idx in zip(owned_blocks, best_combo):
            selected[block_idx] = comp_idx

    return selected


def select_components_by_cost(
    final_mu_np,
    context_jax,
    fitness_fn,
    block_to_agent,
    prev_exec_k=None,
):
    """Greedily select block components by each agent's evaluated cost."""
    block_to_agent = tuple(int(idx) for idx in block_to_agent)
    selected = np.zeros((len(block_to_agent),), dtype=np.int32)
    groups = blocks_by_agent(block_to_agent)

    for agent_idx, owned_blocks in sorted(groups.items()):
        best_val = float("inf")
        best_combo = None
        for combo in product(range(K_COMP), repeat=len(owned_blocks)):
            trial = selected.copy()
            for block_idx, comp_idx in zip(owned_blocks, combo):
                trial[block_idx] = comp_idx

            parts = [
                final_mu_np[block_idx, int(trial[block_idx])]
                for block_idx in range(len(block_to_agent))
            ]
            joint_sample = jnp.concatenate(parts)
            cost = float(fitness_fn(agent_idx, joint_sample, context_jax))

            if prev_exec_k is not None:
                for block_idx, comp_idx in zip(owned_blocks, combo):
                    if comp_idx != int(prev_exec_k[block_idx]):
                        cost += SWITCH_PENALTY
            if cost < best_val:
                best_val = cost
                best_combo = combo

        for block_idx, comp_idx in zip(owned_blocks, best_combo):
            selected[block_idx] = comp_idx

    return selected

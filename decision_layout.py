"""Utilities for decoding MGIGO blocks into named decisions.

MGIGO operates on anonymous block vectors. Scenario/cost/runtime code is easier
to read with named decisions such as ``ego_acc`` and ``ego_steer``. BlockDecoder
is the bridge between those two views.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecisionSlice:
    """Location of one named decision inside one padded block vector."""

    block_index: int
    start: int
    stop: int
    shape: tuple


class BlockDecoder:
    """Encode/decode between solver block arrays and named scenario decisions."""

    def __init__(self, scenario):
        self.scenario = scenario
        # Blocks must be stable and dense because solver arrays are indexed by
        # block_index directly: component_means[block_index, component_index].
        ordered_blocks = sorted(scenario.blocks, key=lambda item: item.block_index)
        block_indices = tuple(block.block_index for block in ordered_blocks)
        expected_indices = tuple(range(len(ordered_blocks)))
        if block_indices != expected_indices:
            raise ValueError(
                "Block indices must be contiguous and zero-based: "
                f"got {block_indices}, expected {expected_indices}"
            )
        decisions_by_name = {decision.name: decision for decision in scenario.decisions}
        self._slices = {}
        for block in ordered_blocks:
            offset = 0
            for decision_name in block.decision_names:
                decision = decisions_by_name[decision_name]
                # Multiple decisions may be packed into one block. The slice
                # records how to split that flat vector back into named arrays.
                self._slices[decision_name] = DecisionSlice(
                    block_index=block.block_index,
                    start=offset,
                    stop=offset + decision.dim,
                    shape=decision.shape,
                )
                offset += decision.dim
            expected_dim = scenario.block_dims[block.block_index]
            if offset != expected_dim:
                raise ValueError(
                    f"Block {block.name!r} declares dim {expected_dim}, "
                    f"but its decisions sum to {offset}"
                )

    @property
    def decision_names(self):
        return tuple(self._slices)

    @property
    def max_block_dim(self):
        return max(self.scenario.block_dims)

    def decode(self, blocks):
        """Convert ``(n_blocks, solver_width)`` arrays into named decisions."""
        decoded = {}
        for name, slc in self._slices.items():
            if blocks.shape[1] < slc.stop:
                raise ValueError(
                    f"Block {slc.block_index} has width {blocks.shape[1]}, "
                    f"but decision {name!r} needs {slc.stop}"
                )
            flat = blocks[slc.block_index][slc.start:slc.stop]
            decoded[name] = flat.reshape(slc.shape)
        return decoded

    def encode(self, decision_sequences, dtype=np.float64, width=None):
        """Pack named decisions into padded block arrays.

        width is usually max(block_dims), matching the padded width used by the
        MGIGO solver for heterogeneous block dimensions.
        """
        width = self.max_block_dim if width is None else int(width)
        encoded = np.zeros((self.scenario.n_control_blocks, width), dtype=dtype)
        for name, slc in self._slices.items():
            value = np.asarray(decision_sequences[name], dtype=dtype).reshape(-1)
            expected = slc.stop - slc.start
            if value.size != expected:
                raise ValueError(
                    f"Decision {name!r} has flat size {value.size}, "
                    f"expected {expected}"
                )
            encoded[slc.block_index, slc.start:slc.stop] = value
        return encoded

    def shift_blocks(self, blocks):
        """Shift each time-sequence decision by one MPC step and repack blocks.

        Used by warm start: the plan executed at index 0 is dropped, future
        controls move forward, and the last control is reset to zero.
        """
        decisions = self.decode(blocks)
        shifted = {}
        for name, value in decisions.items():
            arr = np.asarray(value)
            if arr.ndim == 0:
                shifted[name] = np.zeros_like(arr)
            else:
                shifted[name] = np.concatenate(
                    [arr[1:], np.zeros_like(arr[:1])],
                    axis=0,
                )
        return self.encode(shifted, dtype=np.asarray(blocks).dtype, width=np.asarray(blocks).shape[1])

    def select_blocks_from_components(self, component_means, component_indices):
        """Return one selected sequence per block from GMM component means."""
        selected = []
        for block in sorted(self.scenario.blocks, key=lambda item: item.block_index):
            selected.append(component_means[block.block_index, component_indices[block.block_index]])
        return selected

    def block_sequences_by_name(self, selected_blocks):
        """Return selected raw block vectors keyed by BlockSpec.name."""
        return {
            block.name: selected_blocks[block.block_index]
            for block in sorted(self.scenario.blocks, key=lambda item: item.block_index)
        }

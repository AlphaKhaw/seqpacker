"""
PyTorch utilities for sequence packing in LLM training.

Converts seqpacker packing results into GPU-ready tensors for
packed training with Flash Attention, Megatron-LM, or similar frameworks.

This is a lazy-import module. Users opt in with::

    from seqpacker.torch_utils import pack_result_to_tensors, packed_collate_fn

Requires ``torch`` to be installed (not a seqpacker dependency).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from seqpacker import Pack, PackResult, pack_sequences

# ── Helper functions (private) ────────────────────────────────────────


def _compute_cu_seqlens(lengths: list[int]) -> torch.Tensor:
    """
    Compute cumulative sequence lengths for Flash Attention varlen API.

    Reimplements the Rust ``Pack::cu_seqlens`` logic in pure Python.
    Returns ``[0, len_0, len_0+len_1, ..., total]`` as an int32 tensor.

    Args:
        lengths (list[int]): Sequence lengths within a single pack.

    Returns:
        torch.Tensor: Int32 tensor of shape ``(len(lengths) + 1,)``.
    """
    cu = [0]
    offset = 0
    for length in lengths:
        offset += length
        cu.append(offset)
    return torch.tensor(cu, dtype=torch.int32)


def _compute_position_ids(lengths: list[int], max_len: int) -> torch.Tensor:
    """
    Compute per-sequence reset position IDs, padded to ``max_len``.

    Each sequence gets positions starting from 0. Padding positions
    are filled with 0.

    Args:
        lengths (list[int]): Sequence lengths within a single pack.
        max_len (int): Total length to pad to (pack capacity).

    Returns:
        torch.Tensor: Int64 tensor of shape ``(max_len,)``.
    """
    ids: list[int] = []
    for length in lengths:
        ids.extend(range(length))
    # Pad to max_len
    padding_needed = max_len - len(ids)
    if padding_needed > 0:
        ids.extend([0] * padding_needed)
    return torch.tensor(ids, dtype=torch.long)


def _compute_labels(
    input_ids: torch.Tensor,
    lengths: list[int],
    *,
    label_padding_value: int = -100,
) -> torch.Tensor:
    """
    Compute shifted labels with masking at sequence boundaries and padding.

    For each sequence within a pack, the label is the next token (shifted
    left by one). The last token of each sequence gets ``label_padding_value``
    since there is no next token. Padding positions also get
    ``label_padding_value``.

    Args:
        input_ids (torch.Tensor): Padded input token IDs of shape ``(max_len,)``.
        lengths (list[int]): Sequence lengths within this pack.
        label_padding_value (int): Value for masked label positions (default: -100).

    Returns:
        torch.Tensor: Int64 tensor of shape ``(max_len,)`` with shifted labels.
    """
    max_len = input_ids.shape[0]
    labels = torch.full((max_len,), fill_value=label_padding_value, dtype=torch.long)

    offset = 0
    for length in lengths:
        if length > 1:
            # Shifted: label[i] = input_ids[i+1] for positions within the sequence
            labels[offset : offset + length - 1] = input_ids[
                offset + 1 : offset + length
            ]
        # Last token of each sequence stays as label_padding_value
        offset += length

    return labels


# ── Public API ────────────────────────────────────────────────────────


@dataclass
class PackedBatch:
    """
    All tensors for one packed training batch.

    Holds GPU-ready tensors suitable for Flash Attention varlen API,
    Megatron-LM packed training, or similar frameworks.
    """

    input_ids: torch.Tensor
    """
    Padded token IDs, shape ``(num_packs, max_seq_len)``.
    """

    cu_seqlens: list[torch.Tensor]
    """
    Cumulative sequence lengths per pack (int32), for Flash Attention varlen API.
    Shape: ``(num_packs + 1,)``.
    """

    max_seqlen: int
    """
    Maximum individual sequence length across all packs.
    """

    position_ids: torch.Tensor
    """
    Per-sequence reset position IDs, shape ``(num_packs, max_seq_len)``.
    """

    labels: torch.Tensor | None
    """
    Shifted labels with -100 at boundaries/padding, shape ``(num_packs, max_seq_len)``.
    """

    attention_mask: torch.Tensor | None
    """
    Binary mask (1=real, 0=padding), shape ``(num_packs, max_seq_len)``.
    """


def pack_result_to_tensors(
    result: PackResult,
    token_ids: Sequence[Sequence[int]],
    *,
    padding_value: int = 0,
    label_padding_value: int = -100,
    device: torch.device | str | None = None,
) -> PackedBatch:
    """
    Convert a packing result and token sequences into a PackedBatch.

    For each pack, looks up token IDs by ``sequence_ids``, concatenates
    them, pads to capacity, and generates all LLM training metadata tensors.

    Args:
        result (PackResult): Packing result from ``pack_sequences`` or ``Packer.pack``.
        token_ids (Sequence[Sequence[int]]): Token ID sequences indexed by original
            sequence ID. ``token_ids[i]`` corresponds to sequence ``i``.
        padding_value (int): Value for padding positions in ``input_ids`` (default: 0).
        label_padding_value (int): Value for masked label positions (default: -100).
        device (torch.device | str | None): Device for output tensors (default: CPU).

    Returns:
        PackedBatch: Dataclass with all packed training tensors.
    """
    packs: list[Pack] = result.packs
    num_packs = len(packs)

    # Determine capacity from the first pack's metadata
    # capacity = sum of lengths + padding = used + remaining
    # We use the packer's capacity which is consistent across packs.
    # Since PackResult doesn't expose capacity directly, infer from
    # the max of (pack.used + padding needed). The user passed capacity
    # to the packer, so all packs share the same capacity. We can get it
    # from the metrics: total_capacity = num_packs * capacity, and
    # total_tokens + padding_tokens = total_capacity.
    metrics = result.metrics
    if num_packs > 0:
        capacity = (metrics.total_tokens + metrics.padding_tokens) // num_packs
    else:
        return PackedBatch(
            input_ids=torch.empty((0, 0), dtype=torch.long, device=device),
            cu_seqlens=[],
            max_seqlen=0,
            position_ids=torch.empty((0, 0), dtype=torch.long, device=device),
            labels=torch.empty((0, 0), dtype=torch.long, device=device),
            attention_mask=torch.empty((0, 0), dtype=torch.long, device=device),
        )

    all_input_ids = torch.full(
        (num_packs, capacity),
        fill_value=padding_value,
        dtype=torch.long,
    )
    all_position_ids = torch.zeros((num_packs, capacity), dtype=torch.long)
    all_labels = torch.full(
        (num_packs, capacity),
        fill_value=label_padding_value,
        dtype=torch.long,
    )
    all_attention_mask = torch.zeros((num_packs, capacity), dtype=torch.long)
    all_cu_seqlens: list[torch.Tensor] = []
    global_max_seqlen = 0

    for i, pack in enumerate(packs):
        seq_ids = pack.sequence_ids
        lengths = pack.lengths

        # Concatenate token IDs for this pack
        tokens: list[int] = []
        for sid in seq_ids:
            tokens.extend(token_ids[sid])
        used = len(tokens)

        # Fill input_ids
        all_input_ids[i, :used] = torch.tensor(tokens, dtype=torch.long)

        # Fill attention_mask
        all_attention_mask[i, :used] = 1

        # Compute per-pack metadata
        all_cu_seqlens.append(_compute_cu_seqlens(lengths))
        all_position_ids[i] = _compute_position_ids(lengths, max_len=capacity)
        all_labels[i] = _compute_labels(
            all_input_ids[i],
            lengths,
            label_padding_value=label_padding_value,
        )

        pack_max = max(lengths) if lengths else 0
        if pack_max > global_max_seqlen:
            global_max_seqlen = pack_max

    # Move to device if specified
    if device is not None:
        all_input_ids = all_input_ids.to(device)
        all_position_ids = all_position_ids.to(device)
        all_labels = all_labels.to(device)
        all_attention_mask = all_attention_mask.to(device)
        all_cu_seqlens = [t.to(device) for t in all_cu_seqlens]

    return PackedBatch(
        input_ids=all_input_ids,
        cu_seqlens=all_cu_seqlens,
        max_seqlen=global_max_seqlen,
        position_ids=all_position_ids,
        labels=all_labels,
        attention_mask=all_attention_mask,
    )


def packed_collate_fn(
    capacity: int,
    *,
    strategy: str = "obfd",
    seed: int | None = None,
    padding_value: int = 0,
    label_padding_value: int = -100,
) -> Callable[[list[list[int]]], PackedBatch]:
    """
    Create a collate function for ``torch.utils.data.DataLoader``.

    Returns a callable that packs a batch of token sequences and converts
    them into a ``PackedBatch``. Usage::

        from torch.utils.data import DataLoader
        from seqpacker.torch_utils import packed_collate_fn

        collate = packed_collate_fn(capacity=2048, strategy="obfd")
        loader = DataLoader(dataset, batch_size=64, collate_fn=collate)

    Args:
        capacity (int): Maximum bin capacity in tokens.
        strategy (str): Packing algorithm short name (default: "obfd").
        seed (int | None): Random seed for shuffle-based algorithms.
        padding_value (int): Value for padding positions in ``input_ids`` (default: 0).
        label_padding_value (int): Value for masked label positions (default: -100).

    Returns:
        Callable[[list[list[int]]], PackedBatch]: Collate function for DataLoader.
    """

    def collate(batch: list[list[int]]) -> PackedBatch:
        """
        Pack a batch of token sequences into a PackedBatch.

        Args:
            batch (list[list[int]]): List of token ID sequences from the dataset.

        Returns:
            PackedBatch: Packed training batch with all metadata tensors.
        """
        lengths = [len(seq) for seq in batch]
        result = pack_sequences(
            lengths=lengths,
            capacity=capacity,
            strategy=strategy,
            seed=seed,
        )
        return pack_result_to_tensors(
            result=result,
            token_ids=batch,
            padding_value=padding_value,
            label_padding_value=label_padding_value,
        )

    return collate

"""
HuggingFace datasets utilities for sequence packing in LLM training.

Converts seqpacker packing results directly into a HuggingFace
datasets.Dataset, skipping the torch tensor roundtrip. Ideal for
workflows using HF Trainer, SFTTrainer, or TRL.

This is a lazy-import module. Users opt in with::

    from seqpacker.hf_utils import pack_dataset

Requires `datasets` to be installed (not a seqpacker dependency).
"""

from __future__ import annotations

from collections.abc import Sequence

from datasets import Dataset

from seqpacker import PackResult, pack_sequences

# -- Helper functions (private) ------------------------------------------------


def _list_position_ids(lengths: list[int], max_len: int) -> list[int]:
    """
    Compute per-sequence reset position IDs, padded to `max_len`.

    Each sequence gets positions starting from 0. Padding positions
    are filled with 0.

    Args:
        lengths (list[int]): Sequence lengths within a single pack.
        max_len (int): Total length to pad to (pack capacity).

    Returns:
        list[int]: Position IDs of length `max_len`.
    """
    ids: list[int] = []
    for length in lengths:
        ids.extend(range(length))
    padding_needed = max_len - len(ids)
    if padding_needed > 0:
        ids.extend([0] * padding_needed)
    return ids


def _list_labels(
    input_ids: list[int],
    lengths: list[int],
    *,
    label_padding_value: int = -100,
) -> list[int]:
    """
    Compute shifted labels with masking at sequence boundaries and padding.

    For each sequence within a pack, the label is the next token (shifted
    left by one). The last token of each sequence gets `label_padding_value`
    since there is no next token. Padding positions also get
    `label_padding_value`.

    Args:
        input_ids (list[int]): Padded input token IDs of length `max_len`.
        lengths (list[int]): Sequence lengths within this pack.
        label_padding_value (int): Value for masked label positions (default: -100).

    Returns:
        list[int]: Shifted labels of length `max_len`.
    """
    max_len = len(input_ids)
    labels = [label_padding_value] * max_len

    offset = 0
    for length in lengths:
        if length > 1:
            for i in range(length - 1):
                labels[offset + i] = input_ids[offset + i + 1]
        offset += length

    return labels


# -- Public API ----------------------------------------------------------------


def pack_dataset_from_result(
    result: PackResult,
    token_ids: Sequence[Sequence[int]],
    *,
    padding_value: int = 0,
    label_padding_value: int = -100,
) -> Dataset:
    """
    Convert a packing result and token sequences into a HuggingFace Dataset.

    For each pack, looks up token IDs by `sequence_ids`, concatenates
    them, pads to capacity, and generates all LLM training metadata columns
    (`position_ids`, `labels`, `attention_mask`).

    Args:
        result (PackResult): Packing result from `pack_sequences` or `Packer.pack`.
        token_ids (Sequence[Sequence[int]]): Token ID sequences indexed by original
                                             sequence ID. `token_ids[i]` corresponds
                                             to sequence `i`.
        padding_value (int): Value for padding positions in `input_ids` (default: 0).
        label_padding_value (int): Value for masked label positions (default: -100).

    Returns:
        datasets.Dataset: Dataset with columns `input_ids`, `attention_mask`,
                          `labels`, and `position_ids`. Each row is one packed bin.
    """
    packs = result.packs
    num_packs = len(packs)

    if num_packs == 0:
        return Dataset.from_dict(
            {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
                "position_ids": [],
            }
        )

    # Infer capacity (all packs share the same capacity)
    metrics = result.metrics
    capacity = (metrics.total_tokens + metrics.padding_tokens) // num_packs

    all_input_ids: list[list[int]] = []
    all_attention_mask: list[list[int]] = []
    all_labels: list[list[int]] = []
    all_position_ids: list[list[int]] = []

    for pack in packs:
        seq_ids = pack.sequence_ids
        lengths = pack.lengths

        # Concatenate token IDs for this pack
        tokens: list[int] = []
        for sid in seq_ids:
            tokens.extend(token_ids[sid])
        used = len(tokens)

        # Pad input_ids
        pad_len = capacity - used
        padded_ids = tokens + [padding_value] * pad_len

        # Attention mask: 1 for real tokens, 0 for padding
        mask = [1] * used + [0] * pad_len

        # Position IDs with per-sequence reset
        pos_ids = _list_position_ids(lengths, max_len=capacity)

        # Shifted labels with boundary masking
        labels = _list_labels(
            padded_ids,
            lengths,
            label_padding_value=label_padding_value,
        )

        all_input_ids.append(padded_ids)
        all_attention_mask.append(mask)
        all_labels.append(labels)
        all_position_ids.append(pos_ids)

    return Dataset.from_dict(
        {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
            "position_ids": all_position_ids,
        }
    )


def pack_dataset(
    token_ids: Sequence[Sequence[int]],
    capacity: int,
    *,
    strategy: str = "obfd",
    seed: int | None = None,
    padding_value: int = 0,
    label_padding_value: int = -100,
) -> Dataset:
    """
    Pack token sequences and return a HuggingFace Dataset ready for training.

    One-call convenience that packs sequences and builds a Dataset with
    correctly computed `input_ids`, `attention_mask`, `labels`
    (shifted with boundary masking), and `position_ids` (per-sequence reset).

    Equivalent to::

        result = pack_sequences(lengths, capacity, strategy, seed)
        ds = pack_dataset_from_result(result, token_ids, ...)

    Example::

        from seqpacker.hf_utils import pack_dataset

        tokenized = tokenizer(texts, truncation=True, max_length=2048)
        ds = pack_dataset(tokenized["input_ids"], capacity=2048)
        trainer = SFTTrainer(model=model, train_dataset=ds, ...)

    Args:
        token_ids (Sequence[Sequence[int]]): Token ID sequences to pack.
        capacity (int): Maximum bin capacity in tokens.
        strategy (str): Packing algorithm short name (default: "obfd").
        seed (int | None): Random seed for shuffle-based algorithms.
        padding_value (int): Value for padding positions in `input_ids` (default: 0).
        label_padding_value (int): Value for masked label positions (default: -100).

    Returns:
        datasets.Dataset: Dataset with columns `input_ids`, `attention_mask`,
                          `labels`, and `position_ids`. Each row is one packed bin.
    """
    lengths = [len(seq) for seq in token_ids]
    if not lengths:
        return Dataset.from_dict(
            {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
                "position_ids": [],
            }
        )
    result = pack_sequences(
        lengths=lengths,
        capacity=capacity,
        strategy=strategy,
        seed=seed,
    )
    return pack_dataset_from_result(
        result=result,
        token_ids=token_ids,
        padding_value=padding_value,
        label_padding_value=label_padding_value,
    )

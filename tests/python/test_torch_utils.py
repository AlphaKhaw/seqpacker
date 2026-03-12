"""
Tests for seqpacker.torch_utils.

Requires torch to be installed. Skipped automatically in CI
(torch is not in the test dependency group).
"""

import pytest
from seqpacker import pack_sequences
from seqpacker.torch_utils import (
    PackedBatch,
    _compute_cu_seqlens,
    _compute_labels,
    _compute_position_ids,
    pack_result_to_tensors,
    packed_collate_fn,
)

torch = pytest.importorskip("torch")
# ---------------------------------------------------------------------------
# Helper data
# ---------------------------------------------------------------------------

# 4 sequences of varying length
TOKEN_IDS = [
    [10, 20, 30],  # seq 0, len=3
    [40, 50],  # seq 1, len=2
    [60, 70, 80, 90],  # seq 2, len=4
    [100],  # seq 3, len=1
]
LENGTHS = [len(t) for t in TOKEN_IDS]
CAPACITY = 10


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class TestComputeCuSeqlens:
    """
    Test _compute_cu_seqlens helper.
    """

    def test_basic(self):
        """
        Cumulative lengths for [3, 2, 4] should be [0, 3, 5, 9].
        """
        result = _compute_cu_seqlens([3, 2, 4])
        assert result.tolist() == [0, 3, 5, 9]
        assert result.dtype == torch.int32

    def test_single_sequence(self):
        """
        Single sequence of length 5.
        """
        result = _compute_cu_seqlens([5])
        assert result.tolist() == [0, 5]

    def test_empty(self):
        """
        Empty list should return [0].
        """
        result = _compute_cu_seqlens([])
        assert result.tolist() == [0]


class TestComputePositionIds:
    """
    Test _compute_position_ids helper.
    """

    def test_two_sequences(self):
        """
        Position IDs reset at each sequence boundary.
        """
        result = _compute_position_ids([3, 2], max_len=7)
        assert result.tolist() == [0, 1, 2, 0, 1, 0, 0]

    def test_padding_is_zero(self):
        """
        Padding positions should be filled with 0.
        """
        result = _compute_position_ids([2], max_len=5)
        assert result.tolist() == [0, 1, 0, 0, 0]

    def test_exact_fit(self):
        """
        No padding needed when sequences fill the capacity.
        """
        result = _compute_position_ids([3, 2], max_len=5)
        assert result.tolist() == [0, 1, 2, 0, 1]


class TestComputeLabels:
    """
    Test _compute_labels helper.
    """

    def test_shifted_labels(self):
        """
        Labels are shifted left by one within each sequence,
        with -100 at the last token of each sequence and padding.
        """
        input_ids = torch.tensor([10, 20, 30, 40, 50, 0, 0], dtype=torch.long)
        labels = _compute_labels(input_ids, lengths=[3, 2])
        # seq 0: [20, 30, -100], seq 1: [50, -100], padding: [-100, -100]
        assert labels.tolist() == [20, 30, -100, 50, -100, -100, -100]

    def test_single_token_sequence(self):
        """
        A single-token sequence has no shifted label (all -100).
        """
        input_ids = torch.tensor([99, 0, 0], dtype=torch.long)
        labels = _compute_labels(input_ids, lengths=[1])
        assert labels[0].item() == -100

    def test_custom_padding_value(self):
        """
        Custom label padding value should be used.
        """
        input_ids = torch.tensor([10, 20, 0], dtype=torch.long)
        labels = _compute_labels(input_ids, lengths=[2], label_padding_value=-1)
        assert labels.tolist() == [20, -1, -1]


# ---------------------------------------------------------------------------
# pack_result_to_tensors
# ---------------------------------------------------------------------------


class TestPackResultToTensors:
    """
    Test pack_result_to_tensors end-to-end.
    """

    def test_basic_shapes(self):
        """
        Output tensors should have correct shapes.
        """
        result = pack_sequences(
            lengths=LENGTHS,
            capacity=CAPACITY,
        )
        batch = pack_result_to_tensors(
            result=result,
            token_ids=TOKEN_IDS,
        )

        assert isinstance(batch, PackedBatch)
        num_packs = result.num_bins
        assert batch.input_ids.shape == (num_packs, CAPACITY)
        assert batch.position_ids.shape == (num_packs, CAPACITY)
        assert batch.labels is not None
        assert batch.attention_mask is not None
        assert batch.labels.shape == (num_packs, CAPACITY)
        assert batch.attention_mask.shape == (num_packs, CAPACITY)
        assert len(batch.cu_seqlens) == num_packs

    def test_input_ids_correctness(self):
        """
        Token IDs should be correctly placed in the output tensor.
        """
        result = pack_sequences(
            lengths=LENGTHS,
            capacity=CAPACITY,
        )
        batch = pack_result_to_tensors(
            result=result,
            token_ids=TOKEN_IDS,
        )

        # Every original token should appear somewhere in input_ids
        all_tokens = set()
        for i, pack in enumerate(result.packs):
            used = pack.used
            row_tokens = batch.input_ids[i, :used].tolist()
            all_tokens.update(row_tokens)

        original_tokens = {t for seq in TOKEN_IDS for t in seq}
        assert original_tokens == all_tokens

    def test_attention_mask(self):
        """
        Attention mask should be 1 for real tokens, 0 for padding.
        """
        result = pack_sequences(
            lengths=LENGTHS,
            capacity=CAPACITY,
        )
        batch = pack_result_to_tensors(
            result=result,
            token_ids=TOKEN_IDS,
        )

        assert batch.attention_mask is not None
        for i, pack in enumerate(result.packs):
            used = pack.used
            # Real tokens
            assert batch.attention_mask[i, :used].sum().item() == used
            # Padding
            assert batch.attention_mask[i, used:].sum().item() == 0

    def test_cu_seqlens_per_pack(self):
        """
        cu_seqlens should start at 0 and end at total used tokens.
        """
        result = pack_sequences(
            lengths=LENGTHS,
            capacity=CAPACITY,
        )
        batch = pack_result_to_tensors(
            result=result,
            token_ids=TOKEN_IDS,
        )

        for i, pack in enumerate(result.packs):
            cu = batch.cu_seqlens[i]
            assert cu[0].item() == 0
            assert cu[-1].item() == pack.used
            assert cu.dtype == torch.int32

    def test_max_seqlen(self):
        """
        max_seqlen should be the longest individual sequence.
        """
        result = pack_sequences(
            lengths=LENGTHS,
            capacity=CAPACITY,
        )
        batch = pack_result_to_tensors(
            result=result,
            token_ids=TOKEN_IDS,
        )
        assert batch.max_seqlen == max(LENGTHS)

    def test_labels_masking(self):
        """
        Labels should be -100 at padding positions and at last token of each sequence.
        """
        result = pack_sequences(
            lengths=LENGTHS,
            capacity=CAPACITY,
        )
        batch = pack_result_to_tensors(
            result=result,
            token_ids=TOKEN_IDS,
        )

        assert batch.labels is not None
        for i, pack in enumerate(result.packs):
            used = pack.used
            # All padding positions should be -100
            padding_labels = batch.labels[i, used:]
            assert (padding_labels == -100).all()

    def test_custom_padding_value(self):
        """
        Custom padding_value should fill padding positions in input_ids.
        """
        result = pack_sequences(
            lengths=LENGTHS,
            capacity=CAPACITY,
        )
        batch = pack_result_to_tensors(
            result=result,
            token_ids=TOKEN_IDS,
            padding_value=999,
        )

        for i, pack in enumerate(result.packs):
            used = pack.used
            padding = batch.input_ids[i, used:]
            if padding.numel() > 0:
                assert (padding == 999).all()


# ---------------------------------------------------------------------------
# packed_collate_fn
# ---------------------------------------------------------------------------


class TestPackedCollateFn:
    """
    Test packed_collate_fn factory.
    """

    def test_returns_callable(self):
        """
        Factory should return a callable.
        """
        collate = packed_collate_fn(capacity=CAPACITY)
        assert callable(collate)

    def test_collate_basic(self):
        """
        Collate function should produce a valid PackedBatch.
        """
        collate = packed_collate_fn(capacity=CAPACITY)
        batch = collate(TOKEN_IDS)

        assert isinstance(batch, PackedBatch)
        assert batch.input_ids.ndim == 2
        assert batch.input_ids.shape[1] == CAPACITY

    def test_collate_with_strategy(self):
        """
        Different strategies should all produce valid output.
        """
        for strategy in ["ffd", "obfd", "nf"]:
            collate = packed_collate_fn(capacity=CAPACITY, strategy=strategy)
            batch = collate(TOKEN_IDS)
            assert isinstance(batch, PackedBatch)

    def test_collate_deterministic_with_seed(self):
        """
        Same seed should produce identical results.
        """
        collate = packed_collate_fn(capacity=CAPACITY, strategy="ffs", seed=42)
        batch1 = collate(TOKEN_IDS)
        batch2 = collate(TOKEN_IDS)
        assert torch.equal(batch1.input_ids, batch2.input_ids)

# pyright: reportCallIssue=false, reportArgumentType=false
"""
Tests for seqpacker.hf_utils.

Requires datasets to be installed. Skipped automatically when
datasets is not available.
"""

import pytest

datasets = pytest.importorskip("datasets")

from seqpacker import pack_sequences  # noqa: E402
from seqpacker.hf_utils import (  # noqa: E402
    _list_labels,
    _list_position_ids,
    pack_dataset,
    pack_dataset_from_result,
)

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


class TestListPositionIds:
    """
    Test _list_position_ids helper.
    """

    def test_two_sequences(self):
        """
        Position IDs reset at each sequence boundary.
        """
        result = _list_position_ids([3, 2], max_len=7)
        assert result == [0, 1, 2, 0, 1, 0, 0]

    def test_padding_is_zero(self):
        """
        Padding positions should be filled with 0.
        """
        result = _list_position_ids([2], max_len=5)
        assert result == [0, 1, 0, 0, 0]

    def test_exact_fit(self):
        """
        No padding needed when sequences fill the capacity.
        """
        result = _list_position_ids([3, 2], max_len=5)
        assert result == [0, 1, 2, 0, 1]

    def test_empty(self):
        """
        Empty lengths should produce all-zero padding.
        """
        result = _list_position_ids([], max_len=3)
        assert result == [0, 0, 0]

    def test_single_token_sequence(self):
        """
        A single-token sequence should produce [0] + padding.
        """
        result = _list_position_ids([1], max_len=4)
        assert result == [0, 0, 0, 0]


class TestListLabels:
    """
    Test _list_labels helper.
    """

    def test_shifted_labels(self):
        """
        Labels are shifted left by one within each sequence,
        with -100 at the last token of each sequence and padding.
        """
        input_ids = [10, 20, 30, 40, 50, 0, 0]
        labels = _list_labels(input_ids, lengths=[3, 2])
        # seq 0: [20, 30, -100], seq 1: [50, -100], padding: [-100, -100]
        assert labels == [20, 30, -100, 50, -100, -100, -100]

    def test_single_token_sequence(self):
        """
        A single-token sequence has no shifted label (all -100).
        """
        input_ids = [99, 0, 0]
        labels = _list_labels(input_ids, lengths=[1])
        assert labels[0] == -100

    def test_custom_padding_value(self):
        """
        Custom label padding value should be used.
        """
        input_ids = [10, 20, 0]
        labels = _list_labels(input_ids, lengths=[2], label_padding_value=-1)
        assert labels == [20, -1, -1]

    def test_multiple_sequences(self):
        """
        Labels are correctly shifted across multiple sequences with no
        cross-contamination.
        """
        input_ids = [10, 20, 30, 40, 50, 0]
        labels = _list_labels(input_ids, lengths=[3, 2])
        # seq 0: label[0]=20, label[1]=30, label[2]=-100
        # seq 1: label[3]=50, label[4]=-100
        # padding: label[5]=-100
        assert labels == [20, 30, -100, 50, -100, -100]

    def test_all_single_token(self):
        """
        All single-token sequences should produce all -100 labels.
        """
        input_ids = [10, 20, 30, 0]
        labels = _list_labels(input_ids, lengths=[1, 1, 1])
        assert labels == [-100, -100, -100, -100]


# ---------------------------------------------------------------------------
# pack_dataset_from_result
# ---------------------------------------------------------------------------


class TestPackDatasetFromResult:
    """
    Test pack_dataset_from_result end-to-end.
    """

    def test_basic_columns(self):
        """
        Output Dataset should have the expected columns.
        """
        result = pack_sequences(lengths=LENGTHS, capacity=CAPACITY)
        ds = pack_dataset_from_result(result=result, token_ids=TOKEN_IDS)

        assert isinstance(ds, datasets.Dataset)
        assert set(ds.column_names) == {
            "input_ids",
            "attention_mask",
            "labels",
            "position_ids",
        }

    def test_row_count(self):
        """
        Number of rows should equal number of packs.
        """
        result = pack_sequences(lengths=LENGTHS, capacity=CAPACITY)
        ds = pack_dataset_from_result(result=result, token_ids=TOKEN_IDS)
        assert len(ds) == result.num_bins

    def test_row_length(self):
        """
        Each row should have length equal to capacity.
        """
        result = pack_sequences(lengths=LENGTHS, capacity=CAPACITY)
        ds = pack_dataset_from_result(result=result, token_ids=TOKEN_IDS)

        for row in ds:
            assert len(row["input_ids"]) == CAPACITY
            assert len(row["attention_mask"]) == CAPACITY
            assert len(row["labels"]) == CAPACITY
            assert len(row["position_ids"]) == CAPACITY

    def test_input_ids_correctness(self):
        """
        All original tokens should appear in the output.
        """
        result = pack_sequences(lengths=LENGTHS, capacity=CAPACITY)
        ds = pack_dataset_from_result(result=result, token_ids=TOKEN_IDS)

        all_tokens = set()
        for i, pack in enumerate(result.packs):
            used = pack.used
            row_tokens = ds[i]["input_ids"][:used]
            all_tokens.update(row_tokens)

        original_tokens = {t for seq in TOKEN_IDS for t in seq}
        assert original_tokens == all_tokens

    def test_attention_mask(self):
        """
        Attention mask should be 1 for real tokens, 0 for padding.
        """
        result = pack_sequences(lengths=LENGTHS, capacity=CAPACITY)
        ds = pack_dataset_from_result(result=result, token_ids=TOKEN_IDS)

        for i, pack in enumerate(result.packs):
            used = pack.used
            mask = ds[i]["attention_mask"]
            assert sum(mask[:used]) == used
            assert sum(mask[used:]) == 0

    def test_labels_masking(self):
        """
        Labels should be -100 at padding positions and at the last token of
        each sequence.
        """
        result = pack_sequences(lengths=LENGTHS, capacity=CAPACITY)
        ds = pack_dataset_from_result(result=result, token_ids=TOKEN_IDS)

        for i, pack in enumerate(result.packs):
            used = pack.used
            labels = ds[i]["labels"]
            # All padding positions should be -100
            for lbl in labels[used:]:
                assert lbl == -100

    def test_custom_padding_value(self):
        """
        Custom padding_value should fill padding positions in input_ids.
        """
        result = pack_sequences(lengths=LENGTHS, capacity=CAPACITY)
        ds = pack_dataset_from_result(
            result=result,
            token_ids=TOKEN_IDS,
            padding_value=999,
        )

        for i, pack in enumerate(result.packs):
            used = pack.used
            padding = ds[i]["input_ids"][used:]
            if padding:
                assert all(v == 999 for v in padding)

    def test_position_ids_reset(self):
        """
        Position IDs should reset at each sequence boundary within a pack.
        """
        result = pack_sequences(lengths=LENGTHS, capacity=CAPACITY)
        ds = pack_dataset_from_result(result=result, token_ids=TOKEN_IDS)

        for i, pack in enumerate(result.packs):
            pos = ds[i]["position_ids"]
            offset = 0
            for length in pack.lengths:
                expected = list(range(length))
                assert pos[offset : offset + length] == expected
                offset += length

    def test_empty_input(self):
        """
        Empty token_ids should return an empty Dataset.
        """
        ds = pack_dataset(token_ids=[], capacity=10)
        assert len(ds) == 0
        assert set(ds.column_names) == {
            "input_ids",
            "attention_mask",
            "labels",
            "position_ids",
        }


# ---------------------------------------------------------------------------
# pack_dataset
# ---------------------------------------------------------------------------


class TestPackDataset:
    """
    Test pack_dataset convenience function.
    """

    def test_basic(self):
        """
        Basic pack_dataset call should produce a valid Dataset.
        """
        ds = pack_dataset(token_ids=TOKEN_IDS, capacity=CAPACITY)

        assert isinstance(ds, datasets.Dataset)
        assert set(ds.column_names) == {
            "input_ids",
            "attention_mask",
            "labels",
            "position_ids",
        }
        assert len(ds) > 0

    def test_different_strategies(self):
        """
        Different strategies should all produce valid output.
        """
        for strategy in ["ffd", "obfd", "nf", "bf"]:
            ds = pack_dataset(
                token_ids=TOKEN_IDS,
                capacity=CAPACITY,
                strategy=strategy,
            )
            assert isinstance(ds, datasets.Dataset)
            assert len(ds) > 0

    def test_deterministic_with_seed(self):
        """
        Same seed should produce identical results.
        """
        ds1 = pack_dataset(
            token_ids=TOKEN_IDS,
            capacity=CAPACITY,
            strategy="ffs",
            seed=42,
        )
        ds2 = pack_dataset(
            token_ids=TOKEN_IDS,
            capacity=CAPACITY,
            strategy="ffs",
            seed=42,
        )
        for i in range(len(ds1)):
            assert ds1[i]["input_ids"] == ds2[i]["input_ids"]

    def test_all_tokens_preserved(self):
        """
        All original tokens should appear in the packed dataset.
        """
        ds = pack_dataset(token_ids=TOKEN_IDS, capacity=CAPACITY)

        all_tokens = set()
        for row in ds:
            mask = row["attention_mask"]
            ids = row["input_ids"]
            for j, m in enumerate(mask):
                if m == 1:
                    all_tokens.add(ids[j])

        original_tokens = {t for seq in TOKEN_IDS for t in seq}
        assert original_tokens == all_tokens

    def test_labels_no_cross_contamination(self):
        """
        Labels at sequence boundaries should be -100, not the next sequence's first
        token.
        """
        ds = pack_dataset(token_ids=TOKEN_IDS, capacity=CAPACITY)

        result = pack_sequences(
            lengths=LENGTHS,
            capacity=CAPACITY,
        )

        for i, pack in enumerate(result.packs):
            labels = ds[i]["labels"]
            offset = 0
            for length in pack.lengths:
                # Last token of each sequence should have -100 label
                assert labels[offset + length - 1] == -100
                offset += length

    def test_custom_padding_values(self):
        """
        Custom padding and label padding values should be respected.
        """
        ds = pack_dataset(
            token_ids=TOKEN_IDS,
            capacity=CAPACITY,
            padding_value=999,
            label_padding_value=-1,
        )

        result = pack_sequences(lengths=LENGTHS, capacity=CAPACITY)
        for i, pack in enumerate(result.packs):
            used = pack.used
            row = ds[i]
            # Padding in input_ids
            for v in row["input_ids"][used:]:
                assert v == 999
            # Padding in labels
            for v in row["labels"][used:]:
                assert v == -1

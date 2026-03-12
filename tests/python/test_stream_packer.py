"""
Tests for StreamPacker Python bindings.
"""

import pytest
from seqpacker import Pack, StreamPacker

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestStreamPackerConstruction:
    """
    Test StreamPacker construction and argument handling.
    """

    def test_default_strategy(self):
        """
        Default strategy should be nf (NextFit).
        """
        sp = StreamPacker(capacity=1024)
        assert "NextFit" in repr(sp)

    def test_nf_strategy(self):
        """
        Construct with nf strategy.
        """
        sp = StreamPacker(capacity=1024, strategy="nf")
        assert sp is not None

    def test_hk_strategy(self):
        """
        Construct with hk strategy.
        """
        sp = StreamPacker(capacity=1024, strategy="hk")
        assert "Harmonic" in repr(sp)

    def test_hk_with_k(self):
        """
        Construct with hk strategy and custom k.
        """
        sp = StreamPacker(capacity=1024, strategy="hk", k=5)
        assert sp is not None

    def test_nextfit_alias(self):
        """
        Construct with nextfit alias.
        """
        sp = StreamPacker(capacity=1024, strategy="nextfit")
        assert "NextFit" in repr(sp)

    def test_harmonic_alias(self):
        """
        Construct with harmonic alias.
        """
        sp = StreamPacker(capacity=1024, strategy="harmonic")
        assert "Harmonic" in repr(sp)

    def test_invalid_strategy(self):
        """
        Unknown strategy should raise ValueError with helpful message.
        """
        with pytest.raises(ValueError, match="unknown streaming strategy"):
            StreamPacker(capacity=1024, strategy="ffd")

    def test_invalid_strategy_suggests_packer(self):
        """
        Error message should suggest using Packer.pack() instead.
        """
        with pytest.raises(ValueError, match="Packer.pack"):
            StreamPacker(capacity=1024, strategy="bf")


# ---------------------------------------------------------------------------
# NextFit streaming
# ---------------------------------------------------------------------------


class TestNextFitStreaming:
    """
    Test NextFit streaming behaviour.
    """

    def test_single_item_no_emit(self):
        """
        Adding a single item that fits should not emit any packs.
        """
        sp = StreamPacker(capacity=100)
        packs = sp.add(50)
        assert packs == []

    def test_overflow_emits_pack(self):
        """
        Adding an item that doesn't fit in current bin emits the old bin.
        """
        sp = StreamPacker(capacity=100)
        sp.add(60)
        packs = sp.add(60)
        assert len(packs) == 1
        assert isinstance(packs[0], Pack)
        assert packs[0].used == 60

    def test_finish_flushes_remaining(self):
        """
        Finish should emit all remaining open bins.
        """
        sp = StreamPacker(capacity=100)
        sp.add(30)
        sp.add(40)
        packs = sp.finish()
        assert len(packs) == 1
        assert packs[0].used == 70

    def test_finish_empty_returns_nothing(self):
        """
        Finish with no items added should return empty list.
        """
        sp = StreamPacker(capacity=100)
        packs = sp.finish()
        assert packs == []

    def test_exact_fill_no_emit(self):
        """
        Filling a bin exactly should not emit (bin stays open).
        """
        sp = StreamPacker(capacity=100)
        packs = sp.add(100)
        assert packs == []
        # Next item triggers the full bin to close
        packs = sp.add(1)
        assert len(packs) == 1
        assert packs[0].used == 100

    def test_all_sequences_accounted(self):
        """
        All added sequence IDs appear in emitted packs.
        """
        sp = StreamPacker(capacity=100)
        all_packs = []
        for length in [30, 40, 50, 60, 70]:
            all_packs.extend(sp.add(length))
        all_packs.extend(sp.finish())

        all_ids = sorted(id_ for pack in all_packs for id_ in pack.sequence_ids)
        assert all_ids == [0, 1, 2, 3, 4]

    def test_no_pack_exceeds_capacity(self):
        """
        No emitted pack should exceed the capacity.
        """
        sp = StreamPacker(capacity=100)
        all_packs = []
        for length in [30, 40, 50, 60, 70, 80, 10, 20]:
            all_packs.extend(sp.add(length))
        all_packs.extend(sp.finish())

        for pack in all_packs:
            assert pack.used <= 100

    def test_sequences_added_counter(self):
        """
        Verify sequences_added tracks the count.
        """
        sp = StreamPacker(capacity=100)
        assert sp.sequences_added == 0
        sp.add(10)
        assert sp.sequences_added == 1
        sp.add(20)
        assert sp.sequences_added == 2


# ---------------------------------------------------------------------------
# Harmonic-K streaming
# ---------------------------------------------------------------------------


class TestHarmonicStreaming:
    """
    Test Harmonic-K streaming behaviour.
    """

    def test_large_item_emits_immediately(self):
        """
        Items > capacity/2 should emit a single-item pack immediately.
        """
        sp = StreamPacker(capacity=100, strategy="hk")
        packs = sp.add(60)
        assert len(packs) == 1
        assert packs[0].used == 60

    def test_medium_items_pair_then_emit(self):
        """
        Two items in the same class should pair then emit when the bin fills.
        """
        sp = StreamPacker(capacity=100, strategy="hk")
        packs = sp.add(40)
        assert packs == []
        packs = sp.add(40)
        assert len(packs) == 1
        assert packs[0].used == 80

    def test_different_classes_separate(self):
        """
        Items in different size classes go to different bins.
        """
        sp = StreamPacker(capacity=100, strategy="hk")
        # 40 → class for items in (capacity/3, capacity/2]
        sp.add(40)
        # 20 → different class
        sp.add(20)
        # Should have two open bins, nothing emitted yet
        packs = sp.finish()
        assert len(packs) == 2

    def test_all_sequences_accounted(self):
        """
        All sequence IDs appear in emitted packs.
        """
        sp = StreamPacker(capacity=100, strategy="hk")
        all_packs = []
        for length in [60, 40, 40, 20, 20, 20, 20, 20, 10]:
            all_packs.extend(sp.add(length))
        all_packs.extend(sp.finish())

        all_ids = sorted(id_ for pack in all_packs for id_ in pack.sequence_ids)
        assert all_ids == list(range(9))

    def test_no_pack_exceeds_capacity(self):
        """
        No emitted pack should exceed the capacity.
        """
        sp = StreamPacker(capacity=100, strategy="hk")
        all_packs = []
        for length in [60, 40, 40, 20, 20, 20, 20, 20, 10, 5, 3]:
            all_packs.extend(sp.add(length))
        all_packs.extend(sp.finish())

        for pack in all_packs:
            assert pack.used <= 100


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestStreamPackerErrors:
    """
    Test error conditions.
    """

    def test_oversize_raises(self):
        """
        Adding a length exceeding capacity should raise ValueError.
        """
        sp = StreamPacker(capacity=100)
        with pytest.raises(ValueError, match="exceeds capacity"):
            sp.add(101)

    def test_add_after_finish_raises(self):
        """
        Adding after finish should raise ValueError.
        """
        sp = StreamPacker(capacity=100)
        sp.finish()
        with pytest.raises(ValueError, match="already finished"):
            sp.add(10)

    def test_finish_after_finish_raises(self):
        """
        Calling finish twice should raise ValueError.
        """
        sp = StreamPacker(capacity=100)
        sp.finish()
        with pytest.raises(ValueError, match="already finished"):
            sp.finish()

    def test_sequences_added_after_finish_raises(self):
        """
        Accessing sequences_added after finish should raise ValueError.
        """
        sp = StreamPacker(capacity=100)
        sp.finish()
        with pytest.raises(ValueError, match="already finished"):
            _ = sp.sequences_added


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestStreamPackerRepr:
    """
    Test __repr__ output.
    """

    def test_repr_active(self):
        """
        Active packer shows strategy and count.
        """
        sp = StreamPacker(capacity=100, strategy="nf")
        sp.add(10)
        r = repr(sp)
        assert "NextFit" in r
        assert "added=1" in r

    def test_repr_finished(self):
        """
        Finished packer shows finished state.
        """
        sp = StreamPacker(capacity=100)
        sp.finish()
        assert "finished" in repr(sp)


# ---------------------------------------------------------------------------
# Pack properties
# ---------------------------------------------------------------------------


class TestStreamPackProperties:
    """
    Test that packs emitted by StreamPacker have correct Pack properties.
    """

    def test_pack_has_sequence_ids(self):
        """
        Emitted packs should have sequence_ids.
        """
        sp = StreamPacker(capacity=100)
        sp.add(60)
        packs = sp.add(60)
        assert len(packs) == 1
        assert isinstance(packs[0].sequence_ids, list)
        assert len(packs[0].sequence_ids) > 0

    def test_pack_has_lengths(self):
        """
        Emitted packs should have lengths.
        """
        sp = StreamPacker(capacity=100)
        sp.add(60)
        packs = sp.add(60)
        assert packs[0].lengths == [60]

    def test_pack_used_matches_sum(self):
        """
        Pack.used should equal sum of lengths.
        """
        sp = StreamPacker(capacity=100)
        sp.add(30)
        sp.add(40)
        packs = sp.finish()
        assert packs[0].used == 70
        assert sum(packs[0].lengths) == 70

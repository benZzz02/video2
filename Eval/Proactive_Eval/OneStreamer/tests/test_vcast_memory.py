"""CPU checks for opt-in, causally bounded V-CAST historical token selection."""

from dataclasses import replace
from pathlib import Path
import random
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onestreamer.visual_memory import MemoryConfig, VisualFrame
from onestreamer.vcast_memory import VCastVisualMemoryBank, allocate_tokens, temporal_weights


def native_frame(timestamp, height=4, width=4, channels=3, layers=2):
    generator = torch.Generator().manual_seed(int(timestamp) + 913)
    features = torch.randn(height * width, channels, generator=generator)
    deepstack = tuple(features + 100 * (index + 1) for index in range(layers))
    return VisualFrame(float(timestamp), (1, height * 2, width * 2), features, deepstack,
                       (width * 28, height * 28))


def bank(tokens=1000, **kwargs):
    options = dict(history_compression="vcast", capacity=1000, store_tokens=tokens,
                   store_bytes=1 << 30, min_frame_tokens=2, read_tokens=100, read_frames=4)
    options.update(kwargs)
    return VCastVisualMemoryBank(MemoryConfig(**options))


class AllocationTests(unittest.TestCase):
    def test_reference_curvature_endpoints_and_softmax(self):
        means = F.normalize(torch.tensor([[1., 0.], [0., 1.], [1., 0.]]), dim=-1)
        expected = torch.softmax(torch.tensor([1., 2., 1.]) / 0.7, dim=0)
        torch.testing.assert_close(temporal_weights(means), expected)
        self.assertGreater(temporal_weights(means)[1], temporal_weights(means)[0])
        torch.testing.assert_close(temporal_weights(means[:1]), torch.ones(1))
        torch.testing.assert_close(temporal_weights(means[:2]), torch.full((2,), 0.5))

    def test_weighted_integer_budget_and_saturation(self):
        self.assertEqual(allocate_tokens([1, 4, 1], [10, 10, 10], 12, [1, 1, 1]), [3, 7, 2])
        self.assertEqual(allocate_tokens([100, 1, 1], [2, 8, 8], 12, [1, 1, 1]), [2, 5, 5])
        self.assertEqual(allocate_tokens([0, 0], [8, 8], 6, [1, 1]), [3, 3])

    def test_budgets_never_inflated_to_meet_minimum(self):
        with self.assertRaises(ValueError):
            allocate_tokens([1, 1], [10, 10], 3, [2, 2])
        with self.assertRaises(ValueError):
            allocate_tokens([1, 1], [10, 10], 8, [2, 2], token_bytes=[8, 8], byte_budget=31)

    def test_variable_cost_allocations_respect_both_resources(self):
        rng = random.Random(7)
        for _ in range(100):
            count = rng.randrange(1, 10)
            capacities = [rng.randrange(1, 30) for _ in range(count)]
            costs = [rng.randrange(1, 20) for _ in range(count)]
            weights = [rng.random() for _ in range(count)]
            token_budget = rng.randrange(count, sum(capacities) + 1)
            byte_budget = rng.randrange(sum(costs), sum(c * n for c, n in zip(costs, capacities)) + 1)
            result = allocate_tokens(weights, capacities, token_budget, [1] * count,
                                     token_bytes=costs, byte_budget=byte_budget)
            self.assertLessEqual(sum(result), token_budget)
            self.assertLessEqual(sum(c * n for c, n in zip(costs, result)), byte_budget)
            self.assertTrue(all(1 <= n <= cap for n, cap in zip(result, capacities)))
            if sum(result) < token_budget:
                remaining = byte_budget - sum(c * n for c, n in zip(costs, result))
                self.assertTrue(all(n == cap or c > remaining for n, cap, c in zip(result, capacities, costs)))


class VCastMemoryTests(unittest.TestCase):
    def assert_caps(self, memory):
        frames = memory.candidates(1e10)
        stats = memory.stats()
        self.assertEqual(stats["stored_frames"], len(frames))
        self.assertEqual(stats["stored_tokens"], sum(frame.token_count for frame in frames))
        self.assertEqual(stats["stored_feature_bytes"], sum(frame.nbytes for frame in frames))
        self.assertEqual(stats["stored_bytes"], stats["stored_feature_bytes"] + stats["metadata_bytes"])
        self.assertLessEqual(stats["stored_bytes"], memory.config.store_bytes)
        self.assertLessEqual(stats["budget_cost"], memory.token_budget)
        self.assertLessEqual(len(frames), memory.config.capacity)
        self.assertEqual([f.timestamp for f in frames], sorted(set(f.timestamp for f in frames)))
        for frame in frames:
            if frame.source_indices is not None:
                self.assertEqual(tuple(sorted(set(frame.source_indices))), frame.source_indices)
                self.assertEqual(len(frame.source_indices), frame.token_count)

    def test_no_pressure_preserves_every_token_and_owns_cpu_storage(self):
        memory = bank()
        original = native_frame(0)
        memory.observe(original)
        stored = memory.candidates(1)[0]
        self.assertIsNone(stored.source_indices)
        self.assertEqual(stored.grid_thw, original.grid_thw)
        self.assertEqual(stored.size, original.size)
        for source, copy in zip((original.features, *original.deepstack), (stored.features, *stored.deepstack)):
            torch.testing.assert_close(source, copy)
            self.assertNotEqual(source.data_ptr(), copy.data_ptr())
            self.assertEqual(copy.device.type, "cpu")
        self.assertEqual(memory.stats()["selection_operations"], 0)
        self.assert_caps(memory)

    def test_selection_matches_original_dual_anchor_scores_and_all_branches(self):
        original = native_frame(0)
        memory = bank(tokens=5)
        memory.observe(original)
        stored = memory.candidates(1)[0]
        mean = F.normalize(original.features.mean(dim=0), dim=0, eps=1e-6)
        norms = original.features.norm(dim=-1)
        scores = 1 - F.cosine_similarity(original.features, mean[None], dim=-1, eps=1e-6)
        scores += (norms - norms.min()) / (norms.max() - norms.min() + 1e-6)
        expected = torch.argsort(scores, descending=True, stable=True)[:5].sort().values
        self.assertEqual(stored.source_indices, tuple(expected.tolist()))
        for source, selected in zip((original.features, *original.deepstack), (stored.features, *stored.deepstack)):
            torch.testing.assert_close(selected, source[expected])
        self.assertEqual(stored.grid_thw, original.grid_thw)
        self.assertEqual(stored.size, original.size)
        self.assertEqual(memory.stats()["skipped_oversize"], 0)
        self.assert_caps(memory)

    def test_scores_and_original_means_survive_repeated_pruning(self):
        memory = bank(tokens=12)
        original = native_frame(0)
        memory.observe(original)
        first = memory.candidates(1)[0]
        mean = memory._records[0].original_mean.clone()
        scores = dict(zip(first.source_indices, memory._records[0].scores.tolist()))
        memory.set_budget(7)
        second = memory.candidates(1)[0]
        self.assertTrue(set(second.source_indices).issubset(first.source_indices))
        torch.testing.assert_close(memory._records[0].original_mean, mean)
        self.assertEqual(memory._records[0].scores.tolist(), [scores[i] for i in second.source_indices])
        memory.set_budget(3)
        third = memory.candidates(1)[0]
        self.assertTrue(set(third.source_indices).issubset(second.source_indices))
        memory.set_budget(1000)
        self.assertEqual(memory.candidates(1)[0].source_indices, third.source_indices)
        self.assertEqual(memory._records[0].scores.numel(), 3)
        self.assert_caps(memory)

    def test_equal_scores_and_zero_vectors_are_finite_and_deterministic(self):
        frame = native_frame(0)
        frame = replace(frame, features=torch.zeros_like(frame.features))
        memory = bank(tokens=3)
        memory.observe(frame)
        self.assertEqual(memory.candidates(1)[0].source_indices, (0, 1, 2))
        self.assertTrue(torch.isfinite(memory._records[0].scores).all())

    def test_mixed_spatial_grids_keep_source_geometries(self):
        memory = bank()
        originals = [native_frame(0, 2, 4), native_frame(1, 3, 5)]
        for frame in originals:
            memory.observe(frame)
        memory.set_budget(12)
        for old, selected in zip(originals, memory.candidates(2)):
            self.assertEqual(selected.grid_thw, old.grid_thw)
            self.assertEqual(selected.size, old.size)
            torch.testing.assert_close(selected.features, old.features[list(selected.source_indices)])
        self.assertEqual(memory.stats()["stored_tokens"], 12)
        self.assert_caps(memory)

    def test_oversize_admission_shrinks_even_below_soft_floor(self):
        memory = bank(tokens=3, min_frame_tokens=16)
        memory.observe(native_frame(0))
        self.assertEqual(memory.candidates(1)[0].token_count, 3)
        self.assertEqual(memory.stats()["skipped_oversize"], 0)
        self.assert_caps(memory)

    def test_byte_cap_counts_deepstack_coordinates_scores_and_mean(self):
        # Two channels, three FP32 branches: 24/token + 8 coordinates + 4 score;
        # the original mean costs another 8 bytes. Exactly one token fits.
        memory = bank(tokens=100, store_bytes=44, min_frame_tokens=4)
        memory.observe(native_frame(0, channels=2))
        self.assertEqual(memory.candidates(1)[0].token_count, 1)
        self.assertEqual(memory.stats()["stored_bytes"], 44)
        self.assert_caps(memory)
        impossible = bank(tokens=100, store_bytes=43)
        impossible.observe(native_frame(0, channels=2))
        self.assertEqual(impossible.candidates(1), [])
        self.assertEqual(impossible.stats()["skipped_oversize"], 1)

    def test_long_stream_temporal_thinning_preserves_endpoints_and_bounds(self):
        memory = bank(tokens=20, capacity=5)
        memory.set_budget(20, frame_overhead=2)
        for timestamp in range(100):
            memory.observe(native_frame(timestamp), now=timestamp + 16)
            self.assert_caps(memory)
        frames = memory.candidates(100)
        self.assertEqual(frames[0].timestamp, 0)
        self.assertEqual(frames[-1].timestamp, 99)
        self.assertGreater(memory.stats()["dropped_frames"], 0)
        self.assertEqual(len(memory._records), 5)

    def test_capacity_drop_does_not_prune_dense_frame_that_now_fits(self):
        memory = bank(capacity=1, store_bytes=652)
        memory.observe(native_frame(0))
        memory.observe(native_frame(1))
        stored = memory.candidates(2)[0]
        self.assertEqual(stored.token_count, 16)
        self.assertIsNone(stored.source_indices)
        self.assertEqual(memory.stats()["selection_operations"], 0)
        self.assertEqual(memory.stats()["stored_bytes"], 652)
        self.assert_caps(memory)

    def test_dense_frames_at_floor_do_not_pay_unused_coordinate_bytes(self):
        # Each complete one-token frame costs 12 bytes (features, mean, score).
        # The third frame can shrink to one token plus coordinates (20 bytes),
        # so all three timestamps fit in 44 bytes. Dense frames never need the
        # eight coordinate bytes that the reducible third frame needs.
        memory = bank(store_bytes=52, min_frame_tokens=1)
        for timestamp, height, width in ((0, 1, 1), (1, 1, 1), (2, 2, 2)):
            memory.observe(native_frame(timestamp, height, width, channels=1, layers=0))
        frames = memory.candidates(3)
        self.assertEqual([frame.timestamp for frame in frames], [0, 1, 2])
        self.assertEqual([frame.token_count for frame in frames], [1, 1, 1])
        self.assertEqual(memory.stats()["stored_bytes"], 44)
        self.assertEqual(memory.stats()["dropped_frames"], 0)
        self.assert_caps(memory)

    def test_below_floor_single_record_still_reserves_sparse_coordinates(self):
        memory = bank(store_bytes=23, min_frame_tokens=4)
        memory.observe(native_frame(0, 2, 2, channels=1, layers=0))
        self.assertEqual(memory.candidates(1)[0].token_count, 1)
        self.assertEqual(memory.stats()["stored_bytes"], 20)
        self.assert_caps(memory)

    def test_randomized_mixed_grids_payloads_and_budget_changes(self):
        rng = random.Random(71)
        for _ in range(50):
            memory = bank(tokens=rng.randrange(1, 100), capacity=rng.randrange(1, 10),
                          store_bytes=rng.randrange(50, 8000), min_frame_tokens=rng.randrange(1, 20))
            for timestamp in range(20):
                frame = native_frame(timestamp, rng.randrange(1, 8), rng.randrange(1, 8),
                                     layers=rng.randrange(4))
                memory.observe(frame)
                self.assert_caps(memory)
                if timestamp % 4 == 0:
                    memory.set_budget(rng.randrange(100), rng.randrange(8))
                    self.assert_caps(memory)

    def test_causality_cutoff_and_invalid_admissions(self):
        memory = bank()
        memory.observe(native_frame(0), now=0)
        memory.observe(native_frame(1), now=2)
        self.assertEqual([frame.timestamp for frame in memory.candidates(1)], [0])
        for frame, now in ((native_frame(1), 2), (native_frame(2), 1), (native_frame(2), float("nan"))):
            with self.assertRaises(ValueError):
                memory.observe(frame, now=now)
        with self.assertRaises(ValueError):
            memory.candidates(float("inf"))
        with self.assertRaises(ValueError):
            memory.observe(replace(native_frame(2), source_indices=tuple(range(16))))
        with self.assertRaises(ValueError):
            memory.observe(replace(native_frame(2), pool_source_shape=(4, 4)))

    def test_disabled_and_zero_runtime_budget_keep_no_records(self):
        disabled = bank(capacity=0)
        disabled.observe(native_frame(0))
        self.assertEqual(disabled.stats()["seen"], 1)
        self.assertEqual(disabled.candidates(1), [])
        memory = bank()
        memory.observe(native_frame(0))
        memory.set_budget(0)
        memory.observe(native_frame(1))
        self.assertEqual(memory.candidates(2), [])
        self.assert_caps(memory)

    def test_fixed_reader_remains_budgeted_and_chronological(self):
        memory = bank(tokens=20, read_tokens=8, read_frames=3)
        for timestamp in range(5):
            memory.observe(native_frame(timestamp))
        selected = memory.read(5)
        self.assertLessEqual(len(selected), 3)
        self.assertLessEqual(sum(frame.token_count for frame in selected), 8)
        self.assertEqual([f.timestamp for f in selected], sorted(f.timestamp for f in selected))
        self.assertTrue(selected)


    def test_short_history_matches_budget_pool_span_and_fits_cap(self):
        from onestreamer.question_router import newest_history
        from onestreamer.visual_memory import BudgetVisualMemoryBank

        pooled = BudgetVisualMemoryBank(MemoryConfig(
            history_compression="budget_pool", capacity=1000, store_tokens=10_000,
            store_bytes=1 << 30, min_frame_tokens=2), merge_size=2)
        memory = bank(tokens=10_000)
        for timestamp in range(12):
            frame = native_frame(timestamp, height=8, width=8)
            pooled.observe(frame)
            memory.observe(frame)
        # 2x2 pooling makes 16 tokens per frame: a cap of 50 covers the newest 3.
        expected = [f.timestamp for f in newest_history(pooled.candidates(11), 50)]
        before = [(f.timestamp, f.token_count) for f in memory.candidates(12)]
        frames, meta = memory.short_history(11, 50)
        self.assertEqual([f.timestamp for f in frames], expected)
        self.assertEqual(expected, [8.0, 9.0, 10.0])
        self.assertLessEqual(sum(f.token_count for f in frames), 50)
        self.assertEqual(meta["short_retained_tokens"], 3 * 64)
        self.assertEqual(meta["short_read_tokens"], sum(f.token_count for f in frames))
        for frame in frames:
            stored = next(f for f in memory.candidates(12) if f.timestamp == frame.timestamp)
            torch.testing.assert_close(frame.features, stored.features[list(frame.source_indices)])
        self.assertEqual([(f.timestamp, f.token_count) for f in memory.candidates(12)], before)

    def test_lazy_selection_matches_eager_gathering(self):
        eager, lazy = bank(tokens=40), bank(tokens=40)
        for timestamp in range(30):
            frame = native_frame(timestamp)
            eager.observe(frame)
            eager.candidates(timestamp + 1)  # gather after every frame
            lazy.observe(frame)
        a, b = eager.candidates(30), lazy.candidates(30)
        self.assertEqual([(f.timestamp, f.source_indices) for f in a], [(f.timestamp, f.source_indices) for f in b])
        for x, y in zip(a, b):
            torch.testing.assert_close(x.features, y.features)
        stats = [m.stats() for m in (eager, lazy)]
        self.assertEqual(stats[0], stats[1])


    def test_successor_curvature_keeps_streaming_allocation_proportional(self):
        generator = torch.Generator().manual_seed(7)
        walk, frames = torch.zeros(6), []
        for timestamp in range(80):
            # Back-and-forth motion: interior curvature mostly above the endpoint value 1.
            walk = walk + 0.3 * torch.randn(6, generator=generator) + (2 * (timestamp % 2) - 1) * torch.ones(6)
            features = walk + torch.randn(64, 6, generator=generator)
            frames.append(VisualFrame(float(timestamp), (1, 16, 16), features, (features + 1,), (448, 448)))
        memory = bank(tokens=1200, min_frame_tokens=2)
        for frame, successor in zip(frames, frames[1:]):
            memory.observe(frame, now=successor.timestamp, successor=successor)
        kept = [f.token_count for f in memory.candidates(1e9)]
        means = torch.stack([r.original_mean for r in memory._records])
        weights = temporal_weights(means, successor=memory._successor).tolist()
        ideal = allocate_tokens(weights, [64] * len(kept), 1200, [2] * len(kept))
        self.assertEqual(sum(kept), sum(ideal))
        self.assertLessEqual(max(abs(a - b) for a, b in zip(kept, ideal)), 1)
        with self.assertRaises(ValueError):
            memory.observe(frames[-1], now=100.0, successor=frames[0])


if __name__ == "__main__":
    unittest.main()

"""CPU-only contract tests for the caption-free visual-memory prototype.

Run from anywhere with a Python environment containing torch:
    python -m unittest discover -s <this directory> -p test_visual_memory.py -v
No model weights, video datasets, CUDA, or network access are required.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onestreamer.visual_memory import (
    MemoryConfig,
    ProgressiveVisualMemoryBank,
    RecentFeatureWindow,
    StreamSampler,
    VisualFrame,
    VisualMemoryBank,
    cached_image_features,
    pool_visual_frame,
)
from onestreamer import visual_memory


def frame(timestamp, tokens=4, width=8, layers=2, value=None):
    value = float(timestamp if value is None else value)
    features = torch.full((tokens, width), value, dtype=torch.float32)
    deepstack = tuple(
        torch.full((tokens, width), value + layer + 1, dtype=torch.float32)
        for layer in range(layers)
    )
    return VisualFrame(
        timestamp=float(timestamp),
        grid_thw=(1, 2, 2 * tokens),
        features=features,
        deepstack=deepstack,
        size=(tokens * 28, 28),
    )


class VisualMemoryBankTests(unittest.TestCase):
    def test_payload_bytes_include_every_deepstack_layer(self):
        item = frame(0, tokens=5, width=8, layers=3)
        expected = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (item.features, *item.deepstack)
        )
        self.assertEqual(item.token_count, 5)
        self.assertEqual(item.nbytes, expected)
        self.assertEqual(item.nbytes, 4 * item.features.numel() * 4)

    def test_long_variable_stream_respects_all_storage_and_read_budgets(self):
        config = MemoryConfig(
            capacity=11,
            store_tokens=35,
            store_bytes=2400,
            read_frames=5,
            read_tokens=12,
            seed=17,
        )
        bank = VisualMemoryBank(config)
        for index in range(300):
            bank.observe(frame(index, tokens=1 + index % 7))
            stats = bank.stats()
            self.assertEqual(stats["seen"], index + 1)
            self.assertLessEqual(stats["stored_frames"], config.capacity)
            self.assertLessEqual(stats["stored_tokens"], config.store_tokens)
            self.assertLessEqual(stats["stored_bytes"], config.store_bytes)
            read = bank.read(before=float(index) - 2)
            self.assertLessEqual(len(read), config.read_frames)
            self.assertLessEqual(sum(item.token_count for item in read), config.read_tokens)
            self.assertTrue(all(item.timestamp < index - 2 for item in read))
            times = [item.timestamp for item in read]
            self.assertEqual(times, sorted(set(times)))

    def test_deepstack_bytes_limit_actual_stored_payload(self):
        config = MemoryConfig(
            capacity=10, store_tokens=100, store_bytes=500,
            read_frames=10, read_tokens=100,
        )
        bank = VisualMemoryBank(config)
        for timestamp in range(20):
            bank.observe(frame(timestamp, tokens=4, width=8, layers=2))
            stored = bank.read(before=100)
            actual_bytes = sum(item.nbytes for item in stored)
            self.assertLessEqual(actual_bytes, config.store_bytes)
            self.assertLessEqual(len(stored), 1)
            self.assertEqual(bank.stats()["stored_bytes"], actual_bytes)

    def test_read_excludes_cutoff_recent_and_future_frames(self):
        bank = VisualMemoryBank(MemoryConfig(capacity=20, read_frames=20, read_tokens=100))
        for timestamp in (0, 0.25, 1, 2, 3, 4):
            bank.observe(frame(timestamp))
        self.assertEqual([item.timestamp for item in bank.read(before=2)], [0, 0.25, 1])
        self.assertEqual(bank.read(before=0), [])
        self.assertEqual(bank.read(before=-1), [])
        # A repeated or earlier read must not change which evidence survived writing.
        before = [item.timestamp for item in bank.read(before=5)]
        bank.read(before=1)
        self.assertEqual([item.timestamp for item in bank.read(before=5)], before)

    def test_seed_reproduces_bank_and_reads(self):
        config = MemoryConfig(capacity=8, read_frames=8, read_tokens=100, seed=83)
        left, right = VisualMemoryBank(config), VisualMemoryBank(config)
        for timestamp in range(100):
            left.observe(frame(timestamp))
            right.observe(frame(timestamp))
        self.assertEqual(left.stats(), right.stats())
        self.assertEqual(
            [item.timestamp for item in left.read(before=101)],
            [item.timestamp for item in right.read(before=101)],
        )

    def test_oversized_frame_does_not_evict_valid_evidence(self):
        config = MemoryConfig(
            capacity=10, store_tokens=10, store_bytes=1000,
            read_frames=10, read_tokens=10, seed=1,
        )
        bank = VisualMemoryBank(config)
        bank.observe(frame(0, tokens=2))
        bank.observe(frame(1, tokens=2))
        baseline = bank.read(before=100)
        bank.observe(frame(2, tokens=11))  # Exceeds a single-frame token budget.
        bank.observe(frame(3, tokens=2, width=128))  # Fits tokens but exceeds bytes.
        self.assertEqual(bank.stats()["seen"], 4)
        observed = bank.read(before=100)
        self.assertEqual([item.timestamp for item in observed], [item.timestamp for item in baseline])
        for actual, expected in zip(observed, baseline):
            torch.testing.assert_close(actual.features, expected.features)

    def test_timestamp_must_increase_strictly(self):
        bank = VisualMemoryBank(MemoryConfig())
        bank.observe(frame(2))
        for timestamp in (2, 1):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                bank.observe(frame(timestamp))
        bank.observe(frame(3))
        self.assertEqual([item.timestamp for item in bank.read(before=4)], [2, 3])

    def test_empty_bank_is_readable(self):
        bank = VisualMemoryBank(MemoryConfig())
        self.assertEqual(bank.read(before=100), [])
        stats = bank.stats()
        self.assertEqual(
            [stats[key] for key in ("seen", "stored_frames", "stored_tokens", "stored_bytes")],
            [0, 0, 0, 0],
        )

    def test_fill_candidates_ignore_fixed_read_caps_but_keep_causal_cutoff(self):
        config = MemoryConfig(capacity=20, read_frames=4, read_tokens=1024, read_mode="fill")
        bank = VisualMemoryBank(config)
        for timestamp in range(10):
            bank.observe(frame(timestamp, tokens=1500 if timestamp == 1 else 128))
        candidates = bank.candidates(before=7)
        self.assertEqual([item.timestamp for item in candidates], list(range(7)))
        self.assertGreater(len(candidates), config.read_frames)
        self.assertGreater(sum(item.token_count for item in candidates), config.read_tokens)
        self.assertGreater(candidates[1].token_count, config.read_tokens)
        self.assertEqual(bank.candidates(before=0), [])

    def test_fill_enablement_depends_on_storage_not_fixed_read_limits(self):
        self.assertTrue(MemoryConfig(read_mode="fill", read_frames=0, read_tokens=0).enabled)
        self.assertFalse(MemoryConfig(read_mode="fill", capacity=0).enabled)


class FeaturePoolingTests(unittest.TestCase):
    def spatial_frame(self, height, width):
        values = torch.arange(height * width, dtype=torch.float32).reshape(-1, 1)
        features = torch.cat([values, values * 10], dim=1)
        return VisualFrame(
            timestamp=5.0, grid_thw=(1, height * 2, width * 2),
            features=features, deepstack=(features + 100, features - 100),
            size=(width * 28, height * 28), source_size=(width * 28 - 1, height * 28 - 2),
        )

    def test_even_grid_matches_hand_computed_spatial_means_in_all_branches(self):
        original = self.spatial_frame(4, 4)
        snapshots = [tensor.clone() for tensor in (original.features, *original.deepstack)]
        pooled = pool_visual_frame(original, stride=2, merge_size=2)
        scalar = torch.tensor([2.5, 4.5, 10.5, 12.5]).reshape(-1, 1)
        expected = torch.cat([scalar, scalar * 10], dim=1)
        torch.testing.assert_close(pooled.features, expected)
        torch.testing.assert_close(pooled.deepstack[0], expected + 100)
        torch.testing.assert_close(pooled.deepstack[1], expected - 100)
        self.assertEqual(pooled.grid_thw, (1, 4, 4))
        self.assertEqual(pooled.size, (56, 56))
        self.assertEqual(pooled.source_size, original.source_size)
        self.assertEqual(pooled.timestamp, original.timestamp)
        self.assertEqual(pooled.nbytes * 4, original.nbytes)
        for actual, snapshot in zip((original.features, *original.deepstack), snapshots):
            torch.testing.assert_close(actual, snapshot)

    def test_odd_grid_retains_right_and_bottom_edges_without_zero_padding_bias(self):
        original = self.spatial_frame(3, 5)
        pooled = pool_visual_frame(original)
        scalar = torch.tensor([3.0, 5.0, 6.5, 10.5, 12.5, 14.0]).reshape(-1, 1)
        expected = torch.cat([scalar, scalar * 10], dim=1)
        torch.testing.assert_close(pooled.features, expected)
        self.assertEqual(pooled.grid_thw, (1, 4, 6))
        self.assertEqual(pooled.size, (84, 56))
        self.assertEqual(pooled.token_count, 6)

    def test_stride_one_is_value_identity_with_independent_cpu_storage(self):
        original = self.spatial_frame(3, 5)
        pooled = pool_visual_frame(original, stride=1)
        self.assertEqual(pooled.grid_thw, original.grid_thw)
        self.assertEqual(pooled.size, original.size)
        for actual, expected in zip((pooled.features, *pooled.deepstack),
                                    (original.features, *original.deepstack)):
            torch.testing.assert_close(actual, expected)
            self.assertEqual(actual.device.type, "cpu")
            self.assertNotEqual(actual.data_ptr(), expected.data_ptr())

    def test_pooled_storage_does_not_retain_large_input_backing_buffers(self):
        tensors = tuple(torch.arange(1000, dtype=torch.float32)[:32].reshape(16, 2) for _ in range(3))
        original = VisualFrame(0.0, (1, 8, 8), tensors[0], tensors[1:], (112, 112))
        self.assertEqual(original.features.untyped_storage().nbytes(), 4000)
        pooled = pool_visual_frame(original)
        for tensor in (pooled.features, *pooled.deepstack):
            self.assertEqual(tensor.untyped_storage().nbytes(), tensor.numel() * tensor.element_size())
            self.assertEqual(tensor.device.type, "cpu")
        self.assertLess(pooled.nbytes, original.nbytes)

    def test_pooling_mode_rejects_a_different_history_sampling_rate(self):
        with self.assertRaises(ValueError):
            MemoryConfig(history_compression="avg_pool", history_fps=2.0)

    def test_repeated_odd_grid_pooling_weights_original_cell_areas(self):
        original = self.spatial_frame(3, 5)
        first = pool_visual_frame(original)
        second = pool_visual_frame(first)
        direct = pool_visual_frame(original, stride=4)
        self.assertEqual(second.pool_source_shape, (3, 5))
        self.assertEqual(second.pool_scale, 4)
        for actual, expected in zip((second.features, *second.deepstack),
                                    (direct.features, *direct.deepstack)):
            torch.testing.assert_close(actual, expected)
        final = pool_visual_frame(second)
        self.assertEqual(final.token_count, 1)
        for actual, source in zip((final.features, *final.deepstack),
                                  (original.features, *original.deepstack)):
            torch.testing.assert_close(actual, source.mean(dim=0, keepdim=True))
        saturated = pool_visual_frame(final)
        self.assertEqual(saturated.pool_scale, final.pool_scale)
        torch.testing.assert_close(saturated.features, final.features)


class ProgressiveVisualMemoryBankTests(unittest.TestCase):
    def bank(self, **overrides):
        values = dict(history_compression="progressive_pool", pool_age_seconds=2.0,
                      read_frames=64, read_tokens=16384)
        values.update(overrides)
        return ProgressiveVisualMemoryBank(MemoryConfig(**values))

    def assert_counters_match_payload(self, bank):
        records = bank.candidates(before=10000)
        stats = bank.stats()
        self.assertEqual(stats["stored_frames"], len(records))
        self.assertEqual(stats["stored_tokens"], sum(item.token_count for item in records))
        self.assertEqual(stats["stored_bytes"], sum(item.nbytes for item in records))
        self.assertLessEqual(stats["stored_frames"], bank.config.capacity)
        self.assertLessEqual(stats["stored_tokens"], bank.config.store_tokens)
        self.assertLessEqual(stats["stored_bytes"], bank.config.store_bytes)
        self.assertLessEqual(stats["pending_pool_events"], 2 * stats["stored_frames"] + 64)

    def test_age_starts_at_retirement_and_same_time_does_not_pool_twice(self):
        bank = self.bank()
        bank.observe(frame(0, tokens=64), now=16)
        self.assertEqual(bank.candidates(before=1)[0].token_count, 32)
        bank.advance(17.99)
        self.assertEqual(bank.candidates(before=1)[0].token_count, 32)
        bank.advance(18)
        self.assertEqual(bank.candidates(before=1)[0].token_count, 16)
        after_due = bank.stats()
        bank.advance(18)
        self.assertEqual(bank.stats(), after_due)
        self.assert_counters_match_payload(bank)

    def test_long_jump_matches_incremental_aging_and_stops_at_one_token(self):
        once, incremental = self.bank(), self.bank()
        for bank in (once, incremental):
            bank.observe(frame(0, tokens=64), now=16)
            bank.observe(frame(1, tokens=33), now=17)
        once.advance(100)
        for now in (17.5, 18, 19, 20, 21, 100):
            incremental.advance(now)
        self.assertEqual(once.stats(), incremental.stats())
        self.assertEqual(once.stats()["pending_pool_events"], 0)
        for left, right in zip(once.candidates(10), incremental.candidates(10)):
            self.assertEqual(left.token_count, 1)
            self.assertEqual(left.grid_thw, right.grid_thw)
            torch.testing.assert_close(left.features, right.features)
            for a, b in zip(left.deepstack, right.deepstack):
                torch.testing.assert_close(a, b)
        saturated = once.stats()
        once.advance(10000)
        self.assertEqual(once.stats(), saturated)

    def test_oldest_eviction_and_counters_respect_each_budget(self):
        configurations = (
            dict(capacity=2, store_tokens=100, store_bytes=100000),
            dict(capacity=10, store_tokens=10, store_bytes=100000),
            dict(capacity=10, store_tokens=100, store_bytes=1000),
        )
        for caps in configurations:
            with self.subTest(caps=caps):
                bank = self.bank(pool_age_seconds=10000, **caps)
                for index in range(30):
                    bank.observe(frame(index, tokens=16), now=index + 16)
                    self.assert_counters_match_payload(bank)
                    count = bank.stats()["stored_frames"]
                    self.assertEqual([item.timestamp for item in bank.candidates(1000)],
                                     list(range(index + 1 - count, index + 1)))
                    self.assertEqual(bank.stats()["dropped_oldest"], index + 1 - count)

    def test_due_pooling_frees_budget_before_new_frame_triggers_eviction(self):
        bank = self.bank(store_tokens=6)
        bank.observe(frame(0, tokens=8), now=16)  # Four tokens after admission.
        bank.observe(frame(1, tokens=8), now=18)  # Old becomes two before new four arrive.
        self.assertEqual([item.token_count for item in bank.candidates(2)], [2, 4])
        self.assertEqual(bank.stats()["dropped_oldest"], 0)
        self.assert_counters_match_payload(bank)

    def test_oversized_new_frame_preserves_existing_history(self):
        bank = self.bank(store_tokens=8, pool_age_seconds=1000)
        bank.observe(frame(0, tokens=8), now=16)
        original = bank.candidates(1)[0]
        bank.observe(frame(1, tokens=64), now=17)
        self.assertEqual([item.timestamp for item in bank.candidates(2)], [0])
        self.assertIs(bank.candidates(2)[0], original)
        self.assertEqual(bank.stats()["skipped_oversize"], 1)
        self.assertEqual(bank.stats()["dropped_oldest"], 0)
        self.assert_counters_match_payload(bank)

    def test_dropped_records_do_not_revive_and_pending_heap_stays_bounded(self):
        bank = self.bank(capacity=2, pool_age_seconds=10000)
        for index in range(200):
            bank.observe(frame(index, tokens=16), now=index + 16)
            self.assert_counters_match_payload(bank)
        self.assertEqual([item.timestamp for item in bank.candidates(1000)], [198, 199])
        bank.advance(100000)
        self.assertEqual([item.timestamp for item in bank.candidates(1000)], [198, 199])
        self.assertEqual([item.token_count for item in bank.candidates(1000)], [1, 1])
        self.assertEqual(bank.stats()["pending_pool_events"], 0)
        self.assert_counters_match_payload(bank)

    def test_nonshrinking_stride_and_backward_time_are_rejected(self):
        with self.assertRaises(ValueError):
            MemoryConfig(history_compression="progressive_pool", pool_stride=1)
        bank = self.bank()
        bank.observe(frame(0), now=16)
        with self.assertRaises(ValueError):
            bank.advance(15)
        self.assertEqual(bank.candidates(before=0), [])


class RecentFeatureWindowTests(unittest.TestCase):
    def test_sixteen_full_frames_then_seventeenth_evicts_only_oldest(self):
        window = RecentFeatureWindow()
        originals = [frame(index) for index in range(17)]
        for item in originals[:16]:
            self.assertIsNone(window.observe(item))
        self.assertEqual([item.timestamp for item in window.frames()], list(range(16)))
        self.assertIs(window.observe(originals[16]), originals[0])
        self.assertEqual([item.timestamp for item in window.frames()], list(range(1, 17)))
        for actual, expected in zip(window.frames(), originals[1:]):
            self.assertIs(actual, expected)
        snapshot = window.frames()
        snapshot.clear()
        self.assertEqual(len(window.frames()), 16)

    def test_invalid_timestamp_does_not_corrupt_fifo(self):
        window = RecentFeatureWindow()
        original = frame(2)
        window.observe(original)
        for timestamp in (2, 1, float("nan"), float("inf")):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                window.observe(frame(timestamp))
            self.assertIs(window.frames()[0], original)
            self.assertEqual(len(window.frames()), 1)
        window.observe(frame(3))
        self.assertEqual([item.timestamp for item in window.frames()], [2, 3])


class FakeInnerModel:
    def __init__(self, config):
        self.config = config
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.visual = SimpleNamespace(spatial_merge_size=2, dtype=torch.float32)

    def get_image_features(self, pixel_values, image_grid_thw):
        raise AssertionError("Cached readout must not run the visual encoder again")


def fake_engine():
    config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2, deepstack_visual_indexes=[0, 1]),
        text_config=SimpleNamespace(hidden_size=8),
    )
    inner = FakeInnerModel(config)
    model = SimpleNamespace(model=inner, config=config, dtype=torch.float32, device=torch.device("cpu"))
    return SimpleNamespace(model=model, device=torch.device("cpu"))


class CachedImageFeaturesTests(unittest.TestCase):
    def test_multiframe_preserves_image_split_and_concatenates_each_deepstack_layer(self):
        engine = fake_engine()
        items = [frame(1, tokens=2), frame(3, tokens=5)]
        grid = torch.tensor([item.grid_thw for item in items], dtype=torch.long)
        original = engine.model.model.get_image_features
        with cached_image_features(engine, items) as state:
            features, deepstack = engine.model.model.get_image_features(torch.empty(0), grid)
            self.assertEqual(state["calls"], 1)
            self.assertEqual(len(features), 2)
            for actual, expected in zip(features, items):
                torch.testing.assert_close(actual, expected.features)
            self.assertEqual(len(deepstack), 2)
            for layer, actual in enumerate(deepstack):
                torch.testing.assert_close(actual, torch.cat([item.deepstack[layer] for item in items]))
        self.assertEqual(engine.model.model.get_image_features, original)
        self.assertNotIn("get_image_features", engine.model.model.__dict__)

    def test_cpu_history_and_device_recent_move_before_deepstack_concatenation(self):
        engine = fake_engine()
        engine.device = torch.device("meta")
        history = frame(1, tokens=2)
        cpu_recent = frame(3, tokens=5)
        recent = VisualFrame(
            timestamp=cpu_recent.timestamp, grid_thw=cpu_recent.grid_thw,
            features=cpu_recent.features.to("meta"),
            deepstack=tuple(tensor.to("meta") for tensor in cpu_recent.deepstack),
            size=cpu_recent.size,
        )
        with cached_image_features(engine, [history, recent]):
            final, deepstack = engine.model.model.get_image_features(
                torch.empty(0, device="meta"),
                torch.tensor([history.grid_thw, recent.grid_thw]),
            )
        self.assertTrue(all(tensor.device.type == "meta" for tensor in (*final, *deepstack)))
        self.assertEqual([tuple(tensor.shape) for tensor in final], [(2, 8), (5, 8)])
        self.assertEqual([tuple(tensor.shape) for tensor in deepstack], [(7, 8), (7, 8)])

    def test_exception_restores_existing_instance_override(self):
        engine = fake_engine()

        def original(pixel_values, image_grid_thw):
            return "original"

        engine.model.model.get_image_features = original
        with self.assertRaisesRegex(RuntimeError, "simulated generation failure"):
            with cached_image_features(engine, [frame(0)]):
                raise RuntimeError("simulated generation failure")
        self.assertIs(engine.model.model.__dict__["get_image_features"], original)

    def test_wrong_grid_rejected_and_class_method_restored(self):
        engine = fake_engine()
        original = engine.model.model.get_image_features
        with self.assertRaises((ValueError, RuntimeError)):
            with cached_image_features(engine, [frame(0, tokens=2)]):
                engine.model.model.get_image_features(torch.empty(0), torch.tensor([[1, 2, 6]]))
        self.assertEqual(engine.model.model.get_image_features, original)
        self.assertNotIn("get_image_features", engine.model.model.__dict__)

    def test_wrong_feature_shape_rejected_without_leaving_patch(self):
        engine = fake_engine()
        original = engine.model.model.get_image_features
        with self.assertRaises((ValueError, RuntimeError)):
            malformed = VisualFrame(
                timestamp=0.0, grid_thw=(1, 2, 4),
                features=torch.zeros(3, 8),
                deepstack=(torch.zeros(2, 8), torch.zeros(2, 8)), size=(56, 28),
            )
            with cached_image_features(engine, [malformed]):
                engine.model.model.get_image_features(torch.empty(0), torch.tensor([[1, 2, 4]]))
        self.assertEqual(engine.model.model.get_image_features, original)
        self.assertNotIn("get_image_features", engine.model.model.__dict__)


class FillContextTests(unittest.TestCase):
    def setUp(self):
        # Exercise the real selection policy with an independent token-cost
        # oracle. Real processor equivalence is checked by the model smoke test.
        self.adapter = object.__new__(visual_memory.NativeFeatureAdapter)
        self.adapter.engine = fake_engine()
        self.adapter.engine.model.config.text_config.max_position_embeddings = 2048
        self.recent = [frame(100), frame(101)]

        def input_cost(history, recent, question, direct=False):
            return 100 + sum(item.token_count for item in recent) + sum(
                item.token_count + 3 for item in history
            )

        self.adapter.count_input_tokens = input_cost

    def test_fill_can_use_more_than_four_frames_and_1024_visual_tokens(self):
        candidates = [frame(index, tokens=200) for index in range(8)]
        recent_ids = [id(item) for item in self.recent]
        selected, meta = self.adapter.select_history(candidates, self.recent, "question")
        self.assertEqual([item.timestamp for item in selected], list(range(8)))
        self.assertGreater(len(selected), 4)
        self.assertGreater(sum(item.token_count for item in selected), 1024)
        self.assertEqual([id(item) for item in self.recent], recent_ids)
        self.assertEqual(meta["selected_input_tokens"], self.adapter.count_input_tokens(selected, self.recent, "question"))
        self.assertLessEqual(meta["selected_input_tokens"] + 16, 2048)

    def test_oversized_candidate_does_not_block_smaller_whole_frames(self):
        candidates = [frame(index, tokens=tokens) for index, tokens in enumerate((1000, 100, 80, 60, 40))]
        selected, meta = self.adapter.select_history(
            candidates, self.recent, "question", context_tokens=440,
        )
        self.assertEqual([item.timestamp for item in selected], [1, 2, 3, 4])
        for actual, original in zip(selected, candidates[1:]):
            self.assertIs(actual, original)
        self.assertLessEqual(meta["selected_input_tokens"] + 16, 440)

    def test_output_reserve_and_model_context_limit_are_enforced(self):
        candidate = frame(1, tokens=1)  # Adds four tokens, including text overhead.
        selected, meta = self.adapter.select_history([candidate], self.recent, "question", context_tokens=128)
        self.assertEqual(selected, [candidate])
        self.assertEqual(meta["remaining_input_tokens"], 0)
        self.assertEqual(meta["answer_reserve_tokens"], 16)
        selected, _ = self.adapter.select_history([candidate], self.recent, "question", context_tokens=127)
        self.assertEqual(selected, [])
        self.adapter.engine.model.config.text_config.max_position_embeddings = 128
        selected, meta = self.adapter.select_history([candidate], self.recent, "question", context_tokens=10000)
        self.assertEqual(meta["context_limit"], 128)
        self.assertEqual(selected, [candidate])

    def test_recent_and_question_overflow_is_an_error(self):
        with self.assertRaises(ValueError):
            self.adapter.select_history([], self.recent, "question", context_tokens=123)
        selected, meta = self.adapter.select_history([], self.recent, "question", context_tokens=124)
        self.assertEqual(selected, [])
        self.assertEqual(meta["remaining_input_tokens"], 0)

    def test_future_or_duplicate_candidate_is_rejected_even_if_too_large(self):
        for candidates in ([frame(100, tokens=3000)], [frame(1), frame(1)]):
            with self.subTest(timestamps=[item.timestamp for item in candidates]), self.assertRaises(ValueError):
                self.adapter.select_history(candidates, self.recent, "question", context_tokens=512)

    def test_answer_rechecks_real_processor_length_before_generation(self):
        with patch.object(self.adapter, "prepare_inputs") as prepare:
            with self.assertRaises(ValueError):
                self.adapter.answer([], self.recent, "question", context_tokens=123)
            prepare.assert_not_called()
        with patch.object(self.adapter, "prepare_inputs", return_value=({}, {"input_tokens": 109})):
            with self.assertRaises(RuntimeError):
                self.adapter.answer([], self.recent, "question", context_tokens=124)

    def test_large_candidate_bank_handles_all_fit_and_skips_oversized_frames(self):
        self.adapter.engine.model.config.text_config.max_position_embeddings = 10000
        recent = [frame(1000), frame(1001)]
        candidates = [frame(index, tokens=8) for index in range(180)]
        all_selected, all_meta = self.adapter.select_history(candidates, recent, "question")
        self.assertEqual([item.timestamp for item in all_selected], list(range(180)))
        self.assertLessEqual(all_meta["selected_input_tokens"] + 16, 10000)
        candidates[0] = frame(0, tokens=2000)
        selected, meta = self.adapter.select_history(candidates, recent, "question", context_tokens=600)
        self.assertTrue(selected)
        self.assertNotIn(0, [item.timestamp for item in selected])
        self.assertLessEqual(self.adapter.count_input_tokens(selected, recent, "question") + 16, 600)
        self.assertEqual([item.timestamp for item in selected], sorted({item.timestamp for item in selected}))
        self.assertGreater(meta["budget_count_calls"], 0)


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is needed only for synthetic-video tests")
class StreamSamplerTests(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "synthetic.avi"
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32))
        if not writer.isOpened():
            self.skipTest("This OpenCV build cannot write MJPG test videos")
        try:
            for index in range(40):
                writer.write(np.full((32, 32, 3), index * 5, dtype=np.uint8))
        finally:
            writer.release()

    def sampler(self, start=0.0, fps=2.0):
        sampler = StreamSampler(self.path, start, fps)
        self.addCleanup(sampler.close)
        return sampler

    def test_incremental_queries_keep_fixed_sampling_origin(self):
        incremental = self.sampler(start=0.15, fps=2.0)
        observed = []
        for cutoff in (0.52, 1.03, 1.51, 2.05):
            chunk = list(incremental.advance(cutoff))
            self.assertTrue(all(timestamp < cutoff for timestamp, _ in chunk))
            observed.extend(chunk)
        once = list(self.sampler(start=0.15, fps=2.0).advance(2.05))
        self.assertEqual([timestamp for timestamp, _ in observed], [0.1, 0.6, 1.1, 1.6])
        self.assertEqual([timestamp for timestamp, _ in observed], [timestamp for timestamp, _ in once])
        for (_, left), (_, right) in zip(observed, once):
            self.assertEqual(left.tobytes(), right.tobytes())

    def test_exact_cutoff_is_excluded_and_decoder_does_not_prefetch(self):
        import cv2

        sampler = self.sampler(fps=2.0)
        self.assertEqual(list(sampler.advance(0.0)), [])
        first = list(sampler.advance(0.5))
        self.assertEqual([timestamp for timestamp, _ in first], [0.0])
        self.assertEqual(sampler.capture.get(cv2.CAP_PROP_POS_FRAMES), 1.0)
        self.assertEqual(list(sampler.advance(0.5)), [])
        second = list(sampler.advance(0.5001))
        self.assertEqual([timestamp for timestamp, _ in second], [0.5])
        self.assertEqual(sampler.capture.get(cv2.CAP_PROP_POS_FRAMES), 6.0)

    def test_high_target_fps_deduplicates_source_frames(self):
        sampler = self.sampler(fps=25.0)
        observations = list(sampler.advance(1.0))
        self.assertEqual([timestamp for timestamp, _ in observations], [index / 10 for index in range(10)])

    def test_end_of_video_and_backward_cutoff(self):
        sampler = self.sampler(fps=2.0)
        observations = list(sampler.advance(5.0))
        self.assertEqual([timestamp for timestamp, _ in observations], [index / 2 for index in range(8)])
        self.assertEqual(list(sampler.advance(6.0)), [])
        with self.assertRaises(ValueError):
            list(sampler.advance(5.0))


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.samples = self.root / "samples.json"
        self.output = self.root / "output"
        (self.root / "example.avi").write_bytes(b"dry-run only checks file existence")
        self.row = dict(
            sample_id="ovo-test", index=0, task="backward_tracking", subtask="EPM",
            video="example.avi", end=20, question="Which object was used?",
            options={"A": "cup", "B": "plate"}, answer="A",
        )

    def dry_run(self, rows, extra=()):
        self.samples.write_text(json.dumps(rows), encoding="utf-8")
        script = Path(__file__).resolve().parents[1] / "src/onestreamer/visual_memory.py"
        # -S excludes site-packages: validation must not require torch, transformers,
        # CUDA, OpenCV, or loading the nonexistent model path.
        return subprocess.run(
            [sys.executable, "-S", str(script), "--samples", str(self.samples),
             "--video-root", str(self.root), "--model-path", "/not-a-model",
             "--output", str(self.output), "--dry-run", *extra],
            check=False, capture_output=True, text=True, timeout=20,
        )

    def test_dry_run_defaults_start_and_loads_no_inference_dependencies(self):
        result = self.dry_run([self.row])
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["samples"], 1)
        self.assertEqual(report["streams"], 1)
        self.assertFalse(self.output.exists())

    def test_dry_run_distinguishes_stream_starts(self):
        other = dict(self.row, sample_id="ovo-test-2", index=1, start=1.0)
        result = self.dry_run([self.row, other])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["streams"], 2)

    def test_dry_run_rejects_invalid_time_and_duplicate_ids(self):
        for rows, message in (
            ([dict(self.row, start=-1)], "Invalid stream bounds"),
            ([dict(self.row, end=float("nan"))], "Invalid stream bounds"),
            ([self.row, dict(self.row)], "Duplicate sample IDs"),
        ):
            with self.subTest(message=message):
                result = self.dry_run(rows)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertFalse(self.output.exists())

    def test_progressive_cli_default_capacity_tracks_token_cap_and_control_stays_zero(self):
        for mode, explicit, expected in (
            ("resize", (), 64),
            ("avg_pool", (), 64),
            ("progressive_pool", (), 321),
            ("progressive_pool", ("--history-capacity", "0"), 0),
        ):
            with self.subTest(mode=mode, explicit=explicit):
                result = self.dry_run([self.row], (
                    "--history-compression", mode, "--history-store-tokens", "321", *explicit,
                ))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["config"]["capacity"], expected)

    def test_recent_frames_cli_passes_positive_values_and_rejects_nonpositive(self):
        for count in (1, 4, 8, 16, 32):
            with self.subTest(recent_frames=count):
                result = self.dry_run([self.row], ("--recent-frames", str(count)))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["config"]["recent_frames"], count)
        for count in (0, -1):
            with self.subTest(recent_frames=count):
                result = self.dry_run([self.row], ("--recent-frames", str(count)))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("recent_frames", result.stderr)
                self.assertFalse(self.output.exists())


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is needed only for synthetic-video tests")
class RunnerIntegrationTests(unittest.TestCase):
    def test_pooling_runner_reuses_full_recent_features_and_archives_only_evictions(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "pool-stream.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (32, 32))
            if not writer.isOpened():
                self.skipTest("This OpenCV build cannot write MJPG test videos")
            try:
                for index in range(80):
                    writer.write(np.full((32, 32, 3), index * 3, dtype=np.uint8))
            finally:
                writer.release()
            template = dict(
                task="backward_tracking", subtask="EPM", video=video.name, start=0.0,
                question="Which object was used?", options={"A": "cup", "B": "plate"},
                answer="A", answer_text="cup", eval_mode="mcq_letter",
            )
            rows = [dict(template, sample_id=name, index=index, end=end)
                    for index, (name, end) in enumerate((
                        ("last", 20.0), ("same-b", 17.0), ("first", 16.0), ("same-a", 17.0),
                    ))]
            job = dict(model="test-model", model_path="/unused-model", video_root=str(root))

            class FakeAdapter:
                def __init__(self, engine):
                    self.engine = engine

                def encode(self, images, max_pixels=None, store_on_cpu=True):
                    if max_pixels is not None or store_on_cpu:
                        raise AssertionError("Pooling mode must encode full recent features directly")
                    result = []
                    for timestamp, image in images:
                        self.engine.encoded.append(timestamp)
                        item = frame(timestamp)
                        item = VisualFrame(item.timestamp, item.grid_thw, item.features,
                                           item.deepstack, item.size, source_size=image.size)
                        self.engine.originals[timestamp] = item
                        result.append(item)
                    return result

                def answer(self, history, recent, question, direct=False, max_tokens=16,
                           context_tokens=None):
                    self.engine.recent_queries.append(list(recent))
                    self.engine.history_queries.append(list(history))
                    tokens = sum(item.token_count for item in [*history, *recent])
                    return "A", dict(input_tokens=tokens + 20, visual_tokens=tokens, text_tokens=20)

            engine_module = ModuleType("onestreamer.engine")
            engine_module.OneStreamerEngine = lambda **kwargs: self.fail("Unexpected model load")
            all_recent = {}
            cases = [(mode, origin, capacity, recent_count)
                     for mode in ("avg_pool", "progressive_pool")
                     for origin in (0.0, 0.15) for capacity in (64, 0)
                     for recent_count in (1, 4, 8, 16, 32)]
            for mode, origin, capacity, recent_count in cases:
                with self.subTest(mode=mode, origin=origin, capacity=capacity, recent_frames=recent_count):
                    engine = fake_engine()
                    engine.encoded, engine.originals = [], {}
                    engine.recent_queries, engine.history_queries = [], []
                    config = MemoryConfig(
                        capacity=capacity, history_compression=mode, read_frames=64,
                        pool_age_seconds=2.0, recent_frames=recent_count,
                    )
                    stream_rows = [dict(row, start=origin, end=origin + (row["end"] + recent_count - 16))
                                   for row in rows]
                    with patch.dict(sys.modules, {"onestreamer.engine": engine_module}), \
                            patch.object(visual_memory, "NativeFeatureAdapter", FakeAdapter), \
                            patch("onestreamer.video.read_recent_frames", side_effect=AssertionError("Recent re-decoded")), \
                            patch.object(visual_memory, "pool_visual_frame", wraps=pool_visual_frame) as pool:
                        results = list(visual_memory.infer(job, stream_rows, config, engine=engine))
                    self.assertEqual(engine.encoded, list(range(recent_count + 4)))
                    expected_pools = 7 if mode == "progressive_pool" else 4
                    self.assertEqual(pool.call_count, expected_pools if capacity else 0)
                    self.assertEqual([row["sample_id"] for row in results], ["first", "same-a", "same-b", "last"])
                    self.assertEqual([row["stream_encoded_frames_cumulative"] for row in results],
                                     [recent_count, recent_count + 1, recent_count + 1, recent_count + 4])
                    self.assertEqual([row["bank"]["seen"] for row in results], [0, 1, 1, 4] if capacity else [0] * 4)
                    expected_recent = [list(range(recent_count)), list(range(1, recent_count + 1)),
                                       list(range(1, recent_count + 1)), list(range(4, recent_count + 4))]
                    self.assertEqual([row["recent_timestamps"] for row in results], expected_recent)
                    for recent in engine.recent_queries:
                        for item in recent:
                            self.assertIs(item, engine.originals[item.timestamp])
                            self.assertEqual(item.token_count, 4)
                    for row in results:
                        self.assertEqual(row["recent_sampling"], "fixed_origin_fifo")
                        self.assertTrue(all(timestamp < row["recent_timestamps"][0]
                                            for timestamp in row["history_timestamps"]))
                        self.assertTrue(all(size == [32, 32] for size in row["recent_image_sizes"]))
                    self.assertIn(f"_{mode}_fixed", results[0]["variant"])
                    self.assertIn(f"_r{recent_count}", results[0]["variant"])
                    self.assertEqual(results[0]["variant"].endswith(f"_recent{recent_count}_control"), not bool(capacity))
                    if capacity:
                        self.assertEqual([row["history_timestamps"] for row in results], [[], [0], [0], [0, 1, 2, 3]])
                        self.assertEqual(results[-1]["pooled_input_tokens_cumulative"],
                                         22 if mode == "progressive_pool" else 16)
                        self.assertEqual(results[-1]["pooled_output_tokens_cumulative"],
                                         11 if mode == "progressive_pool" else 8)
                        if mode == "progressive_pool":
                            self.assertEqual(results[1]["bank"], results[2]["bank"])
                            self.assertEqual(results[-1]["history_frame_tokens"], [1, 1, 1, 2])
                    else:
                        self.assertTrue(all(not row["history_timestamps"] for row in results))
                    all_recent[mode, origin, capacity, recent_count] = [
                        [item.features.clone() for item in recent] for recent in engine.recent_queries]
            for mode in ("avg_pool", "progressive_pool"):
                for origin in (0.0, 0.15):
                    for recent_count in (1, 4, 8, 16, 32):
                        for pooled_query, control_query in zip(all_recent[mode, origin, 64, recent_count],
                                                              all_recent[mode, origin, 0, recent_count]):
                            for pooled_recent, control_recent in zip(pooled_query, control_query):
                                torch.testing.assert_close(pooled_recent, control_recent)

    def test_runner_orders_queries_preserves_causality_and_uses_official_scoring(self):
        import cv2
        import numpy as np
        from onestreamer.metrics import aggregate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "long-enough-for-history.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (32, 32))
            if not writer.isOpened():
                self.skipTest("This OpenCV build cannot write MJPG test videos")
            try:
                for index in range(48):
                    writer.write(np.full((32, 32, 3), index * 5, dtype=np.uint8))
            finally:
                writer.release()

            template = dict(
                task="backward_tracking", subtask="EPM", video=video.name, start=0.0,
                question="Which object was used?", options={"A": "cup", "B": "plate"},
                answer="A", answer_text="cup", eval_mode="mcq_letter",
            )
            rows = [
                dict(template, sample_id="late-z", index=2, end=20.0),
                dict(template, sample_id="early", index=0, end=18.0),
                dict(template, sample_id="late-a", index=1, end=20.0,
                     question="How many cups?", options={}, answer="2", answer_text="2",
                     eval_mode="integer_match"),
            ]
            job = dict(model="test-model", model_path="/unused-model", video_root=str(root))

            class FakeAdapter:
                def __init__(self, engine):
                    self.engine = engine

                def encode(self, images, max_pixels=None, store_on_cpu=True):
                    timestamps = [timestamp for timestamp, _ in images]
                    if max_pixels is not None:
                        self.engine.history_encoded.extend(timestamps)
                    else:
                        self.engine.recent_encoded.append(timestamps)
                    return [frame(timestamp) for timestamp in timestamps]

                def select_history(self, candidates, recent, question_text, direct=False, max_tokens=16,
                                   context_tokens=None):
                    self.engine.fill_calls.append((list(candidates), context_tokens))
                    return list(candidates), {"eligible_history_frames": len(candidates)}

                def read_action_token_ids(self):
                    return {"A": 53, "B": 54}

                def context_limit(self, cap):
                    return cap

                def count_input_tokens(self, history, recent, question_text, direct=False):
                    return 20 + sum(item.token_count for item in history + recent)

                def count_read_action_input_tokens(self, history, recent, question_text, status, direct=False):
                    return self.count_input_tokens(history, recent, question_text, direct) + 30

                def probe_read_action(self, history, recent, question_text, status,
                                      direct=False, context_tokens=None):
                    self.engine.action_calls.append((list(history), list(recent), question_text, status))
                    return {"action": "READ_MORE"}

                def answer(self, history, recent, question_text, direct, max_tokens, context_tokens=None):
                    self.engine.answer_calls.append((history, recent, question_text, direct, max_tokens))
                    visual_tokens = sum(item.token_count for item in (*history, *recent))
                    return ("2" if direct else "A"), dict(
                        input_tokens=visual_tokens + 20, visual_tokens=visual_tokens, text_tokens=20,
                    )

            # infer receives an engine, so loading the real engine constructor would
            # be unnecessary. Keep this test independent of model installations.
            engine_module = ModuleType("onestreamer.engine")
            engine_module.OneStreamerEngine = lambda **kwargs: self.fail("Unexpected model load")
            cases = [(capacity, read_mode, recent_count, origin)
                     for capacity, read_mode in ((64, "fixed"), (0, "fixed"), (64, "fill"), (0, "fill"),
                                                 (64, "action"), (0, "action"))
                     for recent_count in (1, 4, 8, 16, 32)
                     for origin in ((0.0, 0.15) if read_mode == "action" else (0.0,))]
            for capacity, read_mode, recent_count, origin in cases:
                with self.subTest(history_capacity=capacity, read_mode=read_mode,
                                  recent_frames=recent_count, origin=origin):
                    engine = SimpleNamespace(
                        device="cpu", history_encoded=[], recent_encoded=[], answer_calls=[], fill_calls=[],
                        action_calls=[],
                    )
                    with patch.dict(sys.modules, {"onestreamer.engine": engine_module}), \
                            patch.object(visual_memory, "NativeFeatureAdapter", FakeAdapter):
                        results = list(visual_memory.infer(
                            job, [dict(row, start=origin) for row in rows],
                            MemoryConfig(capacity=capacity, read_mode=read_mode,
                                                    context_tokens=8192, recent_frames=recent_count), engine=engine,
                        ))
                    self.assertEqual([row["sample_id"] for row in results], ["early", "late-a", "late-z"])
                    self.assertEqual([row["hit"] for row in results], [1, 1, 1])
                    self.assertEqual([row["parsed_output"] for row in results], ["A", "2", "A"])
                    self.assertEqual(len(engine.recent_encoded), 3)
                    self.assertEqual(engine.recent_encoded[1], engine.recent_encoded[2])
                    self.assertEqual([len(timestamps) for timestamps in engine.recent_encoded],
                                     [min(recent_count, 18), min(recent_count, 20), min(recent_count, 20)])
                    for result in results:
                        self.assertEqual(result["recent_frames"], recent_count)
                        self.assertEqual(result["recent_read_frames"], len(result["recent_timestamps"]))
                        self.assertTrue(all(
                            timestamp < result["recent_timestamps"][0]
                            for timestamp in result["history_timestamps"]
                        ))
                        self.assertTrue(all(timestamp < result["end"] for timestamp in result["recent_timestamps"]))
                    self.assertEqual(results[1]["bank"]["seen"], results[2]["bank"]["seen"])
                    self.assertEqual(engine.history_encoded, list(range(20)) if capacity else [])
                    if capacity:
                        self.assertEqual([row["bank"]["seen"] for row in results], [18, 20, 20])
                        self.assertTrue(all(row["history_timestamps"] for row in results))
                    else:
                        self.assertTrue(all(row["bank"]["seen"] == 0 for row in results))
                        self.assertTrue(all(row["history_timestamps"] == [] for row in results))
                    if read_mode == "fill":
                        self.assertEqual(len(engine.fill_calls), 3)
                        self.assertTrue(all(cap == 8192 for _, cap in engine.fill_calls))
                        if capacity:
                            self.assertEqual(results[1]["history_read_frames"], max(1, 21 - recent_count))
                    else:
                        self.assertEqual(engine.fill_calls, [])
                    if read_mode == "action":
                        for result, answer_call in zip(results, engine.answer_calls):
                            self.assertEqual(result["history_timestamps"], [f.timestamp for f in answer_call[0]])
                            self.assertEqual(result["recent_timestamps"], [f.timestamp for f in answer_call[1]])
                            self.assertIn(result["question"], answer_call[2])
                            self.assertEqual(result["action_invalid_count"], 0)
                            self.assertIn(result["action_stop_reason"], ("no_history", "history_exhausted"))
                            for probe in result["action_rounds"]:
                                status = probe["history_status"]
                                self.assertEqual(status["query_time"], result["end"])
                                self.assertEqual(status["stream_start"], result["start"])
                                self.assertLess(status["recent_end"], status["query_time"])
                                self.assertEqual(probe["action"], "READ_MORE")
                        self.assertEqual(len(engine.action_calls), sum(len(r["action_rounds"]) for r in results))
                        if not capacity:
                            self.assertEqual(engine.action_calls, [])
                    summary = aggregate(results)
                    self.assertEqual(summary["rows"], 3)
                    self.assertEqual(summary["metrics"][0]["sample_micro_accuracy"], 1.0)
                    self.assertEqual(
                        summary["metrics"][0]["task_subtask_accuracy"]["backward_tracking"]["EPM"], 1.0,
                    )
                    self.assertIn("Best option:(", engine.answer_calls[0][2])
                    self.assertEqual(engine.answer_calls[1][2], "How many cups?")
                    self.assertTrue(all(call[4] == 16 for call in engine.answer_calls))


if __name__ == "__main__":
    unittest.main()

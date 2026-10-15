"""CPU tests for the context-sized budget_pool history and end-inclusive sampler.

    python -m unittest discover -s <this directory> -p test_budget_memory.py -v
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib.util
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onestreamer.visual_memory import (
    BudgetVisualMemoryBank,
    MemoryConfig,
    StreamSampler,
    VisualFrame,
    context_history_budget,
)
from onestreamer import visual_memory


def grid_frame(timestamp, height=8, width=8, channels=4, layers=2):
    """A merged height x width grid, as Qwen3-VL returns for one image."""
    value = float(timestamp)
    features = torch.full((height * width, channels), value)
    deepstack = tuple(torch.full((height * width, channels), value + i + 1) for i in range(layers))
    return VisualFrame(float(timestamp), (1, 2 * height, 2 * width), features, deepstack,
                       (2 * width * 14, 2 * height * 14))


def budget_bank(tokens, overhead=0, **overrides):
    values = dict(history_compression="budget_pool", capacity=100000, store_tokens=tokens,
                  store_bytes=1 << 30, min_frame_tokens=4)
    values.update(overrides)
    bank = BudgetVisualMemoryBank(MemoryConfig(**values))
    bank.set_budget(tokens, overhead)
    return bank


class BudgetBankTests(unittest.TestCase):
    def assert_invariants(self, bank):
        records = bank.candidates(before=1e9)
        stats = bank.stats()
        self.assertEqual(stats["stored_frames"], len(records))
        self.assertEqual(stats["stored_tokens"], sum(f.token_count for f in records))
        self.assertEqual(stats["stored_bytes"], sum(f.nbytes for f in records))
        self.assertLessEqual(stats["budget_cost"], bank.token_budget)
        self.assertEqual(stats["budget_cost"], stats["stored_tokens"] + bank.frame_overhead * len(records))
        tokens = [f.token_count for f in records]
        # Resolution never increases with age.
        self.assertEqual(tokens, sorted(tokens))
        times = [f.timestamp for f in records]
        self.assertEqual(times, sorted(set(times)))

    def test_under_budget_keeps_every_record_after_admission_pooling(self):
        bank = budget_bank(1000)
        for t in range(10):
            bank.observe(grid_frame(t))
        self.assertEqual([f.timestamp for f in bank.candidates(100)], list(range(10)))
        self.assertTrue(all(f.token_count == 16 for f in bank.candidates(100)))
        self.assertEqual(bank.stats()["dropped_frames"], 0)
        self.assert_invariants(bank)

    def test_pressure_coarsens_oldest_finest_before_dropping(self):
        bank = budget_bank(16 * 3 + 4 * 2)
        for t in range(5):
            bank.observe(grid_frame(t))
            self.assert_invariants(bank)
        self.assertEqual([f.token_count for f in bank.candidates(100)], [4, 4, 16, 16, 16])
        self.assertEqual(bank.stats()["dropped_frames"], 0)

    def test_floor_then_uniform_thinning_keeps_earliest_observation(self):
        bank = budget_bank(40)
        for t in range(200):
            bank.observe(grid_frame(t))
            self.assert_invariants(bank)
            self.assertEqual(bank.candidates(1e9)[0].timestamp, 0.0)
            self.assertEqual(bank.candidates(1e9)[-1].timestamp, float(t))
        times = [f.timestamp for f in bank.candidates(1e9)]
        gaps = [b - a for a, b in zip(times, times[1:])]
        self.assertLessEqual(max(gaps), 4 * min(gaps) + 1)
        self.assertGreater(bank.stats()["dropped_frames"], 0)

    def test_per_image_overhead_counts_against_budget(self):
        bank = budget_bank(10 * (16 + 7), overhead=7)
        for t in range(10):
            bank.observe(grid_frame(t))
        self.assertEqual(bank.stats()["pool_operations"], 10)
        bank.observe(grid_frame(10))
        self.assertGreater(bank.stats()["pool_operations"], 11)
        self.assert_invariants(bank)

    def test_shrinking_budget_reenforces_and_oversize_records_are_skipped(self):
        bank = budget_bank(1000)
        for t in range(10):
            bank.observe(grid_frame(t))
        bank.set_budget(60)
        self.assert_invariants(bank)
        bank.set_budget(10)
        bank.observe(grid_frame(10, height=16, width=16))
        self.assertEqual(bank.stats()["skipped_oversize"], 1)
        self.assert_invariants(bank)

    def test_long_high_resolution_stream_covers_whole_history(self):
        # Mirrors 1080p: 690-token admitted records, 86K budget, one-hour stream.
        bank = budget_bank(86000, overhead=9, min_frame_tokens=16, store_bytes=1 << 34)
        for t in range(3600):
            bank.observe(grid_frame(t, height=46, width=60, channels=1, layers=0))
        stats = bank.stats()
        self.assertEqual(stats["earliest_timestamp"], 0.0)
        self.assertLessEqual(stats["budget_cost"], 86000)
        self.assertGreaterEqual(stats["frame_tokens_min"], 12)
        self.assertGreater(stats["stored_frames"], 1000)
        self.assert_invariants(bank)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            MemoryConfig(store_budget="context", history_compression="progressive_pool")
        with self.assertRaises(ValueError):
            MemoryConfig(store_budget="context", history_compression="budget_pool", context_tokens=None)
        with self.assertRaises(ValueError):
            MemoryConfig(history_compression="budget_pool", min_frame_tokens=0)
        MemoryConfig(store_budget="context", history_compression="budget_pool", context_tokens=4096)


class ContextBudgetTests(unittest.TestCase):
    def test_budget_subtracts_recent_prompt_reserve_and_margin(self):
        class Adapter:
            def context_limit(self, cap):
                return cap

            def count_input_tokens(self, history, recent, question, direct=False):
                return len(question) + 30 + sum(f.token_count + 5 for f in [*history, *recent])

        budget, overhead = context_history_budget(
            Adapter(), 100, [("q" * 50, {}), ("q" * 80, {"direct": True})], 16, 16, 10000)
        self.assertEqual(overhead, 5 + 2)
        fixed = 80 + 30 + 5
        self.assertEqual(budget, 10000 - 16 - fixed - 16 * (100 + 7) - visual_memory.HISTORY_BUDGET_MARGIN_TOKENS)


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is needed only for synthetic-video tests")
class EndInclusiveSamplerTests(unittest.TestCase):
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

    def sampler(self, include_end, start=0.0, fps=1.0):
        sampler = StreamSampler(self.path, start, fps, include_end=include_end)
        self.addCleanup(sampler.close)
        return sampler

    def test_last_frame_matches_official_end_aligned_reader(self):
        from onestreamer.video import read_recent_frames

        for end in (2.0, 2.02, 3.0):
            with self.subTest(end=end):
                observed = [t for t, _ in self.sampler(True).advance(end)]
                official = [t for t, _ in read_recent_frames(self.path, 0.0, end, 16, 1.0)]
                self.assertAlmostEqual(observed[-1], official[-1])
                self.assertTrue(all(t < end for t in observed))
                legacy = [t for t, _ in self.sampler(False).advance(end)]
                self.assertEqual(len(observed), len(legacy) + (1 if end == int(end) else 0))

    def test_incremental_queries_match_one_shot_and_cutoff_slot_is_not_lost(self):
        incremental = self.sampler(True)
        self.assertEqual(list(incremental.advance(0.0)), [])
        chunks = [[t for t, _ in incremental.advance(end)] for end in (1.0, 1.0, 2.5, 3.9)]
        self.assertEqual(chunks, [[0.0, 0.9], [], [2.0], [3.0]])
        self.assertTrue(all(a < b for a, b in zip(sum(chunks, []), sum(chunks, [])[1:])))


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is needed only for synthetic-video tests")
class BudgetRunnerTests(unittest.TestCase):
    def test_infer_sizes_bank_from_context_and_fills_all_history(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "stream.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (32, 32))
            if not writer.isOpened():
                self.skipTest("This OpenCV build cannot write MJPG test videos")
            try:
                for index in range(240):
                    writer.write(np.full((32, 32, 3), index, dtype=np.uint8))
            finally:
                writer.release()
            template = dict(task="backward_tracking", subtask="EPM", video=video.name, start=0.0,
                            question="Which object?", options={"A": "cup", "B": "plate"},
                            answer="A", answer_text="cup", eval_mode="mcq_letter")
            rows = [dict(template, sample_id=f"q{end}", index=i, end=float(end))
                    for i, end in enumerate((20, 60, 110))]
            job = dict(model="m", model_path="/unused", video_root=str(root))

            class Adapter:
                def __init__(self, engine):
                    self.engine = engine

                def encode(self, images, max_pixels=None, store_on_cpu=True):
                    return [grid_frame(t, channels=2, layers=1) for t, _ in images]

                def context_limit(self, cap):
                    return cap

                def count_input_tokens(self, history, recent, question, direct=False):
                    return 40 + len(question) // 4 + sum(f.token_count + 6 for f in [*history, *recent])

                def select_history(self, candidates, recent, question, direct=False, max_tokens=16,
                                   context_tokens=None):
                    used = self.count_input_tokens(candidates, recent, question, direct)
                    self.engine.fits.append(used + max_tokens <= context_tokens)
                    return list(candidates), {}

                def answer(self, history, recent, question, direct=False, max_tokens=16, context_tokens=None):
                    tokens = sum(f.token_count for f in [*history, *recent])
                    return "A", dict(input_tokens=tokens, visual_tokens=tokens, text_tokens=0)

            engine = SimpleNamespace(device="cpu", fits=[], model=SimpleNamespace(
                config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))))
            engine_module = ModuleType("onestreamer.engine")
            engine_module.OneStreamerEngine = lambda **kwargs: self.fail("Unexpected model load")
            config = MemoryConfig(capacity=100000, store_tokens=100000, store_bytes=1 << 30,
                                  read_mode="fill", context_tokens=2600, history_compression="budget_pool",
                                  store_budget="context", min_frame_tokens=4, stream_include_end=True)
            with patch.dict(sys.modules, {"onestreamer.engine": engine_module}), \
                    patch.object(visual_memory, "NativeFeatureAdapter", Adapter):
                results = list(visual_memory.infer(job, rows, config, engine=engine))
            self.assertTrue(all(engine.fits))
            for row in results:
                self.assertEqual(row["recent_timestamps"][-1], row["end"] - 0.5)
                self.assertEqual(row["history_timestamps"][0], 0.0)
                self.assertEqual(row["history_read_frames"], row["bank"]["stored_frames"])
                self.assertLessEqual(row["bank"]["budget_cost"], row["bank"]["token_budget"])
            self.assertGreater(results[-1]["bank"]["pool_operations"], results[-1]["bank"]["seen"])
            self.assertGreater(results[-1]["history_read_frames"], 50)


if __name__ == "__main__":
    unittest.main()

"""Small native-model end-to-end checks for the opt-in streaming selector."""

from __future__ import annotations

from contextlib import nullcontext, redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import transformers
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from onestreamer import visual_memory
from onestreamer.visual_memory import MemoryConfig, NativeFeatureAdapter
from onestreamer.ovobench import build_question_text
from onestreamer.schema import Sample
import test_native_feature_adapter as native_fixture


def sample_rows():
    template = dict(task="backward_tracking", subtask="EPM", video="test.mp4", start=0.0,
                    question="t4", options={"A": "t5", "B": "t6"}, answer="A")
    return [dict(template, sample_id=f"q{end}", index=index, end=float(end))
            for index, end in enumerate((2, 4, 6))]


class DeterministicSampler:
    """Tiny image source with the same causal schedule as the real decoder."""

    instances = []

    def __init__(self, path, start, fps, include_end=False):
        self.origin, self.fps, self.include_end = start, fps, include_end
        self.observation_time = start
        self.slot, self.closed = 0, False
        self.__class__.instances.append(self)

    def advance(self, end):
        while self.origin + self.slot / self.fps < end:
            timestamp = self.origin + self.slot / self.fps
            self.observation_time = timestamp
            image = Image.new("RGB", (8, 8))
            image.putdata([((x * 19 + self.slot * 17) % 256,
                            (y * 31 + self.slot * 29) % 256,
                            (x * y * 11 + self.slot * 41) % 256)
                           for y in range(8) for x in range(8)])
            self.slot += 1
            yield timestamp, image

    def close(self):
        self.closed = True


class RecordingNativeAdapter(NativeFeatureAdapter):
    """Record boundaries while executing the actual encoder and generation."""

    def encode(self, frames, **kwargs):
        encoded = super().encode(frames, **kwargs)
        self.engine.encoded.update((frame.timestamp, frame) for frame in encoded)
        return encoded

    def answer(self, history, recent, question, **kwargs):
        self.engine.answer_views.append((list(history), list(recent)))
        sparse = any(frame.source_indices is not None for frame in history)
        pixel_guard = (patch.object(self.engine.processor.image_processor, "preprocess",
                                    side_effect=AssertionError("Sparse readout created dummy pixels"))
                       if sparse else nullcontext())
        with pixel_guard:
            return super().answer(history, recent, question, **kwargs)


@unittest.skipUnless(transformers.__version__ == "4.57.6", "Exact supported HF version required")
class VCastNativeRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native_fixture.NativeFeatureRoundtripTests.setUpClass.__func__(cls)

    processor_engine = native_fixture.NativeFeatureRoundtripTests.processor_engine

    def run_stream(self, config):
        engine = self.processor_engine()
        engine.encoded, engine.answer_views = {}, []
        job = dict(model="tiny-native", model_path="/unused", video_root="/unused")
        with patch.object(visual_memory, "StreamSampler", DeterministicSampler), \
                patch.object(visual_memory, "NativeFeatureAdapter", RecordingNativeAdapter), \
                patch.object(visual_memory, "pool_visual_frame", side_effect=AssertionError("Pooling ran")), \
                patch("onestreamer.video.read_recent_frames", side_effect=AssertionError("Recent re-decoded")), \
                patch.object(engine.model.visual, "forward", wraps=engine.model.visual.forward) as visual:
            results = list(visual_memory.infer(job, sample_rows(), config, engine=engine))
        self.assertTrue(DeterministicSampler.instances[-1].closed)
        self.assertEqual(visual.call_count, 6)  # Six observations; cached QA never encodes again.
        self.assertEqual(list(engine.encoded), [float(i) for i in range(6)])
        return results, engine

    def config(self, **changes):
        config = MemoryConfig(history_compression="vcast", recent_frames=2,
                              store_tokens=5, min_frame_tokens=1, capacity=10,
                              read_tokens=5, read_frames=10, context_tokens=256)
        return replace(config, **changes)

    def test_native_stream_preserves_recent_and_selects_original_history_in_both_read_modes(self):
        for mode in ("fixed", "fill"):
            with self.subTest(read_mode=mode):
                config = self.config(read_mode=mode)
                results, engine = self.run_stream(config)
                self.assertEqual([r["recent_timestamps"] for r in results],
                                 [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
                self.assertEqual([r["bank"]["seen"] for r in results], [0, 2, 4])
                self.assertEqual([r["stream_encoded_frames_cumulative"] for r in results], [2, 4, 6])
                self.assertGreater(results[-1]["bank"]["selection_operations"], 0)
                self.assertEqual(results[-1]["history_read_frames"], 4)
                for row, (history, recent) in zip(results, engine.answer_views):
                    self.assertTrue(row["variant"].startswith(visual_memory.VCAST_VERSION))
                    self.assertEqual(row["pooled_input_tokens_cumulative"], 0)
                    self.assertEqual(row["pooled_output_tokens_cumulative"], 0)
                    self.assertLessEqual(row["bank"]["budget_cost"], config.store_tokens)
                    self.assertLessEqual(row["bank"]["stored_bytes"], config.store_bytes)
                    self.assertLessEqual(row["input_tokens"] + row["answer_reserve_tokens"], 256)
                    self.assertEqual(row["visual_tokens"], row["history_visual_tokens"] + 8)
                    self.assertEqual(row["history_timestamps"], sorted(row["history_timestamps"]))
                    for frame in recent:
                        self.assertIs(frame, engine.encoded[frame.timestamp])
                        self.assertIsNone(frame.source_indices)
                        self.assertEqual(frame.token_count, 4)
                    for frame in history:
                        original = engine.encoded[frame.timestamp]
                        self.assertLess(frame.timestamp, recent[0].timestamp)
                        self.assertEqual(frame.grid_thw, original.grid_thw)
                        self.assertEqual(frame.pool_scale, 1)
                        selected = list(frame.source_indices or range(original.token_count))
                        torch.testing.assert_close(frame.features, original.features[selected], rtol=0, atol=0)
                        for actual, full in zip(frame.deepstack, original.deepstack):
                            torch.testing.assert_close(actual, full[selected], rtol=0, atol=0)
                    if mode == "fill":
                        self.assertEqual(row["history_selection_meta"]["selected_input_tokens"], row["input_tokens"])
                self.assertNotIn("get_rope_index", self.model.model.__dict__)
                self.assertNotIn("get_image_features", self.model.model.__dict__)

    def test_context_sized_bank_and_native_readout_both_respect_cap(self):
        adapter = NativeFeatureAdapter(self.processor_engine())
        prompts = [(build_question_text(Sample.from_dict(row)), dict(direct=False)) for row in sample_rows()]
        available, overhead = visual_memory.context_history_budget(adapter, 4, prompts, 2, 16, 256)
        target_budget = overhead * 2 + 5
        cap = 256 - available + target_budget
        self.assertLessEqual(cap, 256)
        config = self.config(store_tokens=10000, store_budget="context", read_mode="fill", context_tokens=cap)
        results, _ = self.run_stream(config)
        for row in results:
            self.assertEqual(row["bank"]["token_budget"], target_budget)
            self.assertEqual(row["bank"]["frame_overhead"], overhead)
            self.assertLessEqual(row["bank"]["budget_cost"], target_budget)
            self.assertLessEqual(row["input_tokens"] + row["answer_reserve_tokens"], cap)
        self.assertGreater(results[-1]["bank"]["selection_operations"], 0)
        self.assertGreater(results[-1]["history_visual_tokens"], 0)

    def test_disabled_history_matches_existing_budget_pool_control(self):
        all_results = []
        for compression in ("budget_pool", "vcast"):
            config = self.config(history_compression=compression, capacity=0, read_mode="fill")
            results, engine = self.run_stream(config)
            self.assertTrue(all(not history for history, _ in engine.answer_views))
            self.assertTrue(all(row["history_visual_tokens"] == 0 for row in results))
            all_results.append(results)
        fields = ("prediction", "parsed_output", "hit", "input_tokens", "visual_tokens", "text_tokens",
                  "recent_timestamps", "recent_image_sizes", "stream_encoded_frames_cumulative")
        for legacy, selected in zip(*all_results):
            for field in fields:
                self.assertEqual(selected[field], legacy[field], field)


class VCastCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "test.mp4").write_bytes(b"Input validation only; no decoding")
        self.samples = self.root / "samples.json"
        self.samples.write_text(json.dumps(sample_rows()[:1]), encoding="utf-8")

    def args(self, name):
        return ["--samples", str(self.samples), "--model-path", "/unused-model",
                "--video-root", str(self.root), "--output", str(self.root / name)]

    def test_dry_run_leaves_existing_defaults_and_is_model_free(self):
        with patch.object(visual_memory, "infer", side_effect=AssertionError("Inference ran")):
            for name, extra in (("legacy", []), ("vcast", ["--history-compression", "vcast"])):
                output = io.StringIO()
                with self.subTest(mode=name), redirect_stdout(output):
                    visual_memory.main([*self.args(name), "--dry-run", *extra])
                metadata = json.loads(output.getvalue())
                self.assertFalse((self.root / name).exists())
                self.assertEqual(metadata["samples"], 1)
                self.assertEqual(metadata["config"]["recent_frames"], 16)
                self.assertEqual(metadata["config"]["read_mode"], "fixed")
                self.assertEqual(metadata["config"]["history_compression"], "resize" if name == "legacy" else "vcast")
                self.assertEqual(metadata["config"]["capacity"], 64 if name == "legacy" else 16384)

    def test_written_manifest_versions_and_hashes_distinguish_the_opt_in_implementation(self):
        def inference_stub(job, rows, config):
            for row in rows:
                yield dict(row, model_alias="test", variant="stub", prediction="A", hit=1,
                           history_visual_tokens=0, observation_seconds_delta=0.0,
                           runtime_seconds=0.0, total_runtime_seconds=0.0)

        for name, extra in (("legacy-run", []), ("vcast-run", ["--history-compression", "vcast"])):
            with self.subTest(mode=name), patch.object(visual_memory, "infer", side_effect=inference_stub), \
                    redirect_stdout(io.StringIO()):
                visual_memory.main([*self.args(name), *extra])
            manifest = json.loads((self.root / name / "manifest.json").read_text())
            source_names = {"visual_memory.py"}
            if name == "vcast-run":
                source_names.update({"vcast_memory.py", "sparse_visual_inputs.py"})
                self.assertEqual(manifest["version"], visual_memory.VCAST_VERSION)
                self.assertEqual(manifest["protocol"]["visual_positions"], "original_merged_grid_coordinates")
                self.assertIn("causal", manifest["protocol"]["compression_variant"])
                self.assertEqual(manifest["protocol"]["history_write"], "on_recent_fifo_eviction_then_budget_select")
            else:
                self.assertEqual(manifest["version"], "visual_memory_v7_context_budget_pool")
                self.assertNotIn("compression_variant", manifest["protocol"])
            self.assertEqual(set(manifest["source_sha256"]), source_names)
            for source in source_names:
                path = Path(visual_memory.__file__).with_name(source)
                self.assertEqual(manifest["source_sha256"][source], hashlib.sha256(path.read_bytes()).hexdigest())
            records = (self.root / name / "predictions.jsonl").read_text().splitlines()
            self.assertEqual(len(records), 1)
            self.assertEqual(json.loads(records[0])["fingerprint"], manifest["fingerprint"])


if __name__ == "__main__":
    unittest.main()

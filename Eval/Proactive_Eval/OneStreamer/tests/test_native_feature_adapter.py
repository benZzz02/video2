"""Real HF Qwen3-VL CPU tests with tiny random weights and no downloads.

Requires torch and transformers==4.57.6. These verify the native-feature
injection interface, not checkpoint accuracy or accelerator performance.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch
import transformers
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onestreamer.visual_memory import (
    NativeFeatureAdapter, VisualFrame, cached_image_features, pool_visual_frame,
)


@unittest.skipUnless(transformers.__version__ == "4.57.6", "Exact supported HF version required")
class NativeFeatureRoundtripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(73)
        config = Qwen3VLConfig(
            vision_config=dict(
                depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
                in_channels=3, patch_size=2, spatial_merge_size=2,
                temporal_patch_size=2, out_hidden_size=32,
                num_position_embeddings=16, deepstack_visual_indexes=[0, 1],
            ),
            text_config=dict(
                vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=3, num_attention_heads=4,
                num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
                rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
                pad_token_id=0, bos_token_id=1, eos_token_id=2,
            ),
            image_token_id=60, video_token_id=61,
            vision_start_token_id=58, vision_end_token_id=59,
        )
        cls.model = Qwen3VLForConditionalGeneration(config).eval()
        cls.engine = SimpleNamespace(model=cls.model, device=torch.device("cpu"))
        # Different image grids exercise per-image splitting, DeepStack offsets,
        # placeholder expansion, and native mRoPE on more than one visual block.
        cls.grid = torch.tensor([[1, 4, 4], [1, 2, 4]], dtype=torch.long)
        cls.pixels = torch.randn(24, 24)
        cls.ids = torch.tensor([[1, 3, 58, 60, 60, 60, 60, 59, 4, 58, 60, 60, 59, 5]])
        with torch.inference_mode():
            final, deepstack = cls.model.get_image_features(cls.pixels, cls.grid)
        cls.frames = []
        offset = 0
        for index, (features, grid) in enumerate(zip(final, cls.grid.tolist())):
            count = features.shape[0]
            cls.frames.append(VisualFrame(
                float(index), tuple(grid), features.detach().clone(),
                tuple(layer[offset:offset + count].detach().clone() for layer in deepstack),
                (grid[2] * 2, grid[1] * 2),
            ))
            offset += count

    def native_inputs(self, ids=None):
        ids = self.ids if ids is None else ids
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                    pixel_values=self.pixels, image_grid_thw=self.grid)

    def cached_inputs(self, ids=None):
        inputs = self.native_inputs(ids)
        # This is the exact empty sentinel used by NativeFeatureAdapter.
        inputs["pixel_values"] = torch.empty(0, dtype=self.model.dtype)
        return inputs

    def processor_engine(self):
        from PIL import ImageOps
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor
        from transformers.video_processing_utils import BaseVideoProcessor

        vocab = {f"t{index}": index for index in range(64)}
        specials = {0: "[UNK]", 58: "<|vision_start|>", 59: "<|vision_end|>",
                    60: "<|image_pad|>", 61: "<|video_pad|>"}
        for index, token in specials.items():
            del vocab[f"t{index}"]
            vocab[token] = index
        backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = Qwen2TokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]", eos_token="t2", pad_token="t1",
            additional_special_tokens=[specials[index] for index in (58, 59, 60, 61)],
        )
        image_processor = Qwen2VLImageProcessor(
            patch_size=2, temporal_patch_size=2, merge_size=2,
            min_pixels=16, max_pixels=256,
        )
        template = (
            "{% for message in messages %}{% for content in message['content'] %}"
            "{% if content['type'] == 'image' %}"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "{% else %} {{ content['text'] }} {% endif %}"
            "{% endfor %}{% endfor %}{% if add_generation_prompt %} t3{% endif %}"
        )
        # The CPU test environment has no torchvision. HF otherwise validates
        # this unused video component against a lazy dummy class. Only skip
        # that constructor check: image preprocessing, tokenizer expansion,
        # chat rendering, and all model calls below are the real HF methods.
        original_validator = Qwen3VLProcessor.check_argument_for_proper_class

        def validate(processor, name, value):
            if name == "video_processor":
                return type(value)
            return original_validator(processor, name, value)

        with patch.object(Qwen3VLProcessor, "check_argument_for_proper_class", validate):
            processor = Qwen3VLProcessor(
                image_processor=image_processor, tokenizer=tokenizer,
                video_processor=BaseVideoProcessor(), chat_template=template,
            )

        def pad(image):
            width, height = image.size
            return ImageOps.expand(image, border=(0, 0, (-width) % 4, (-height) % 4))

        return SimpleNamespace(
            model=self.model, device=torch.device("cpu"), processor=processor,
            _pad_to_patch_grid=pad, _strip_end_tokens=lambda text: text.strip(),
        )

    def test_prefill_logits_and_rope_match_pixels_without_running_visual(self):
        with torch.inference_mode():
            native = self.model(**self.native_inputs(), use_cache=False)
            expected_logits = native.logits.clone()
            expected_rope = native.rope_deltas.clone()
            with patch.object(self.model.visual, "forward", side_effect=AssertionError("ViT ran")):
                with cached_image_features(self.engine, self.frames) as state:
                    cached = self.model(**self.cached_inputs(), use_cache=False)
        self.assertEqual(state["calls"], 1)
        torch.testing.assert_close(cached.logits, expected_logits, rtol=0, atol=0)
        torch.testing.assert_close(cached.rope_deltas, expected_rope, rtol=0, atol=0)
        self.assertNotIn("get_image_features", self.model.model.__dict__)

    def test_generate_tokens_and_step_scores_match_across_new_queries(self):
        generation = dict(
            max_new_tokens=4, min_new_tokens=4, do_sample=False, use_cache=True,
            num_beams=1, num_return_sequences=1, pad_token_id=0,
            return_dict_in_generate=True, output_scores=True,
        )
        # A second, different-length query confirms that old rope_deltas are not
        # accidentally reused as if this were a continuation of the first QA.
        for ids in (self.ids, torch.cat([self.ids, torch.tensor([[6, 7, 8]])], dim=1)):
            with self.subTest(prompt_length=ids.shape[1]), torch.inference_mode():
                native = self.model.generate(**self.native_inputs(ids), **generation)
                with patch.object(self.model.visual, "forward", side_effect=AssertionError("ViT ran")):
                    with cached_image_features(self.engine, self.frames) as state:
                        cached = self.model.generate(**self.cached_inputs(ids), **generation)
                self.assertEqual(state["calls"], 1)
                self.assertTrue(torch.equal(cached.sequences, native.sequences))
                self.assertEqual(len(cached.scores), 4)
                for actual, expected in zip(cached.scores, native.scores):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_deepstack_is_not_optional_for_equivalent_logits(self):
        without_deepstack = [replace(
            frame, deepstack=tuple(torch.zeros_like(layer) for layer in frame.deepstack)
        ) for frame in self.frames]
        with torch.inference_mode():
            native = self.model(**self.native_inputs(), use_cache=False).logits
            with cached_image_features(self.engine, without_deepstack):
                changed = self.model(**self.cached_inputs(), use_cache=False).logits
        self.assertGreater((native - changed).abs().max().item(), 1e-5)

    def test_native_placeholder_validation_rejects_wrong_token_count(self):
        inputs = self.cached_inputs()
        inputs["input_ids"] = inputs["input_ids"].clone()
        inputs["input_ids"][0, 3] = 9
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, "features and image tokens"):
            with cached_image_features(self.engine, self.frames):
                self.model(**inputs, use_cache=False)
        self.assertNotIn("get_image_features", self.model.model.__dict__)

    def test_real_processor_encode_prepare_and_answer_with_image_only_chat_items(self):
        from PIL import Image

        engine = self.processor_engine()
        adapter = NativeFeatureAdapter(engine)
        images = [Image.new("RGB", (7, 5), (180, 20, 70)),
                  Image.new("RGB", (8, 4), (30, 200, 10))]
        with patch.object(self.model.visual, "forward", wraps=self.model.visual.forward) as visual:
            frames = adapter.encode(list(enumerate(images)))
        self.assertEqual(visual.call_count, 1)
        self.assertEqual([frame.size for frame in frames], [(8, 8), (8, 4)])
        chat = [dict(role="user", content=[
            dict(type="text", text="t4"), dict(type="image"),
            dict(type="image"), dict(type="text", text="t5"),
        ])]
        inputs, metrics = adapter.prepare_inputs(chat, frames)
        self.assertEqual(inputs["pixel_values"].numel(), 0)
        self.assertEqual(metrics["visual_tokens"], 6)
        prompt = engine.processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        native_inputs = engine.processor(
            text=[prompt], images=[engine._pad_to_patch_grid(image) for image in images],
            padding=True, do_resize=False, return_tensors="pt",
        )
        self.assertTrue(torch.equal(inputs["input_ids"], native_inputs["input_ids"]))
        self.assertTrue(torch.equal(inputs["image_grid_thw"], native_inputs["image_grid_thw"]))
        with torch.inference_mode():
            native_logits = self.model(**native_inputs, use_cache=False).logits
            with patch.object(self.model.visual, "forward", side_effect=AssertionError("ViT ran")):
                with cached_image_features(engine, frames):
                    cached_logits = self.model(**inputs, use_cache=False).logits
                answer, answer_metrics = adapter.answer(frames[:1], frames[1:], "t5", max_tokens=3)
        torch.testing.assert_close(cached_logits, native_logits, rtol=0, atol=0)
        self.assertIsInstance(answer, str)
        self.assertEqual(answer_metrics["visual_tokens"], 6)
        self.assertNotIn("get_image_features", self.model.model.__dict__)

    def test_real_processor_history_resize_and_deepstack_match_native_encoding(self):
        from PIL import Image

        engine = self.processor_engine()
        adapter = NativeFeatureAdapter(engine)
        image = Image.new("RGB", (64, 64), (20, 150, 80))
        frames = adapter.encode([(0.0, image)], max_pixels=1024)
        self.assertEqual(frames[0].size, (32, 32))
        self.assertEqual(frames[0].grid_thw, (1, 16, 16))
        self.assertEqual(frames[0].token_count, 64)
        pixels = engine.processor.image_processor(
            images=[image], do_resize=True, min_pixels=1024, max_pixels=1024, return_tensors="pt",
        )
        with torch.inference_mode():
            final, deepstack = self.model.get_image_features(pixels["pixel_values"], pixels["image_grid_thw"])
        torch.testing.assert_close(frames[0].features, final[0], rtol=0, atol=0)
        for cached, native in zip(frames[0].deepstack, deepstack):
            torch.testing.assert_close(cached, native, rtol=0, atol=0)
        inputs, _ = adapter.prepare_inputs(
            [dict(role="user", content=[dict(type="image"), dict(type="text", text="t5")])], frames,
        )
        self.assertTrue(torch.equal(inputs["image_grid_thw"], pixels["image_grid_thw"]))

    def test_budget_count_matches_real_processor_for_empty_and_variable_history(self):
        engine = self.processor_engine()
        adapter = NativeFeatureAdapter(engine)
        recent = [replace(self.frames[1], timestamp=20.125)]
        histories = ([], [replace(self.frames[0], timestamp=0.0)],
                     [replace(self.frames[0], timestamp=0.0),
                      replace(self.frames[1], timestamp=12.375)])
        for history in histories:
            for direct in (False, True):
                with self.subTest(history=len(history), direct=direct):
                    # Token accounting must not preprocess any dummy images.
                    with patch.object(engine.processor.image_processor, "preprocess",
                                      side_effect=AssertionError("Budget count processed pixels")):
                        count = adapter.count_input_tokens(history, recent, "t5 t6", direct)
                    inputs, metrics = adapter.prepare_inputs(
                        adapter.build_chat(history, recent, "t5 t6", direct), history + recent,
                    )
                    self.assertEqual(count, inputs["input_ids"].shape[-1])
                    self.assertEqual(count, metrics["input_tokens"])
        self.assertEqual(adapter.context_limit(), 256)
        self.assertEqual(adapter.context_limit(200), 200)
        self.assertEqual(adapter.context_limit(1000), 256)

    def test_fill_keeps_recent16_and_fits_real_prepared_context_with_answer_reserve(self):
        from PIL import Image

        engine = self.processor_engine()
        adapter = NativeFeatureAdapter(engine)
        recent = [replace(self.frames[1], timestamp=100 + index) for index in range(16)]
        # The first historical candidate is too large; later small candidates
        # must still be considered. All grids represent complete real images.
        large = adapter.encode([(0.0, Image.new("RGB", (32, 32)))])[0]
        candidates = [large] + [replace(self.frames[0], timestamp=index) for index in range(1, 8)]
        reserve = 4
        cap = adapter.count_input_tokens(candidates[1:3], recent, "t5") + reserve
        selected, budget = adapter.select_history(
            candidates, recent, "t5", max_tokens=reserve, context_tokens=cap,
        )
        self.assertEqual(len(selected), 2)
        self.assertNotIn(0.0, [frame.timestamp for frame in selected])
        self.assertEqual(len(recent), 16)
        inputs, metrics = adapter.prepare_inputs(
            adapter.build_chat(selected, recent, "t5"), selected + recent,
        )
        self.assertLessEqual(inputs["input_ids"].shape[-1] + reserve, cap)
        self.assertEqual(budget["selected_input_tokens"], metrics["input_tokens"])
        self.assertEqual(budget["remaining_input_tokens"], cap - reserve - metrics["input_tokens"])
        self.assertEqual(budget["answer_reserve_tokens"], reserve)
        self.assertEqual(budget["eligible_history_frames"], len(candidates))

    def test_recent_overflow_and_final_processor_budget_drift_fail_before_generation(self):
        engine = self.processor_engine()
        adapter = NativeFeatureAdapter(engine)
        recent = [replace(self.frames[1], timestamp=100 + index) for index in range(16)]
        reserve = 4
        base = adapter.count_input_tokens([], recent, "t5")
        with patch.object(self.model, "generate") as generate:
            with self.assertRaisesRegex(ValueError, "Recent context/question"):
                adapter.select_history([], recent, "t5", max_tokens=reserve,
                                       context_tokens=base + reserve - 1)
            with self.assertRaisesRegex(ValueError, "exceeds context limit"):
                adapter.answer([], recent, "t5", max_tokens=reserve,
                               context_tokens=base + reserve - 1)
            generate.assert_not_called()

        original_prepare = adapter.prepare_inputs

        def prepare_with_drift(chat, frames):
            inputs, metrics = original_prepare(chat, frames)
            return inputs, dict(metrics, input_tokens=metrics["input_tokens"] + 1)

        with patch.object(adapter, "prepare_inputs", side_effect=prepare_with_drift):
            with patch.object(self.model, "generate") as generate:
                with self.assertRaisesRegex(RuntimeError, "length differs"):
                    adapter.answer([], recent, "t5", max_tokens=reserve, context_tokens=256)
                generate.assert_not_called()
        self.assertNotIn("get_image_features", self.model.model.__dict__)

    def test_pool_stride_one_keeps_native_logits_and_generated_tokens_identical(self):
        identity = [pool_visual_frame(frame, stride=1, merge_size=2) for frame in self.frames]
        for actual, original in zip(identity, self.frames):
            self.assertEqual(actual.grid_thw, original.grid_thw)
            self.assertEqual(actual.size, original.size)
            for copied, source in zip((actual.features, *actual.deepstack),
                                      (original.features, *original.deepstack)):
                self.assertNotEqual(copied.data_ptr(), source.data_ptr())
                torch.testing.assert_close(copied, source, rtol=0, atol=0)
        generation = dict(max_new_tokens=3, min_new_tokens=3, do_sample=False,
                          use_cache=True, num_beams=1, num_return_sequences=1, pad_token_id=0)
        with torch.inference_mode():
            native_logits = self.model(**self.native_inputs(), use_cache=False).logits
            native_tokens = self.model.generate(**self.native_inputs(), **generation)
            with patch.object(self.model.visual, "forward", side_effect=AssertionError("ViT ran")):
                with cached_image_features(self.engine, identity):
                    cached_logits = self.model(**self.cached_inputs(), use_cache=False).logits
                with cached_image_features(self.engine, identity):
                    cached_tokens = self.model.generate(**self.cached_inputs(), **generation)
        torch.testing.assert_close(cached_logits, native_logits, rtol=0, atol=0)
        self.assertTrue(torch.equal(cached_tokens, native_tokens))

    def test_pooled_odd_grid_is_readable_with_aligned_deepstack_without_reencoding(self):
        from PIL import Image

        engine = self.processor_engine()
        adapter = NativeFeatureAdapter(engine)
        # With 2-pixel patches and a merge factor of 2 this is a 3x5 merged
        # spatial grid, exercising incomplete windows on both right/bottom edges.
        native, recent = adapter.encode([
            (0.0, Image.new("RGB", (20, 12), (180, 20, 70))),
            (10.0, Image.new("RGB", (8, 8), (30, 200, 10))),
        ])
        self.assertEqual(native.grid_thw, (1, 6, 10))
        for stride, expected_grid, expected_size, expected_tokens in (
                (2, (1, 4, 6), (12, 8), 6),
                (4, (1, 2, 4), (8, 4), 2)):
            with self.subTest(stride=stride):
                pooled = pool_visual_frame(native, stride=stride, merge_size=2)
                self.assertEqual(pooled.grid_thw, expected_grid)
                self.assertEqual(pooled.size, expected_size)
                self.assertEqual(pooled.source_size, (20, 12))
                self.assertEqual(pooled.token_count, expected_tokens)
                self.assertLess(pooled.nbytes, native.nbytes)
                # Independent per-cell reference verifies edge means and the
                # same row-major grouping in final and each DeepStack branch.
                for result, source in zip((pooled.features, *pooled.deepstack),
                                          (native.features, *native.deepstack)):
                    spatial = source.reshape(3, 5, -1)
                    means = [spatial[y:y + stride, x:x + stride].mean(dim=(0, 1))
                             for y in range(0, 3, stride) for x in range(0, 5, stride)]
                    torch.testing.assert_close(result, torch.stack(means), rtol=1e-6, atol=1e-7)
                    self.assertEqual(result.shape, (expected_tokens, 32))

                chat = adapter.build_chat([pooled], [recent], "t5")
                inputs, metrics = adapter.prepare_inputs(chat, [pooled, recent])
                self.assertEqual(metrics["visual_tokens"], expected_tokens + recent.token_count)
                self.assertEqual(adapter.count_input_tokens([pooled], [recent], "t5"),
                                 inputs["input_ids"].shape[-1])
                self.assertEqual(inputs["image_grid_thw"][0].tolist(), list(expected_grid))
                with torch.inference_mode():
                    with patch.object(self.model.visual, "forward", side_effect=AssertionError("ViT ran")):
                        with cached_image_features(engine, [pooled, recent]) as state:
                            logits = self.model(**inputs, use_cache=False).logits
                        self.assertEqual(state["calls"], 1)
                        self.assertTrue(torch.isfinite(logits).all().item())
                        answer, answer_metrics = adapter.answer([pooled], [recent], "t5", max_tokens=3)
                self.assertIsInstance(answer, str)
                self.assertEqual(answer_metrics["visual_tokens"], metrics["visual_tokens"])
                self.assertNotIn("get_image_features", self.model.model.__dict__)

    def test_repeated_odd_grid_pooling_preserves_original_area_weights_and_is_readable(self):
        from PIL import Image

        engine = self.processor_engine()
        adapter = NativeFeatureAdapter(engine)
        native, recent = adapter.encode([
            (0.0, Image.new("RGB", (20, 12), (180, 20, 70))),
            (10.0, Image.new("RGB", (8, 8), (30, 200, 10))),
        ])
        current = native
        with torch.inference_mode():
            with patch.object(self.model.visual, "forward", side_effect=AssertionError("ViT ran")):
                for scale, expected_grid, expected_tokens in (
                        (2, (1, 4, 6), 6), (4, (1, 2, 4), 2), (8, (1, 2, 2), 1)):
                    with self.subTest(cumulative_scale=scale):
                        current = pool_visual_frame(current, stride=2, merge_size=2)
                        self.assertEqual(current.pool_source_shape, (3, 5))
                        self.assertEqual(current.pool_scale, scale)
                        self.assertEqual(current.grid_thw, expected_grid)
                        self.assertEqual(current.token_count, expected_tokens)
                        self.assertEqual(current.source_size, (20, 12))
                        self.assertEqual(current.timestamp, native.timestamp)
                        # Compare against original cells, never against an
                        # average of the preceding step's unequal-area cells.
                        for result, original in zip((current.features, *current.deepstack),
                                                    (native.features, *native.deepstack)):
                            spatial = original.reshape(3, 5, -1)
                            means = [spatial[y:y + scale, x:x + scale].mean(dim=(0, 1))
                                     for y in range(0, 3, scale) for x in range(0, 5, scale)]
                            torch.testing.assert_close(result, torch.stack(means),
                                                       rtol=2e-5, atol=1e-7)
                        inputs, metrics = adapter.prepare_inputs(
                            adapter.build_chat([current], [recent], "t5"), [current, recent],
                        )
                        self.assertEqual(metrics["visual_tokens"], expected_tokens + recent.token_count)
                        with cached_image_features(engine, [current, recent]) as state:
                            logits = self.model(**inputs, use_cache=False).logits
                        self.assertEqual(state["calls"], 1)
                        self.assertTrue(torch.isfinite(logits).all().item())
                        answer, answer_metrics = adapter.answer([current], [recent], "t5", max_tokens=2)
                        self.assertIsInstance(answer, str)
                        self.assertEqual(answer_metrics["visual_tokens"], metrics["visual_tokens"])
        for result, original in zip((current.features, *current.deepstack),
                                    (native.features, *native.deepstack)):
            torch.testing.assert_close(result[0], original.mean(dim=0), rtol=2e-5, atol=1e-7)
        self.assertNotIn("get_image_features", self.model.model.__dict__)

    def test_pooled_history_readback_accepts_recent_counts_on_both_sides_of_sixteen(self):
        engine = self.processor_engine()
        adapter = NativeFeatureAdapter(engine)
        pooled = pool_visual_frame(self.frames[0], stride=2, merge_size=2)
        candidates = [replace(pooled, timestamp=float(index)) for index in range(3)]
        for recent_count in (1, 3, 17):
            with self.subTest(recent_count=recent_count):
                recent = [replace(self.frames[1], timestamp=100.0 + index)
                          for index in range(recent_count)]
                history, budget = adapter.select_history(
                    candidates, recent, "t5", max_tokens=2, context_tokens=256,
                )
                self.assertEqual(len(history), 3)
                self.assertEqual(len(recent), recent_count)
                inputs, metrics = adapter.prepare_inputs(
                    adapter.build_chat(history, recent, "t5"), history + recent,
                )
                self.assertEqual(inputs["image_grid_thw"].shape[0], 3 + recent_count)
                self.assertEqual(metrics["visual_tokens"], 3 + 2 * recent_count)
                self.assertEqual(metrics["input_tokens"], budget["selected_input_tokens"])
                self.assertLessEqual(metrics["input_tokens"] + 2, 256)
                with patch.object(self.model.visual, "forward", side_effect=AssertionError("ViT ran")):
                    answer, answer_metrics = adapter.answer(
                        history, recent, "t5", max_tokens=2, context_tokens=256,
                    )
                self.assertIsInstance(answer, str)
                self.assertEqual(answer_metrics["visual_tokens"], metrics["visual_tokens"])


if __name__ == "__main__":
    unittest.main()

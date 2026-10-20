"""Native Qwen3-VL checks for retained source coordinates and sparse decoding."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from onestreamer.sparse_visual_inputs import prepare_sparse_inputs, sparse_visual_positions
from onestreamer.visual_memory import NativeFeatureAdapter, cached_image_features
import test_native_feature_adapter as native_fixture


@unittest.skipUnless(transformers.__version__ == "4.57.6", "Exact supported HF version required")
class SparseVisualInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native_fixture.NativeFeatureRoundtripTests.setUpClass.__func__(cls)

    native_inputs = native_fixture.NativeFeatureRoundtripTests.native_inputs
    cached_inputs = native_fixture.NativeFeatureRoundtripTests.cached_inputs
    processor_engine = native_fixture.NativeFeatureRoundtripTests.processor_engine

    def selected_frames(self, indices=(0, 3)):
        first = self.frames[0]
        selected = torch.tensor(indices, dtype=torch.long)
        return [replace(first, source_indices=tuple(indices),
                        features=first.features.index_select(0, selected),
                        deepstack=tuple(layer.index_select(0, selected) for layer in first.deepstack)),
                self.frames[1]]

    def sparse_inputs(self, frames, ids=None):
        inputs = self.cached_inputs(ids)
        prepare_sparse_inputs(self.engine, inputs, frames)
        return inputs

    def test_dense_path_is_an_exact_noop(self):
        inputs = self.cached_inputs()
        original = dict(inputs)
        self.assertIs(prepare_sparse_inputs(self.engine, inputs, self.frames), inputs)
        with sparse_visual_positions(self.engine, self.frames):
            self.assertNotIn("get_rope_index", self.model.model.__dict__)
        for name, value in original.items():
            self.assertIs(inputs[name], value)

    def test_original_coordinates_and_delta_survive_nonuniform_selection(self):
        frames = self.selected_frames()
        original_positions, original_delta = self.model.model.get_rope_index(
            self.ids, self.grid, attention_mask=torch.ones_like(self.ids),
        )
        keep = torch.tensor([0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13])
        inputs = self.sparse_inputs(frames)
        torch.testing.assert_close(inputs["input_ids"], self.ids[:, keep], rtol=0, atol=0)
        torch.testing.assert_close(inputs["image_grid_thw"], self.grid, rtol=0, atol=0)
        previous_delta = self.model.model.rope_deltas
        with sparse_visual_positions(self.engine, frames):
            positions, delta = self.model.model.get_rope_index(
                inputs["input_ids"], self.grid, attention_mask=inputs["attention_mask"],
            )
        torch.testing.assert_close(positions, original_positions[:, :, keep], rtol=0, atol=0)
        torch.testing.assert_close(delta, original_delta + 2, rtol=0, atol=0)
        self.assertIs(self.model.model.rope_deltas, previous_delta)
        self.assertNotIn("get_rope_index", self.model.model.__dict__)

    def test_all_token_identity_matches_native_multistep_generation(self):
        frames = [replace(f, source_indices=tuple(range(f.token_count))) for f in self.frames]
        generation = dict(max_new_tokens=4, do_sample=False, use_cache=True,
                          num_beams=1, pad_token_id=0, eos_token_id=None,
                          return_dict_in_generate=True, output_scores=True)
        for ids in (self.ids, torch.cat([self.ids, torch.tensor([[6, 7, 8]])], dim=-1)):
            with self.subTest(prompt_length=ids.shape[-1]), torch.inference_mode():
                expected = self.model.generate(**self.native_inputs(ids), **generation)
                inputs = self.sparse_inputs(frames, ids)
                with patch.object(self.model.visual, "forward", side_effect=AssertionError("ViT ran")):
                    with cached_image_features(self.engine, frames) as state:
                        actual = self.model.generate(**inputs, **generation)
                self.assertEqual(state["calls"], 1)
                torch.testing.assert_close(actual.sequences, expected.sequences, rtol=0, atol=0)
                self.assertEqual(len(actual.scores), 4)
                for score, reference in zip(actual.scores, expected.scores):
                    torch.testing.assert_close(score, reference, rtol=0, atol=0)

    def test_sparse_generation_matches_explicit_positions_on_every_decode_step(self):
        frames = self.selected_frames()
        generation = dict(max_new_tokens=4, do_sample=False, use_cache=True,
                          num_beams=1, pad_token_id=0, eos_token_id=None,
                          return_dict_in_generate=True, output_scores=True)
        # Each query starts from a fresh cache; changed text length must not reuse
        # the previous query's mRoPE delta.
        for ids in (self.ids, torch.cat([self.ids, torch.tensor([[8, 9, 10]])], dim=-1)):
            with self.subTest(prompt_length=ids.shape[-1]), torch.inference_mode():
                dense_positions, _ = self.model.model.get_rope_index(
                    ids, self.grid, attention_mask=torch.ones_like(ids),
                )
                keep = torch.tensor([i for i in range(ids.shape[-1]) if i not in (4, 5)])
                original_positions = dense_positions.index_select(-1, keep)
                inputs = self.sparse_inputs(frames, ids)
                with cached_image_features(self.engine, frames):
                    reference = self.model(
                        **inputs, position_ids=original_positions, use_cache=True, logits_to_keep=1,
                        cache_position=torch.arange(inputs["input_ids"].shape[-1]),
                    )
                reference_scores = [reference.logits[:, -1, :]]
                cache = reference.past_key_values
                next_position = int(dense_positions.max()) + 1
                prompt_length = inputs["input_ids"].shape[-1]
                for step in range(3):
                    previous_token = reference_scores[-1].argmax(-1, keepdim=True)
                    reference = self.model(
                        input_ids=previous_token,
                        attention_mask=torch.ones(1, prompt_length + step + 1, dtype=torch.long),
                        position_ids=torch.full((3, 1, 1), next_position + step, dtype=torch.long),
                        cache_position=torch.tensor([prompt_length + step]),
                        past_key_values=cache, use_cache=True, logits_to_keep=1,
                    )
                    cache = reference.past_key_values
                    reference_scores.append(reference.logits[:, -1, :])
                with patch.object(self.model.visual, "forward", side_effect=AssertionError("ViT ran")):
                    with cached_image_features(self.engine, frames):
                        actual = self.model.generate(**inputs, **generation)
                self.assertEqual(len(actual.scores), 4)
                for score, expected in zip(actual.scores, reference_scores):
                    torch.testing.assert_close(score, expected, rtol=0, atol=0)

    def test_real_processor_preserves_recent_and_selects_every_deepstack_layer(self):
        engine = self.processor_engine()
        adapter = NativeFeatureAdapter(engine)
        frames = self.selected_frames()
        chat = [dict(role="user", content=[dict(type="image"), dict(type="text", text="t4"),
                                           dict(type="image"), dict(type="text", text="t5")])]
        inputs, metrics = adapter.prepare_inputs(chat, frames)
        self.assertEqual(metrics["visual_tokens"], 4)
        self.assertEqual(int((inputs["input_ids"] == 60).sum()), 4)
        with cached_image_features(engine, frames):
            final, deepstack = self.model.model.get_image_features(inputs["pixel_values"], self.grid)
        torch.testing.assert_close(final[0], self.frames[0].features[[0, 3]], rtol=0, atol=0)
        torch.testing.assert_close(final[1], self.frames[1].features, rtol=0, atol=0)
        for index, layer in enumerate(deepstack):
            expected = torch.cat([self.frames[0].deepstack[index][[0, 3]],
                                  self.frames[1].deepstack[index]], dim=0)
            torch.testing.assert_close(layer, expected, rtol=0, atol=0)

    def test_text_only_sparse_preparation_matches_native_expansion_without_pixels(self):
        engine = self.processor_engine()
        adapter = NativeFeatureAdapter(engine)
        chat = [dict(role="user", content=[dict(type="text", text="t7"), dict(type="image"),
                                           dict(type="image"), dict(type="text", text="t5")])]
        dense, dense_metrics = adapter.prepare_inputs(chat, self.frames)
        identity = [replace(f, source_indices=tuple(range(f.token_count))) for f in self.frames]
        with patch.object(engine.processor.image_processor, "preprocess", side_effect=AssertionError("Dummy pixels allocated")):
            all_tokens, _ = adapter.prepare_inputs(chat, identity)
            sparse, sparse_metrics = adapter.prepare_inputs(chat, self.selected_frames())
        for key in ("input_ids", "attention_mask", "image_grid_thw"):
            torch.testing.assert_close(all_tokens[key], dense[key], rtol=0, atol=0)
        expected = prepare_sparse_inputs(engine, dense, self.selected_frames())
        for key in ("input_ids", "attention_mask", "image_grid_thw"):
            torch.testing.assert_close(sparse[key], expected[key], rtol=0, atol=0)
        self.assertEqual(sparse_metrics["input_tokens"], dense_metrics["input_tokens"] - 2)
        self.assertEqual(sparse_metrics["input_tokens"], adapter._count_chat_tokens(chat, self.selected_frames()))

    def test_invalid_indices_and_masked_visuals_are_rejected(self):
        for indices in ((0, 0), (3, 0), (-1, 0), (0, 4), (False, 3)):
            frames = [replace(self.frames[0], features=self.frames[0].features[:2],
                              source_indices=indices), self.frames[1]]
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, "source indices"):
                prepare_sparse_inputs(self.engine, self.cached_inputs(), frames)
        frames = self.selected_frames()
        inputs = self.cached_inputs()
        inputs["attention_mask"][0, 3] = 0
        with self.assertRaisesRegex(ValueError, "masked or padded"):
            prepare_sparse_inputs(self.engine, inputs, frames)

    def test_scoped_method_and_rope_delta_are_restored_on_failure(self):
        frames = self.selected_frames()
        previous = torch.tensor([[17]])
        self.model.model.rope_deltas = previous
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with sparse_visual_positions(self.engine, frames):
                self.model.model.rope_deltas = torch.tensor([[29]])
                raise RuntimeError("intentional")
        self.assertIs(self.model.model.rope_deltas, previous)
        self.assertNotIn("get_rope_index", self.model.model.__dict__)


if __name__ == "__main__":
    unittest.main()

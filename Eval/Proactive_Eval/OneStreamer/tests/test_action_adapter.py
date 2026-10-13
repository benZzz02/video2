"""Real tiny-Qwen3-VL tests for action protocol and final-answer isolation."""

from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import torch
import transformers

import test_native_feature_adapter as native
from onestreamer.visual_memory import MemoryConfig, NativeFeatureAdapter, cached_image_features


@unittest.skipUnless(transformers.__version__ == "4.57.6", "Exact supported HF version required")
class ActionAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native.NativeFeatureRoundtripTests.setUpClass()
        cls.fixture = native.NativeFeatureRoundtripTests()

    def setUp(self):
        self.engine = self.fixture.processor_engine(native_states=True, read_actions=True)
        self.adapter = NativeFeatureAdapter(self.engine)
        self.history = [self.fixture.frames[0]]
        self.recent = [replace(self.fixture.frames[1], timestamp=20.0)]
        self.status = dict(query_time=20.7, stream_start=0.0, recent_start=20.0, recent_end=20.0,
                           retained_history_records=3, selected_history_records=1, unread_history_records=2,
                           unread_time_bins=[dict(start=1.0, end=19.0, unread_records=2)])
        context = patch.object(self.engine.model.config.text_config, "max_position_embeddings", 4096)
        context.start()
        self.addCleanup(context.stop)

    def capture_forward(self, operation):
        calls = []

        def capture(module, args, output):
            calls.append((output.logits.detach().clone(), output.past_key_values))

        handle = self.engine.model.register_forward_hook(capture)
        try:
            return operation(), calls
        finally:
            handle.remove()

    def test_action_input_preserves_evidence_and_counts_status_without_answer_prefix(self):
        question = "How many events occurred? Output only an integer."
        for direct in (False, True):
            final = self.adapter.build_chat(self.history, self.recent, question, direct)
            chat = self.adapter.build_read_action_chat(self.history, self.recent, question, self.status, direct)
            self.assertEqual(chat[2]["content"][:-2], final[1]["content"][:-1])
            self.assertIn(question, chat[1]["content"][0]["text"])
            payload = chat[2]["content"][-2]["text"].split("\n")[1]
            self.assertEqual(json.loads(payload), self.status)
            text = " ".join(c.get("text", "") for message in chat for c in message["content"])
            self.assertNotIn("</Response>", text)
            self.assertIn("later final", text)
            self.assertEqual(final, self.adapter.build_chat(self.history, self.recent, question, direct))
            inputs, _ = self.adapter.prepare_inputs(chat, self.history + self.recent)
            self.assertEqual(inputs["input_ids"].shape[-1], self.adapter.count_read_action_input_tokens(
                self.history, self.recent, question, self.status, direct))

    def test_raw_argmax_distinguishes_actions_from_native_state_or_other_output(self):
        for predicted_id, action in ((53, "STOP"), (54, "READ_MORE"), (55, None), (4, None)):
            with self.subTest(predicted_id=predicted_id):
                def head(hidden):
                    logits = torch.zeros((*hidden.shape[:-1], 64), device=hidden.device)
                    logits[..., predicted_id] = 3.0
                    return logits

                with patch.object(self.engine.model.lm_head, "forward", side_effect=head), \
                     patch.object(self.engine.model.visual, "forward", side_effect=AssertionError("ViT reran")):
                    result, calls = self.capture_forward(lambda: self.adapter.probe_read_action(
                        self.history, self.recent, "t5", self.status))
                self.assertEqual(result["action"], action)
                self.assertEqual(result["predicted_token_id"], predicted_id)
                self.assertEqual(set(result["action_probabilities"]), {"STOP", "READ_MORE"})
                self.assertLess(result["action_probability_mass"], 1.0)
                expected = torch.tensor([3.0] + [0.0] * 63).softmax(0)[0].item()
                self.assertAlmostEqual(result["predicted_probability"], expected)
                self.assertEqual(len(calls), 1)
                self.assertIsNone(calls[0][1])
                self.assertNotIn("get_image_features", self.engine.model.model.__dict__)

    def test_rejects_missing_special_and_multitoken_action_labels(self):
        with self.assertRaisesRegex(ValueError, "ordinary atomic"):
            NativeFeatureAdapter(self.fixture.processor_engine()).read_action_token_ids()
        tokenizer = self.engine.processor.tokenizer
        original = tokenizer.encode
        with patch.object(tokenizer, "encode", side_effect=lambda text, **kw: [53, 54] if text == "A" else original(text, **kw)):
            with self.assertRaisesRegex(ValueError, "ordinary atomic"):
                self.adapter.read_action_token_ids()
        tokenizer.add_special_tokens({"additional_special_tokens": ["A"]})
        with self.assertRaisesRegex(ValueError, "ordinary atomic"):
            self.adapter.read_action_token_ids()

    def test_metadata_overflow_and_nonfinite_logits_fail_explicitly(self):
        status = dict(self.status, unread_time_bins=[dict(start=i, end=i + 1, unread_records=1) for i in range(20)])
        small = self.adapter.count_read_action_input_tokens(self.history, self.recent, "t5", self.status)
        large = self.adapter.count_read_action_input_tokens(self.history, self.recent, "t5", status)
        self.assertGreater(large, small)
        with patch.object(self.engine.model, "forward", side_effect=AssertionError("Model must not run")):
            with self.assertRaisesRegex(ValueError, "exceeds context limit"):
                self.adapter.probe_read_action(self.history, self.recent, "t5", status, context_tokens=large)

        def head(hidden):
            return torch.full((*hidden.shape[:-1], 64), float("nan"), device=hidden.device)

        with patch.object(self.engine.model.lm_head, "forward", side_effect=head):
            with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                self.adapter.probe_read_action(self.history, self.recent, "t5", self.status)
        self.assertNotIn("get_image_features", self.engine.model.model.__dict__)

    def test_probe_logits_match_generation_after_prior_rope_state(self):
        frames = self.history + self.recent
        chat = self.adapter.build_read_action_chat(self.history, self.recent, "t5", self.status)
        inputs, _ = self.adapter.prepare_inputs(chat, frames)
        self.engine.model.model.rope_deltas = torch.tensor([[999999]])
        result, calls = self.capture_forward(lambda: self.adapter.probe_read_action(
            self.history, self.recent, "t5", self.status))
        probe_rope = self.engine.model.model.rope_deltas.clone()
        with cached_image_features(self.engine, frames), torch.inference_mode():
            generated = self.engine.model.generate(
                **inputs, max_new_tokens=1, do_sample=False, use_cache=True,
                num_beams=1, num_return_sequences=1,
                pad_token_id=self.engine.processor.tokenizer.pad_token_id,
                return_dict_in_generate=True, output_logits=True)
        torch.testing.assert_close(calls[0][0][:, -1].float(), generated.logits[0].float(), rtol=0, atol=0)
        torch.testing.assert_close(probe_rope, self.engine.model.model.rope_deltas, rtol=0, atol=0)
        self.assertEqual(result["predicted_token_id"], int(generated.sequences[0, -1]))

    def test_interleaved_action_probes_preserve_final_answer_logits_and_features(self):
        def answer():
            return self.adapter.answer(self.history, self.recent, "t5", max_tokens=3)

        before, before_calls = self.capture_forward(answer)
        with patch.object(self.engine.model.visual, "forward", side_effect=AssertionError("ViT reran")):
            self.adapter.probe_read_action([], self.recent, "t5 t6", dict(self.status, selected_history_records=0))
            self.adapter.probe_read_action(self.history, self.recent, "t6", self.status)
            after, after_calls = self.capture_forward(answer)
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[1]["input_tokens"], after[1]["input_tokens"])
        self.assertEqual(len(before_calls), len(after_calls))
        for (left, _), (right, _) in zip(before_calls, after_calls):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertNotIn("get_image_features", self.engine.model.model.__dict__)

    def test_action_mode_enables_history_without_fixed_read_limits(self):
        self.assertTrue(MemoryConfig(read_mode="action", read_frames=0, read_tokens=0).enabled)
        self.assertFalse(MemoryConfig(read_mode="action", capacity=0).enabled)


if __name__ == "__main__":
    unittest.main()

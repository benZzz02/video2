"""Native Qwen3-VL regressions for stateless probes and subsequent answers.

These tests use random tiny weights to check the inference interface.  They
cannot establish that a checkpoint's state decisions are semantically useful.
"""

from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
import transformers

import test_native_feature_adapter as native
from onestreamer.visual_memory import NativeFeatureAdapter, cached_image_features


@unittest.skipUnless(transformers.__version__ == "4.57.6", "Exact supported HF version required")
class StateProbeRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native.NativeFeatureRoundtripTests.setUpClass()
        cls.fixture = native.NativeFeatureRoundtripTests()

    def setUp(self):
        self.engine = self.fixture.processor_engine(native_states=True)
        self.adapter = NativeFeatureAdapter(self.engine)
        self.history = [self.fixture.frames[0]]
        self.recent = [replace(self.fixture.frames[1], timestamp=20.0)]
        limit = patch.object(self.engine.model.config.text_config, "max_position_embeddings", 2048)
        limit.start()
        self.addCleanup(limit.stop)

    def capture_forward(self, operation):
        outputs = []

        def capture(module, args, output):
            outputs.append((output.logits.detach().clone(), output.past_key_values))

        handle = self.engine.model.register_forward_hook(capture)
        try:
            result = operation()
        finally:
            handle.remove()
        return result, outputs

    def test_probe_matches_one_token_generate_after_poisoned_rope_and_changed_views(self):
        # The model stores rope_deltas on the object.  Every independent visual
        # prefill must recompute it, regardless of the previous query's shape.
        for history, question in (([], "t5"), (self.history, "t5 t6"), ([], "t5 t6 t7")):
            with self.subTest(history=len(history), question=question):
                frames = history + self.recent
                inputs, _ = self.adapter.prepare_inputs(
                    self.adapter.build_state_chat(history, self.recent, question), frames)
                self.engine.model.model.rope_deltas = torch.tensor([[999999]])
                with patch.object(self.engine.model.visual, "forward",
                                  side_effect=AssertionError("Probe reran ViT")):
                    probe, calls = self.capture_forward(
                        lambda: self.adapter.probe_state(history, self.recent, question))
                    self.assertEqual(len(calls), 1)
                    self.assertIsNone(calls[0][1])
                    probe_rope = self.engine.model.model.rope_deltas.clone()
                    with cached_image_features(self.engine, frames) as hook, torch.inference_mode():
                        generated = self.engine.model.generate(
                            **inputs, max_new_tokens=1, do_sample=False, use_cache=True,
                            num_beams=1, num_return_sequences=1,
                            pad_token_id=self.engine.processor.tokenizer.pad_token_id,
                            return_dict_in_generate=True, output_logits=True)
                self.assertEqual(hook["calls"], 1)
                torch.testing.assert_close(calls[0][0][:, -1].float(), generated.logits[0].float(),
                                           rtol=0, atol=0)
                torch.testing.assert_close(probe_rope, self.engine.model.model.rope_deltas,
                                           rtol=0, atol=0)
                self.assertEqual(probe["predicted_token_id"], int(generated.sequences[0, -1]))
                probabilities = generated.logits[0][0].float().softmax(-1)
                for state, token_id in self.adapter.state_token_ids().items():
                    self.assertEqual(probe["state_probabilities"][state], float(probabilities[token_id]))

    def test_interleaved_probes_do_not_change_final_answer_logits_or_tokens(self):
        def answer():
            return self.adapter.answer(self.history, self.recent, "t5", max_tokens=3)

        baseline, baseline_calls = self.capture_forward(answer)
        with patch.object(self.engine.model.visual, "forward",
                          side_effect=AssertionError("Cached probe/answer reran ViT")):
            self.adapter.probe_state([], self.recent, "t5 t6 t7")
            self.adapter.probe_state(self.history, self.recent, "t6")
            self.adapter.probe_state([], self.recent, "t5")
            actual, actual_calls = self.capture_forward(answer)
        self.assertEqual(actual[0], baseline[0])
        self.assertEqual(actual[1]["input_tokens"], baseline[1]["input_tokens"])
        self.assertEqual(len(actual_calls), len(baseline_calls))
        for (actual_logits, _), (baseline_logits, _) in zip(actual_calls, baseline_calls):
            torch.testing.assert_close(actual_logits, baseline_logits, rtol=0, atol=0)
        self.assertNotIn("get_image_features", self.engine.model.model.__dict__)

    def test_configured_suppression_changes_generation_but_not_unrestricted_probe(self):
        # This synthetic config distinguishes raw model logits from generation
        # processors.  It makes no assertion about the unavailable checkpoint's
        # generation_config.json and does not coerce a prediction into a state.
        baseline = self.adapter.probe_state(self.history, self.recent, "t5")
        raw_id = baseline["predicted_token_id"]
        frames = self.history + self.recent
        inputs, _ = self.adapter.prepare_inputs(
            self.adapter.build_state_chat(self.history, self.recent, "t5"), frames)
        with patch.object(self.engine.model.generation_config, "suppress_tokens", [raw_id]):
            actual = self.adapter.probe_state(self.history, self.recent, "t5")
            with cached_image_features(self.engine, frames), torch.inference_mode():
                generated = self.engine.model.generate(
                    **inputs, max_new_tokens=1, do_sample=False, use_cache=True,
                    pad_token_id=self.engine.processor.tokenizer.pad_token_id,
                    return_dict_in_generate=True, output_logits=True, output_scores=True)
        self.assertEqual(actual["predicted_token_id"], raw_id)
        self.assertEqual(int(generated.logits[0][0].argmax()), raw_id)
        self.assertTrue(bool(torch.isneginf(generated.scores[0][0, raw_id])))
        self.assertNotEqual(int(generated.sequences[0, -1]), raw_id)


if __name__ == "__main__":
    unittest.main()

"""State probes exercise a real tiny Qwen3-VL prefill without another ViT pass."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
import transformers

import test_native_feature_adapter as native
from onestreamer.visual_memory import MemoryConfig, NativeFeatureAdapter, STATE_TOKENS


@unittest.skipUnless(transformers.__version__ == "4.57.6", "Exact supported HF version required")
class StateAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native.NativeFeatureRoundtripTests.setUpClass()
        cls.fixture = native.NativeFeatureRoundtripTests()

    def setUp(self):
        self.engine = self.fixture.processor_engine(native_states=True)
        self.adapter = NativeFeatureAdapter(self.engine)
        self.history = [self.fixture.frames[0]]
        self.recent = [replace(self.fixture.frames[1], timestamp=20.0)]
        self.limit = patch.object(self.engine.model.config.text_config, "max_position_embeddings", 2048)
        self.limit.start()
        self.addCleanup(self.limit.stop)

    def test_state_count_matches_native_processor_and_preserves_visual_question(self):
        final_chat = self.adapter.build_chat(self.history, self.recent, "t5 t6")
        state_chat = self.adapter.build_state_chat(self.history, self.recent, "t5 t6")
        self.assertEqual(final_chat[1]["content"][:-1], state_chat[2]["content"])
        self.assertEqual([message["role"] for message in state_chat], ["system", "user", "user"])
        self.assertTrue(state_chat[1]["content"][0]["text"].startswith("t5 t6\n"))
        self.assertIn("begin with </Response> before the requested answer", state_chat[1]["content"][0]["text"])
        state_text = " ".join(item.get("text", "") for message in state_chat for item in message["content"])
        self.assertNotIn("Do not answer", state_text)
        self.assertNotIn("State token:", state_text)
        self.assertIn("</Response> followed by the answer", state_text)
        self.assertNotIn("</Response>", final_chat[0]["content"][0]["text"])
        self.assertIn("prediction", state_chat[0]["content"][0]["text"])
        self.assertIn("zero count", state_chat[0]["content"][0]["text"])
        inputs, _ = self.adapter.prepare_inputs(state_chat, self.history + self.recent)
        self.assertEqual(self.adapter.count_state_input_tokens(self.history, self.recent, "t5 t6"),
                         inputs["input_ids"].shape[-1])

    def test_direct_and_option_formats_remain_answer_tasks_with_a_state_prefix(self):
        for direct, question in ((False, "Question: What color?\nA. Red\nB. Blue\nBest option:("),
                                 (True, "Is a cup visible? Answer only Yes or No."),
                                 (True, "How many cups? Output only an integer.")):
            with self.subTest(direct=direct, question=question):
                before = self.adapter.build_chat(self.history, self.recent, question, direct)
                state = self.adapter.build_state_chat(self.history, self.recent, question, direct)
                after = self.adapter.build_chat(self.history, self.recent, question, direct)
                self.assertEqual(before, after)
                self.assertTrue(state[1]["content"][0]["text"].startswith(question + "\n"))
                self.assertEqual(state[2]["content"], before[1]["content"][:-1])
                self.assertIn("format applies to the answer after that prefix", state[0]["content"][0]["text"])

    def test_probe_uses_real_cached_prefill_and_unrestricted_probabilities(self):
        for predicted_id, expected in ((55, "</Response>"), (56, "</Standby>"),
                                       (57, "</Silence>"), (4, None)):
            with self.subTest(predicted_id=predicted_id):
                def head(hidden):
                    logits = torch.zeros((*hidden.shape[:-1], 64), device=hidden.device)
                    logits[..., predicted_id] = 3.0
                    return logits
                with patch.object(self.engine.model.lm_head, "forward", side_effect=head), \
                     patch.object(self.engine.model.visual, "forward", side_effect=AssertionError("ViT reran")):
                    result = self.adapter.probe_state(self.history, self.recent, "t5")
                self.assertEqual(result["state"], expected)
                self.assertEqual(result["predicted_token_id"], predicted_id)
                self.assertEqual(set(result["state_probabilities"]), set(STATE_TOKENS))
                self.assertLess(result["state_probability_mass"], 1.0)
                probability = torch.tensor([3.0] + [0.0] * 63).softmax(0)[0].item()
                self.assertAlmostEqual(result["predicted_probability"], probability)
                self.assertEqual(result["visual_tokens"], 6)
                self.assertNotIn("get_image_features", self.engine.model.model.__dict__)
                # The next ordinary answer must work after a cache-free probe.
                _, metrics = self.adapter.answer(self.history, self.recent, "t5", max_tokens=1)
                self.assertEqual(metrics["visual_tokens"], 6)

    def test_probe_rejects_missing_native_special_tokens(self):
        adapter = NativeFeatureAdapter(self.fixture.processor_engine())
        with self.assertRaisesRegex(ValueError, "atomic native special"):
            adapter.probe_state(self.history, self.recent, "t5")

    def test_probe_rejects_overflow_before_model_call(self):
        count = self.adapter.count_state_input_tokens(self.history, self.recent, "t5")
        with patch.object(self.engine.model, "forward", side_effect=AssertionError("model ran")), \
             self.assertRaisesRegex(ValueError, "exceeds context limit"):
            self.adapter.probe_state(self.history, self.recent, "t5", context_tokens=count)

    def test_state_enablement_and_budget_validation(self):
        self.assertTrue(MemoryConfig(read_mode="state", read_frames=0, read_tokens=0).enabled)
        self.assertFalse(MemoryConfig(read_mode="state", capacity=0).enabled)
        for budgets in ((), (0,), (True,), (10, 10), (10, 1)):
            with self.subTest(budgets=budgets), self.assertRaises(ValueError):
                MemoryConfig(read_mode="state", state_history_budgets=budgets)


if __name__ == "__main__":
    unittest.main()

"""CPU-only contracts for model-controlled visual history reading."""

from dataclasses import dataclass
from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onestreamer.state_readout import RESPONSE, SILENCE, STANDBY, run_state_readout


@dataclass(frozen=True)
class Frame:
    timestamp: float
    token_count: int = 1


class Adapter:
    def __init__(self, states, limit=10000, answer_base=20, state_base=30):
        self.states = iter(states)
        self.limit = limit
        self.answer_base = answer_base
        self.state_base = state_base
        self.probes = []
        self.count_calls = 0

    def context_limit(self, cap):
        return self.limit if cap is None else min(self.limit, cap)

    def count_input_tokens(self, history, recent, question, direct=False):
        self.count_calls += 1
        return self.answer_base + sum(frame.token_count + 2 for frame in history + recent)

    def count_state_input_tokens(self, history, recent, question, direct=False):
        self.count_calls += 1
        return self.state_base + sum(frame.token_count + 2 for frame in history + recent)

    def probe_state(self, history, recent, question, direct=False, context_tokens=None):
        self.probes.append((tuple(history), tuple(recent), question, direct))
        return dict(state=next(self.states), predicted_token_id=42,
                    predicted_token="raw model output", state_probabilities={RESPONSE: 0.4})


class StateReadoutTests(unittest.TestCase):
    def test_response_keeps_exact_recent_only_evidence(self):
        adapter = Adapter([RESPONSE])
        recent = [Frame(10), Frame(11)]
        selected, meta = run_state_readout(adapter, [Frame(0)], recent, "What is visible?")
        self.assertEqual(selected, [])
        self.assertEqual(adapter.probes, [((), tuple(recent), "What is visible?", False)])
        self.assertEqual(meta["state_stop_reason"], "response")
        self.assertEqual(meta["state_probe_count"], 1)
        self.assertEqual(meta["state_rounds"][0]["history_budget_tokens"], 0)
        self.assertEqual(meta["state_rounds"][0]["predicted_token_id"], 42)
        json.dumps(meta)

    def test_standby_and_silence_expand_nested_visual_views(self):
        adapter = Adapter([SILENCE, STANDBY, RESPONSE])
        candidates = [Frame(i, 2) for i in range(8)]
        selected, meta = run_state_readout(adapter, candidates, [Frame(10)], "Why?",
                                           direct=True, history_budgets=(4, 8, 16))
        first = adapter.probes[1][0]
        second = adapter.probes[2][0]
        self.assertEqual([frame.timestamp for frame in first], [0, 7])
        self.assertTrue(set(first) < set(second))
        self.assertEqual(list(second), selected)
        self.assertEqual([r["history_visual_tokens"] for r in meta["state_rounds"]], [0, 4, 8])
        self.assertEqual(meta["state_stop_reason"], "response")
        self.assertTrue(all(probe[3] for probe in adapter.probes))

    def test_budget_exhaustion_is_bounded(self):
        adapter = Adapter([STANDBY] * 3)
        selected, meta = run_state_readout(adapter, [Frame(i) for i in range(8)], [Frame(10)],
                                           "How many?", history_budgets=(2, 4))
        self.assertEqual(len(selected), 4)
        self.assertEqual(meta["state_probe_count"], 3)
        self.assertEqual(meta["state_stop_reason"], "history_budget_exhausted")

    def test_invalid_state_preserves_current_view(self):
        for invalid in (None, "</Summary>", "Response", "</Response> extra"):
            with self.subTest(invalid=invalid):
                adapter = Adapter([STANDBY, invalid])
                selected, meta = run_state_readout(adapter, [Frame(i) for i in range(8)],
                                                   [Frame(10)], "Question", history_budgets=(2, 4))
                self.assertEqual(len(selected), 2)
                self.assertEqual(meta["state_stop_reason"], "invalid_state")
                self.assertEqual(meta["state_probe_count"], 2)

    def test_no_history_still_probes_once(self):
        adapter = Adapter([SILENCE])
        selected, meta = run_state_readout(adapter, [], [Frame(10)], "Question")
        self.assertEqual(selected, [])
        self.assertEqual(meta["state_stop_reason"], "no_history")
        self.assertEqual(meta["state_probe_count"], 1)

    def test_all_history_exhaustion_stops(self):
        adapter = Adapter([STANDBY, STANDBY])
        selected, meta = run_state_readout(adapter, [Frame(1)], [Frame(10)], "Question",
                                           history_budgets=(1, 2, 3))
        self.assertEqual(selected, [Frame(1)])
        self.assertEqual(meta["state_stop_reason"], "history_exhausted")
        self.assertEqual(meta["state_probe_count"], 2)

    def test_oversized_frames_wait_for_larger_budget_without_duplicate_probe(self):
        adapter = Adapter([STANDBY, RESPONSE])
        selected, meta = run_state_readout(adapter, [Frame(1, 8)], [Frame(10)], "Question",
                                           history_budgets=(1, 4, 8))
        self.assertEqual(selected, [Frame(1, 8)])
        self.assertEqual([r["history_budget_tokens"] for r in meta["state_rounds"]], [0, 8])
        self.assertEqual([r["added_frames"] for r in meta["state_selection_stages"]], [0, 0, 1])

    def test_skips_large_bundle_to_admit_smaller_ones(self):
        adapter = Adapter([STANDBY, RESPONSE])
        selected, _ = run_state_readout(adapter, [Frame(0, 100), Frame(1, 1), Frame(2, 2)],
                                         [Frame(10)], "Question", history_budgets=(3, 100))
        self.assertEqual(selected, [Frame(1, 1), Frame(2, 2)])

    def test_answer_context_reserve_binds_independently(self):
        adapter = Adapter([STANDBY, STANDBY], limit=50, answer_base=20, state_base=10)
        selected, meta = run_state_readout(adapter, [Frame(i, 5) for i in range(8)], [Frame(10)],
                                           "Question", history_budgets=(10, 20), max_tokens=16)
        self.assertEqual(len(selected), 1)
        self.assertEqual(meta["state_stop_reason"], "context_limit")
        self.assertLessEqual(meta["selected_input_tokens"] + 16, 50)

    def test_longer_state_prompt_binds_independently(self):
        adapter = Adapter([STANDBY, STANDBY], limit=50, answer_base=5, state_base=35)
        selected, meta = run_state_readout(adapter, [Frame(i, 5) for i in range(8)], [Frame(10)],
                                           "Question", history_budgets=(10, 20), max_tokens=16)
        self.assertEqual(len(selected), 1)
        self.assertLessEqual(meta["state_selected_input_tokens"] + 1, 50)
        self.assertEqual(meta["state_stop_reason"], "context_limit")

    def test_context_clip_accepts_exact_boundary(self):
        adapter = Adapter([STANDBY, RESPONSE], limit=42, answer_base=20, state_base=30)
        selected, meta = run_state_readout(adapter, [Frame(1), Frame(2)], [Frame(10)], "Question",
                                           history_budgets=(1, 2), max_tokens=16)
        self.assertEqual(selected, [Frame(1)])
        self.assertEqual(meta["remaining_input_tokens"], 0)

    def test_recent_only_must_fit_both_prompts_before_probe(self):
        for answer_base, state_base in ((40, 0), (0, 50)):
            with self.subTest(answer_base=answer_base, state_base=state_base):
                adapter = Adapter([], limit=50, answer_base=answer_base, state_base=state_base)
                with self.assertRaisesRegex(ValueError, "Recent context/question"):
                    run_state_readout(adapter, [], [Frame(10)], "Question")
                self.assertEqual(adapter.probes, [])

    def test_large_bank_context_search_uses_logarithmic_prompt_counts(self):
        adapter = Adapter([STANDBY, RESPONSE], limit=1000)
        selected, meta = run_state_readout(adapter, [Frame(i) for i in range(4096)],
                                           [Frame(10000)], "Question", history_budgets=(4096,))
        self.assertGreater(len(selected), 100)
        self.assertLess(adapter.count_calls, 40)
        self.assertEqual(adapter.count_calls, meta["budget_count_calls"])
        self.assertLessEqual(meta["selected_input_tokens"] + 16, 1000)
        self.assertLessEqual(meta["state_selected_input_tokens"] + 1, 1000)

    def test_input_frames_unchanged_and_candidates_sorted(self):
        candidates = [Frame(2), Frame(0), Frame(1)]
        recent = [Frame(10)]
        adapter = Adapter([STANDBY, RESPONSE])
        selected, _ = run_state_readout(adapter, candidates, recent, "Question", history_budgets=(3,))
        self.assertEqual(candidates, [Frame(2), Frame(0), Frame(1)])
        self.assertEqual(recent, [Frame(10)])
        self.assertEqual(selected, [Frame(0), Frame(1), Frame(2)])

    def test_rejects_invalid_chronology_and_token_bundles(self):
        cases = [([Frame(0), Frame(0)], [Frame(10)]),
                 ([Frame(10)], [Frame(10)]), ([Frame(11)], [Frame(10)]),
                 ([], [Frame(11), Frame(10)]), ([], []),
                 ([Frame(float("nan"))], [Frame(10)]),
                 ([Frame(0, 0)], [Frame(10)]), ([], [Frame(10, True)])]
        for candidates, recent in cases:
            with self.subTest(candidates=candidates, recent=recent):
                with self.assertRaises(ValueError):
                    run_state_readout(Adapter([]), candidates, recent, "Question")

    def test_rejects_invalid_budget_schedules_and_reserves(self):
        for budgets in ((), (0,), (-1,), (1, 1), (2, 1), (True,), (1.5,)):
            with self.subTest(budgets=budgets):
                with self.assertRaises(ValueError):
                    run_state_readout(Adapter([]), [], [Frame(10)], "Q", history_budgets=budgets)
        for reserve in (0, -1, True, 1.5, 10000):
            with self.subTest(reserve=reserve):
                with self.assertRaises(ValueError):
                    run_state_readout(Adapter([]), [], [Frame(10)], "Q", max_tokens=reserve)


if __name__ == "__main__":
    unittest.main()

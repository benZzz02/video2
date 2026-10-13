"""CPU contracts for explicit, training-free historical reading actions."""

from dataclasses import dataclass
from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onestreamer.action_readout import READ_MORE, STOP, run_action_readout


@dataclass(frozen=True)
class Frame:
    timestamp: float
    token_count: int = 1


class Adapter:
    def __init__(self, actions=(), limit=10000, answer_base=20, action_base=30):
        self.actions = iter(actions)
        self.limit = limit
        self.answer_base = answer_base
        self.action_base = action_base
        self.probes = []
        self.measured_status = {}
        self.count_calls = 0

    def context_limit(self, cap):
        return self.limit if cap is None else min(self.limit, cap)

    def count_input_tokens(self, history, recent, question, direct=False):
        self.count_calls += 1
        return self.answer_base + sum(frame.token_count + 2 for frame in history + recent)

    def count_read_action_input_tokens(self, history, recent, question, history_status, direct=False):
        self.count_calls += 1
        self.measured_status[tuple(history)] = json.loads(json.dumps(history_status))
        return self.action_base + sum(frame.token_count + 2 for frame in history + recent)

    def probe_read_action(self, history, recent, question, history_status,
                          direct=False, context_tokens=None):
        self.probes.append(dict(history=tuple(history), recent=tuple(recent), question=question,
                                history_status=history_status, direct=direct))
        return dict(action=next(self.actions), predicted_token_id=42, predicted_token="raw")


def run(adapter, candidates, recent=None, question="Question", **kwargs):
    return run_action_readout(adapter, candidates, [Frame(10)] if recent is None else recent,
                              question, query_time=kwargs.pop("query_time", 12),
                              stream_start=kwargs.pop("stream_start", 0), **kwargs)


class ActionReadoutTests(unittest.TestCase):
    def test_stop_preserves_recent_and_reports_truthful_unread_status(self):
        adapter = Adapter([STOP])
        recent = [Frame(10), Frame(11)]
        selected, meta = run(adapter, [Frame(1), Frame(3), Frame(9)], recent, "What is visible?")
        self.assertEqual(selected, [])
        self.assertEqual(meta["action_stop_reason"], "model_stop")
        self.assertEqual(meta["action_probe_count"], 1)
        probe = adapter.probes[0]
        self.assertEqual(probe["recent"], tuple(recent))
        status = probe["history_status"]
        self.assertEqual((status["query_time"], status["stream_start"]), (12, 0))
        self.assertEqual(status["earliest_retained_timestamp"], 1)
        self.assertEqual((status["recent_start"], status["recent_end"]), (10, 11))
        self.assertEqual(status["retained_history_records"], 3)
        self.assertEqual(status["selected_history_records"], 0)
        self.assertEqual(status["unread_history_records"], 3)
        self.assertEqual(sum(item["unread_records"] for item in status["unread_time_bins"]), 3)
        self.assertIn("not_complete_coverage", status["history_observations"])
        self.assertEqual(meta["action_rounds"][0]["predicted_token_id"], 42)
        self.assertEqual(status, adapter.measured_status[()])
        self.assertEqual(meta["budget_count_calls"], adapter.count_calls)
        json.dumps(meta, allow_nan=False)

    def test_read_more_expands_nested_original_temporal_order(self):
        adapter = Adapter([READ_MORE, READ_MORE, STOP])
        candidates = [Frame(i, 2) for i in range(8)]
        selected, meta = run(adapter, candidates, history_budgets=(4, 8, 16), direct=True)
        first, second = adapter.probes[1]["history"], adapter.probes[2]["history"]
        self.assertEqual([frame.timestamp for frame in first], [0, 7])
        self.assertTrue(set(first) < set(second))
        self.assertEqual(selected, list(second))
        self.assertEqual([item["history_visual_tokens"] for item in meta["action_rounds"]], [0, 4, 8])
        self.assertTrue(all(probe["direct"] for probe in adapter.probes))
        for probe in adapter.probes:
            status = probe["history_status"]
            self.assertEqual(status["selected_history_records"], len(probe["history"]))
            self.assertEqual(status["unread_history_records"], len(candidates) - len(probe["history"]))
            self.assertEqual(sum(item["unread_records"] for item in status["unread_time_bins"]),
                             status["unread_history_records"])

    def test_invalid_action_advances_and_remains_in_trace(self):
        for invalid in (None, "</Response>", "STOP extra", "", 9):
            with self.subTest(invalid=invalid):
                adapter = Adapter([invalid, STOP])
                selected, meta = run(adapter, [Frame(i) for i in range(8)], history_budgets=(2, 4))
                self.assertEqual(len(selected), 2)
                self.assertEqual(meta["action_invalid_count"], 1)
                first = meta["action_rounds"][0]
                self.assertEqual(first["action"], invalid)
                self.assertTrue(first["invalid_action"])
                self.assertEqual(first["effective_action"], READ_MORE)
                self.assertEqual(meta["action_stop_reason"], "model_stop")

    def test_no_history_skips_probe_and_needs_only_answer_space(self):
        adapter = Adapter([], limit=50, action_base=1000)
        selected, meta = run(adapter, [], [Frame(0), Frame(1)], query_time=2)
        self.assertEqual(selected, [])
        self.assertEqual(meta["action_stop_reason"], "no_history")
        self.assertEqual(meta["action_probe_count"], 0)
        self.assertGreater(meta["action_selected_input_tokens"], 50)
        self.assertLessEqual(meta["selected_input_tokens"] + 16, 50)

    def test_all_history_exhaustion_skips_terminal_probe(self):
        adapter = Adapter([READ_MORE])
        selected, meta = run(adapter, [Frame(1)], history_budgets=(1, 2, 3))
        self.assertEqual(selected, [Frame(1)])
        self.assertEqual(meta["action_stop_reason"], "history_exhausted")
        self.assertEqual(meta["action_probe_count"], 1)
        self.assertEqual(meta["final_history_status"]["unread_history_records"], 0)

    def test_final_budget_skips_terminal_probe(self):
        adapter = Adapter([READ_MORE, READ_MORE])
        selected, meta = run(adapter, [Frame(i) for i in range(8)], history_budgets=(2, 4))
        self.assertEqual(len(selected), 4)
        self.assertEqual(meta["action_stop_reason"], "history_budget_exhausted")
        self.assertEqual(meta["action_probe_count"], 2)

    def test_oversized_record_skips_unchanged_views(self):
        adapter = Adapter([READ_MORE])
        selected, meta = run(adapter, [Frame(1, 8)], history_budgets=(1, 4, 8))
        self.assertEqual(selected, [Frame(1, 8)])
        self.assertEqual(meta["action_probe_count"], 1)
        self.assertEqual([stage["added_frames"] for stage in meta["action_selection_stages"]], [0, 0, 1])

    def test_unaffordable_history_exhausts_budget_without_probe(self):
        adapter = Adapter([])
        selected, meta = run(adapter, [Frame(1, 9)], history_budgets=(1, 4, 8))
        self.assertEqual(selected, [])
        self.assertEqual(meta["action_stop_reason"], "history_budget_exhausted")
        self.assertEqual(meta["action_probe_count"], 0)

    def test_selector_skips_large_bundle_and_keeps_small_records(self):
        adapter = Adapter([READ_MORE, STOP])
        selected, _ = run(adapter, [Frame(0, 100), Frame(1), Frame(2, 2)], history_budgets=(3, 200))
        self.assertEqual(selected, [Frame(1), Frame(2, 2)])

    def test_answer_context_clipping_and_exact_reserve(self):
        adapter = Adapter([READ_MORE, READ_MORE], limit=50, answer_base=20, action_base=5)
        selected, meta = run(adapter, [Frame(i, 5) for i in range(8)], history_budgets=(10, 20))
        self.assertEqual(len(selected), 1)
        self.assertEqual(meta["action_stop_reason"], "context_limit")
        self.assertTrue(meta["action_context_clipped"])
        self.assertLessEqual(meta["selected_input_tokens"] + 16, 50)
        self.assertEqual(meta["action_probe_count"], 2)

    def test_context_blocks_all_extensions_without_reprobing_unchanged_view(self):
        adapter = Adapter([READ_MORE], limit=50, answer_base=30, action_base=5)
        selected, meta = run(adapter, [Frame(i, 2) for i in range(4)], history_budgets=(2, 4))
        self.assertEqual(selected, [])
        self.assertEqual(meta["action_stop_reason"], "context_limit")
        self.assertEqual(meta["action_probe_count"], 1)
        self.assertEqual(meta["action_invalid_count"], 0)
        self.assertTrue(all(stage["context_clipped"] for stage in meta["action_selection_stages"]))
        self.assertTrue(all(stage["added_frames"] == 0 for stage in meta["action_selection_stages"]))

    def test_nonmonotone_metadata_count_does_not_lose_larger_fitting_prefix(self):
        class NonmonotoneAdapter(Adapter):
            def count_read_action_input_tokens(self, history, recent, question, history_status, direct=False):
                super().count_read_action_input_tokens(history, recent, question, history_status, direct)
                # Prefixes of length 1 and 3 fit, while 2 and 4 do not. A
                # monotonic binary search would discard the fitting length 3.
                return {0: 30, 1: 40, 2: 200, 3: 50, 4: 200}.get(len(history), 200)

        adapter = NonmonotoneAdapter([READ_MORE, STOP], limit=100)
        selected, meta = run(adapter, [Frame(i) for i in range(8)], history_budgets=(4, 8))
        self.assertEqual(len(selected), 3)
        self.assertEqual(meta["action_selected_input_tokens"], 50)
        self.assertEqual(meta["action_stop_reason"], "model_stop")
        self.assertEqual(meta["action_selection_stages"][0]["added_frames"], 3)
        self.assertEqual(adapter.measured_status[tuple(selected)], meta["final_history_status"])

    def test_terminal_view_can_exceed_action_prompt_cap(self):
        adapter = Adapter([READ_MORE], limit=50, answer_base=5, action_base=45)
        selected, meta = run(adapter, [Frame(i, 5) for i in range(3)], history_budgets=(10,))
        self.assertEqual(len(selected), 2)
        self.assertGreater(meta["action_selected_input_tokens"] + 1, 50)
        self.assertLessEqual(meta["selected_input_tokens"] + 16, 50)
        self.assertEqual(meta["action_stop_reason"], "history_budget_exhausted")
        self.assertEqual(meta["action_probe_count"], 1)

    def test_recent_exceeding_required_prompt_cap_fails_before_probe(self):
        for answer_base, action_base in ((40, 0), (0, 50)):
            with self.subTest(answer_base=answer_base, action_base=action_base):
                adapter = Adapter([], limit=50, answer_base=answer_base, action_base=action_base)
                with self.assertRaisesRegex(ValueError, "Recent context/question"):
                    run(adapter, [Frame(1)])
                self.assertEqual(adapter.probes, [])

    def test_no_history_answer_overflow_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Recent context/question"):
            run(Adapter([], limit=50, answer_base=40), [])

    def test_history_bins_are_bounded_and_report_retained_records_only(self):
        adapter = Adapter([STOP])
        _, meta = run(adapter, [Frame(index * 2) for index in range(100)], [Frame(300)], query_time=350)
        status = meta["final_history_status"]
        self.assertEqual(len(status["unread_time_bins"]), 8)
        self.assertEqual(sum(group["unread_records"] for group in status["unread_time_bins"]), 100)
        self.assertEqual(status["unread_time_bins"][-1]["end"], 198)
        self.assertEqual(status["query_time"], 350)
        self.assertEqual(status["recent_end"], 300)
        self.assertEqual(set(status), {
            "query_time", "stream_start", "recent_start", "recent_end",
            "retained_history_records", "selected_history_records", "unread_history_records",
            "unread_time_bins", "history_observations", "earliest_retained_timestamp",
            "timestamp_origin_note",
        })

    def test_fractional_stream_origin_preserves_quantized_source_timestamps(self):
        adapter = Adapter([STOP])
        candidates, recent = [Frame(0.1)], [Frame(1.1), Frame(2.1)]
        selected, meta = run(adapter, candidates, recent, query_time=3.15, stream_start=0.15)
        self.assertEqual(selected, [])
        status = meta["final_history_status"]
        self.assertEqual(status["stream_start"], 0.15)
        self.assertEqual(status["earliest_retained_timestamp"], 0.1)
        self.assertEqual(status["unread_time_bins"], [{"start": 0.1, "end": 0.1, "unread_records": 1}])
        self.assertEqual(adapter.probes[0]["recent"], tuple(recent))
        self.assertIn("quantizes", status["timestamp_origin_note"])

    def test_input_lists_are_unchanged_and_history_is_sorted(self):
        candidates, recent = [Frame(2), Frame(0), Frame(1)], [Frame(10)]
        selected, _ = run(Adapter([READ_MORE]), candidates, recent, history_budgets=(3,))
        self.assertEqual(candidates, [Frame(2), Frame(0), Frame(1)])
        self.assertEqual(recent, [Frame(10)])
        self.assertEqual(selected, [Frame(0), Frame(1), Frame(2)])

    def test_invalid_chronology_and_frame_tokens_are_rejected(self):
        cases = [([Frame(0), Frame(0)], [Frame(10)]),
                 ([Frame(10)], [Frame(10)]), ([Frame(11)], [Frame(10)]),
                 ([], [Frame(11), Frame(10)]), ([], []),
                 ([Frame(float("nan"))], [Frame(10)]),
                 ([Frame(0, 0)], [Frame(10)]), ([], [Frame(10, True)]),
                 ([], [Frame(13)])]
        for candidates, recent in cases:
            with self.subTest(candidates=candidates, recent=recent), self.assertRaises(ValueError):
                run(Adapter([]), candidates, recent)

    def test_invalid_explicit_query_bounds_are_rejected(self):
        for query_time, stream_start in ((float("nan"), 0), (float("inf"), 0),
                                         (12, float("inf")), (12, 13), (9, 0), (10, 0),
                                         (True, 0), (12, None)):
            with self.subTest(query_time=query_time, stream_start=stream_start), self.assertRaises(ValueError):
                run(Adapter([]), [], query_time=query_time, stream_start=stream_start)

    def test_invalid_budgets_and_reserves_are_rejected(self):
        for budgets in ((), (0,), (-1,), (1, 1), (2, 1), (True,), (1.5,)):
            with self.subTest(budgets=budgets), self.assertRaises(ValueError):
                run(Adapter([]), [], history_budgets=budgets)
        for reserve in (0, -1, True, 1.5, 10000):
            with self.subTest(reserve=reserve), self.assertRaises(ValueError):
                run(Adapter([]), [], max_tokens=reserve)


if __name__ == "__main__":
    unittest.main()

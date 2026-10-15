"""CPU-only checks for the experimental full-run frame-ID answer prompt."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "Eval/Proactive_Eval/OneStreamer/src"
sys.path.insert(0, str(SOURCE))

from onestreamer import frame_evidence  # noqa: E402
from onestreamer.ovobench import score_prediction  # noqa: E402
from onestreamer.schema import Sample  # noqa: E402
from onestreamer.visual_memory import NativeFeatureAdapter  # noqa: E402


def frame(timestamp):
    return SimpleNamespace(timestamp=float(timestamp))


def sample(mode, answer, options=None):
    return Sample.from_dict(dict(
        sample_id="test", index=0, task="test", subtask="test", video="test.mp4",
        start=0, end=12, question="What happened?", options=options or {},
        answer=answer, answer_text=answer, eval_mode=mode))


class FrameEvidencePromptTests(unittest.TestCase):
    def setUp(self):
        self.history = [frame(1), frame(5)]
        self.recent = [frame(9), frame(10)]
        self.indices = {1.0: 0, 5.0: 3}
        self.specs = frame_evidence.frame_specs(self.history, self.recent, self.indices)
        # build_chat uses no engine state; the real adapter method is sufficient.
        self.adapter = object.__new__(NativeFeatureAdapter)

    def test_ids_stay_adjacent_to_their_image_and_question_is_unchanged(self):
        question = "Question:\nWhich item?\nOptions:\nA. chair\nB. table\nBest option:("
        original = self.adapter.build_chat(self.history, self.recent, question)
        witness = frame_evidence.build_chat(
            self.adapter, self.history, self.recent, question, False, self.specs,
            "mcq_letter")
        expected = [("H00", 1.0), ("H03", 5.0), ("R00", 9.0), ("R01", 10.0)]
        self.assertEqual([(spec["id"], spec["timestamp"]) for spec in self.specs], expected)
        self.assertEqual(witness[1]["content"][-2], {"type": "text", "text": question})
        self.assertEqual(original[1]["content"][-1], {"type": "text", "text": question})
        positions = [i for i, part in enumerate(witness[1]["content"])
                     if part["type"] == "image"]
        self.assertEqual(len(positions), len(expected))
        for position, (fid, timestamp) in zip(positions, expected):
            self.assertEqual(witness[1]["content"][position - 1],
                             {"type": "text", "text": f"[{fid}] <{timestamp:.3f}s>"})
        self.assertEqual(witness[0]["content"][0]["text"], frame_evidence.SYSTEM)
        self.assertEqual(witness[1]["content"][-1]["text"], frame_evidence.FOOTER)

    def test_substring_question_uses_short_answer_footer(self):
        chat = frame_evidence.build_chat(
            self.adapter, self.history, self.recent, "Name the object", True,
            self.specs, "substring_match")
        self.assertEqual(chat[1]["content"][-2]["text"], "Name the object")
        self.assertEqual(chat[1]["content"][-1]["text"],
                         frame_evidence.SUBSTRING_FOOTER)

    def test_missing_or_nonchronological_frame_map_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "eligible-history ID"):
            frame_evidence.frame_specs(self.history, self.recent, {1.0: 0})
        with self.assertRaisesRegex(ValueError, "nonunique or nonchronological"):
            frame_evidence.frame_specs(self.history, self.recent, {1.0: 0, 5.0: 0})
        with self.assertRaisesRegex(ValueError, "at least one recent"):
            frame_evidence.frame_specs(self.history, [], self.indices)

    def test_mcq_answer_scored_from_answer_field_only(self):
        row = sample("mcq_letter", "B", {"A": "chair", "B": "table"})
        raw = "ANSWER: B\nEVIDENCE: H00, R01"
        witness = frame_evidence.parse_witness(
            raw, self.specs, row.eval_mode, row.question,
            valid_letters="".join(row.options))
        self.assertTrue(witness["witness_structure_valid"])
        self.assertEqual(witness["cited_ids"], ["H00", "R01"])
        self.assertEqual(frame_evidence.answer_for_scoring(witness), "B")
        self.assertEqual(score_prediction(row, frame_evidence.answer_for_scoring(witness)),
                         ("B", 1))

    def test_integer_and_substring_scoring_ignores_evidence_text(self):
        count = sample("integer_match", "3")
        witness = frame_evidence.parse_witness(
            "ANSWER: 3\nEVIDENCE: H00, H03, R01", self.specs,
            count.eval_mode, count.question)
        self.assertTrue(witness["witness_structure_valid"])
        self.assertEqual(score_prediction(count, frame_evidence.answer_for_scoring(witness)),
                         ("3", 1))

        object_row = sample("substring_match", "red cup")
        witness = frame_evidence.parse_witness(
            "ANSWER: blue cup\nEVIDENCE: R01", self.specs,
            object_row.eval_mode, object_row.question)
        self.assertTrue(witness["witness_structure_valid"])
        self.assertEqual(score_prediction(object_row, frame_evidence.answer_for_scoring(witness)),
                         ("blue cup", 0))
        self.assertEqual(score_prediction(object_row,
                                          frame_evidence.answer_for_scoring(
                                              frame_evidence.parse_witness(
                                                  "ANSWER: red cup\nEVIDENCE: R01", self.specs,
                                                  object_row.eval_mode, object_row.question))),
                         ("red cup", 1))

    def test_invalid_reference_is_flagged_separately_from_answer_accuracy(self):
        row = sample("mcq_letter", "B", {"A": "chair", "B": "table"})
        witness = frame_evidence.parse_witness(
            "ANSWER: B\nEVIDENCE: H99", self.specs, row.eval_mode,
            row.question, valid_letters="".join(row.options))
        self.assertTrue(witness["format_valid"])
        self.assertEqual(witness["unknown_ids"], ["H99"])
        self.assertFalse(witness["witness_structure_valid"])
        self.assertEqual(score_prediction(row, frame_evidence.answer_for_scoring(witness)),
                         ("B", 1))

    def test_missing_answer_scores_zero(self):
        rows = [sample("mcq_letter", "B", {"A": "chair", "B": "table"}),
                sample("integer_match", "3"),
                sample("substring_match", "red cup")]
        for row in rows:
            for raw in ("EVIDENCE: H00", "B, EVIDENCE: H00",
                        "ANSWER:\nEVIDENCE: H00"):
                with self.subTest(mode=row.eval_mode, raw=raw):
                    witness = frame_evidence.parse_witness(
                        raw, self.specs, row.eval_mode, row.question,
                        valid_letters="".join(row.options))
                    self.assertEqual(frame_evidence.answer_for_scoring(witness), "")
                    self.assertEqual(score_prediction(row, "")[1], 0)

    def test_citation_format_failure_does_not_erase_valid_answer(self):
        rows = [sample("mcq_letter", "B", {"A": "chair", "B": "table"}),
                sample("integer_match", "3"),
                sample("substring_match", "red cup")]
        for row, answer in zip(rows, ("B", "3", "red cup")):
            with self.subTest(mode=row.eval_mode):
                witness = frame_evidence.parse_witness(
                    f"ANSWER: {answer}\nEVIDEN: [R01]", self.specs,
                    row.eval_mode, row.question, valid_letters="".join(row.options))
                self.assertFalse(witness["format_valid"])
                self.assertFalse(witness["witness_structure_valid"])
                self.assertEqual(frame_evidence.answer_for_scoring(witness), answer)
                self.assertEqual(score_prediction(row, answer)[1], 1)


if __name__ == "__main__":
    unittest.main()

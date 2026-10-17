"""CPU tests for query-time routing between Recent16 and stored history."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib.util
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onestreamer import question_router, visual_memory
from onestreamer.visual_memory import MemoryConfig
from test_budget_memory import grid_frame


class RouterHelperTests(unittest.TestCase):
    def test_calibration_removes_label_prior(self):
        prior = {"NOW": -0.5, "SHORT": -2.0, "LONG": -1.5}
        self.assertEqual(question_router.choose_scope({"NOW": -0.6, "SHORT": -1.9, "LONG": -1.0}, prior), "LONG")
        self.assertEqual(question_router.choose_scope(prior, prior), "NOW")

    def test_rotations_cover_every_label_and_keep_the_question_quoted(self):
        chats = list(question_router.scope_chats("How many times? Answer only with a number."))
        self.assertEqual(len(chats), 3)
        firsts = set()
        for chat in chats:
            system, user = chat[0]["content"][0]["text"], chat[1]["content"][0]["text"]
            self.assertTrue(all(label in system for label in question_router.LABELS))
            self.assertIn("<VIEWER_QUESTION>\nHow many times? Answer only with a number.\n</VIEWER_QUESTION>", user)
            firsts.add(system.split("Options:\n")[1].split(" ")[0])
        self.assertEqual(firsts, set(question_router.LABELS))

    def test_newest_history_keeps_latest_records_within_cap(self):
        frames = [grid_frame(t, height=2, width=2) for t in range(10)]  # 4 tokens each
        picked = question_router.newest_history(frames, 13)
        self.assertEqual([f.timestamp for f in picked], [7.0, 8.0, 9.0])
        self.assertEqual(question_router.newest_history(frames, 3), [])

    def test_config_validation(self):
        MemoryConfig(read_mode="routed", route_threshold=0.8, route_escalation="LONG")
        for bad in (dict(route_threshold=1.5), dict(route_escalation="NOW"), dict(short_history_tokens=0)):
            with self.assertRaises(ValueError):
                MemoryConfig(read_mode="routed", **bad)


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is needed only for synthetic-video tests")
class RoutedRunnerTests(unittest.TestCase):
    def test_now_confident_now_escalated_and_long_paths(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "stream.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (32, 32))
            if not writer.isOpened():
                self.skipTest("This OpenCV build cannot write MJPG test videos")
            try:
                for index in range(160):
                    writer.write(np.full((32, 32, 3), index, dtype=np.uint8))
            finally:
                writer.release()
            template = dict(task="backward_tracking", subtask="EPM", video=video.name, start=0.0,
                            options={"A": "cup", "B": "plate"}, answer="A", answer_text="cup",
                            eval_mode="mcq_letter", media=str(video))
            rows = [dict(template, sample_id="now-sure", index=0, end=40.0, question="now sure"),
                    dict(template, sample_id="now-unsure", index=1, end=60.0, question="now unsure"),
                    dict(template, sample_id="long", index=2, end=78.0, question="long")]
            job = dict(model="m", model_path="/unused", video_root=str(root))
            answered = []

            class Adapter:
                def __init__(self, engine):
                    self.engine = engine

                def encode(self, images, max_pixels=None, store_on_cpu=True):
                    return [grid_frame(t, channels=2, layers=1) for t, _ in images]

                def context_limit(self, cap):
                    return cap

                def count_input_tokens(self, history, recent, question, direct=False):
                    return 40 + sum(f.token_count + 6 for f in [*history, *recent])

                def select_history(self, candidates, recent, question, direct=False, max_tokens=16,
                                   context_tokens=None):
                    return list(candidates), {"eligible_history_frames": len(candidates)}

                def answer(self, history, recent, question, direct=False, max_tokens=16, context_tokens=None):
                    answered.append((question, list(history)))
                    return "B", dict(input_tokens=1, visual_tokens=1, text_tokens=0)

            class Router:
                def __init__(self, engine):
                    pass

                def route(self, question):
                    return dict(label="LONG" if question == "long" else "NOW")

            def official(engine, sample, media, question_text):
                confidence = 0.95 if sample.question == "now sure" else 0.3
                from PIL import Image
                return "A", dict(input_tokens=2, visual_tokens=2, text_tokens=0), confidence, [(sample.end - 0.5, Image.new("RGB", (32, 32)))]

            engine = SimpleNamespace(device="cpu", model=SimpleNamespace(
                config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))))
            engine_module = ModuleType("onestreamer.engine")
            engine_module.OneStreamerEngine = lambda **kwargs: self.fail("Unexpected model load")
            config = MemoryConfig(capacity=100000, store_tokens=100000, store_bytes=1 << 30, read_mode="routed",
                                  context_tokens=100000, history_compression="budget_pool",
                                  store_budget="context", min_frame_tokens=4, stream_include_end=True,
                                  route_threshold=0.8, route_escalation="SHORT", short_history_tokens=64)
            with patch.dict(sys.modules, {"onestreamer.engine": engine_module}), \
                    patch.object(visual_memory, "NativeFeatureAdapter", Adapter), \
                    patch.object(question_router, "QuestionRouter", Router), \
                    patch.object(question_router, "official_recent_answer", official):
                results = {r["sample_id"]: r for r in visual_memory.infer(job, rows, config, engine=engine)}

            sure, unsure, long = results["now-sure"], results["now-unsure"], results["long"]
            self.assertEqual((sure["prediction"], sure["history_read_frames"]), ("A", 0))
            self.assertEqual(sure["route"]["final_scope"], "NOW")
            self.assertEqual(sure["recent_sampling"], "official_end_aligned")
            self.assertEqual(unsure["route"]["final_scope"], "SHORT")
            self.assertEqual(unsure["prediction"], "B")
            self.assertLessEqual(unsure["history_visual_tokens"], 64)
            self.assertEqual(unsure["history_timestamps"][-1], unsure["recent_timestamps"][0] - 1)
            self.assertEqual(long["route"]["final_scope"], "LONG")
            self.assertEqual(long["history_timestamps"][0], 0.0)
            self.assertGreater(long["history_visual_tokens"], 64)
            self.assertEqual(len(answered), 2)


if __name__ == "__main__":
    unittest.main()

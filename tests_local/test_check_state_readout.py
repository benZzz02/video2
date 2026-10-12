"""CPU-only checks for diagnostic output integrity and protocol isolation."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "tools/check_state_readout.py"
spec = importlib.util.spec_from_file_location("state_check", SOURCE)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "video.mp4").write_bytes(b"fake-video")
        (self.root / "config.json").write_text("{}")
        (self.root / "tokenizer_config.json").write_text("{}")
        (self.root / "model.safetensors").write_bytes(b"fake-weights")
        rows = [dict(sample_id=f"sample-{i}", index=i, task="real_time_visual_perception", subtask="FPD",
                     video="video.mp4", media="/stale/path/video.mp4", start=0, end=10,
                     question="Which?", options={"A": "yes", "B": "no"}, answer="A", answer_text="yes",
                     eval_mode="mcq_letter") for i in range(2)]
        self.manifest = self.root / "samples.json"
        self.manifest.write_text(json.dumps(dict(model="test", samples=rows)))

    def args(self, *extras):
        return check.build_parser().parse_args([
            "--manifest", str(self.manifest), "--model-path", str(self.root),
            "--video-root", str(self.root), "--output", str(self.root / "output"), *extras])

    def test_preflight_selects_exact_ids_and_resolves_media_without_writes(self):
        args = self.args("--sample-ids", "sample-1", "sample-0")
        manifest, config = check.prepare(args)
        self.assertEqual([r["sample_id"] for r in manifest["samples"]], ["sample-1", "sample-0"])
        self.assertTrue(all(r["media"] == str(self.root / "video.mp4") for r in manifest["samples"]))
        self.assertEqual(config.read_mode, "state")
        self.assertEqual(config.recent_frames, 16)
        self.assertEqual(config.context_tokens, 131072)
        self.assertFalse(args.output.exists())
        with self.assertRaisesRegex(ValueError, "Unknown sample"):
            check.prepare(self.args("--sample-ids", "missing"))

    def test_preflight_rejects_existing_output_and_missing_weights(self):
        output = self.root / "output"
        output.mkdir()
        with self.assertRaisesRegex(FileExistsError, "never overwritten"):
            check.prepare(self.args())
        output.rmdir()
        (self.root / "model.safetensors").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "downloads are disabled"):
            check.prepare(self.args())

    def test_legacy_override_restores_method_and_version_after_failure(self):
        original = check.vm.NativeFeatureAdapter.build_state_chat
        version = check.vm.STATE_PROMPT_VERSION
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with check.probe_protocol("legacy_v2"):
                self.assertIs(check.vm.NativeFeatureAdapter.build_state_chat, check.legacy_state_chat)
                self.assertEqual(check.vm.STATE_PROMPT_VERSION, "native_readiness_v2")
                raise RuntimeError("interrupted")
        self.assertIs(check.vm.NativeFeatureAdapter.build_state_chat, original)
        self.assertEqual(check.vm.STATE_PROMPT_VERSION, version)

    def test_reports_degenerate_and_invalid_predictions_without_forcing_states(self):
        def row(state):
            return dict(task="test", hit=True, state_rounds=[dict(state=state, predicted_token="raw")],
                        state_stop_reason="test", state_probe_seconds=1, runtime_seconds=2)
        summary = check.summarize([row("</Silence>")])
        self.assertEqual(summary["warnings"], ["all_silence"])
        self.assertEqual(summary["diagnostic_status"], "suspect")
        summary = check.summarize([row(None), row("</Response>")])
        self.assertEqual(summary["state_counts"], {"invalid_state": 1, "</Response>": 1})
        self.assertEqual(summary["warnings"], ["invalid_state_predictions"])
        self.assertEqual(summary["state_probe_seconds"], 2)

    def test_one_engine_paired_outputs_and_no_overwrite(self):
        args = self.args()
        manifest, config = check.prepare(args)
        engines, calls = [], []

        def engine(**kwargs):
            obj = object()
            engines.append(obj)
            return obj

        def infer(job, rows, config, engine):
            legacy = check.vm.NativeFeatureAdapter.build_state_chat is check.legacy_state_chat
            calls.append((engine, [r["sample_id"] for r in rows], check.vm.STATE_PROMPT_VERSION))
            for row in rows:
                state = "</Silence>" if legacy else "</Response>"
                yield dict(row, prediction="B" if legacy else "A", hit=not legacy,
                           state_rounds=[dict(state=state, predicted_token=state)],
                           state_stop_reason="history_exhausted" if legacy else "response",
                           state_probe_seconds=1, runtime_seconds=2)

        with patch.dict(sys.modules, {"onestreamer.engine": SimpleNamespace(OneStreamerEngine=engine)}), \
             patch.dict(check.os.environ, {}, clear=False), patch.object(check.vm, "infer", infer), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(check.run(args, manifest, config), 0)
        self.assertEqual(len(engines), 1)
        self.assertTrue(all(call[0] is engines[0] for call in calls))
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual([call[2] for call in calls], ["native_readiness_v2", "native_response_prefix_v3"])
        result = json.loads((args.output / "comparison.json").read_text())
        self.assertEqual(result["paired"]["revised_gains"], 2)
        self.assertEqual(result["summaries"]["legacy_v2"]["warnings"], ["all_silence"])
        self.assertEqual(result["summaries"]["native_prefix_v3"]["warnings"], [])
        for protocol in ("legacy_v2", "native_prefix_v3"):
            self.assertEqual(len((args.output / protocol / "predictions.jsonl").read_text().splitlines()), 2)
        with self.assertRaises(FileExistsError):
            check.run(args, manifest, config)


if __name__ == "__main__":
    unittest.main()

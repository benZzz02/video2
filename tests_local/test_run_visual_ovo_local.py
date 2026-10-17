"""CPU-only regression tests for the local two-GPU supervisor."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "tools/run_visual_ovo_local.py"
spec = importlib.util.spec_from_file_location("local_visual_runner", MODULE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def sample(sid, video="a.mp4", start=0.0, end=10.0):
    return dict(sample_id=sid, media="/video/" + video, video=video, start=start, end=end,
                index=int(sid), task="backward_tracking", subtask="EPM", question="Which?",
                options={"A": "yes", "B": "no"}, answer="A", answer_text="yes", eval_mode="mcq_letter")


class PureTests(unittest.TestCase):
    def test_readout_cli_defaults_and_worker_configuration_match(self):
        parser = runner.build_parser()
        for flags, mode, budgets in [([], "fill", [1024, 4096, 16384]),
                                     (["--history-read-mode", "state", "--state-history-budgets", "512", "2048"],
                                      "state", [512, 2048]),
                                     (["--history-read-mode", "action"], "action", [1024, 4096, 16384])]:
            args = parser.parse_args(["--output", "/tmp/test-output", *flags])
            config = runner.configuration(args)
            self.assertEqual(config["read_mode"], mode)
            self.assertEqual(config["answer_prompt"], "standard")
            self.assertEqual(config["state_history_budgets"], budgets)
            self.assertEqual(config["context_tokens"], 131072)
            self.assertEqual(config, json.loads(json.dumps(config)))
            cmd = runner.command(args, "/tmp/samples.json", args.output)
            self.assertEqual(cmd[cmd.index("--history-read-mode") + 1], mode)
            self.assertEqual(cmd[cmd.index("--answer-prompt") + 1], "standard")
            self.assertEqual(cmd[cmd.index("--state-history-budgets") + 1:], list(map(str, budgets)))
        args = parser.parse_args(["--output", "/tmp/test-output", "--answer-prompt", "frame_id_witness"])
        self.assertEqual(runner.configuration(args)["answer_prompt"], "frame_id_witness")
        cmd = runner.command(args, "/tmp/samples.json", args.output)
        self.assertEqual(cmd[cmd.index("--answer-prompt") + 1], "frame_id_witness")

    def test_invalid_state_budgets_fail_before_manifest_or_gpu_access(self):
        for mode in ("state", "action"):
            for budgets in [[], [0], [-1], [1024, 1024], [4096, 1024]]:
                with self.subTest(mode=mode, budgets=budgets), self.assertRaises(ValueError):
                    runner.configuration(SimpleNamespace(history_read_mode=mode, state_history_budgets=budgets))

    def test_witness_prompt_rejects_incompatible_read_modes_before_gpu_access(self):
        for mode in ("state", "action"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "requires read_mode=fill"):
                runner.configuration(SimpleNamespace(history_read_mode=mode, answer_prompt="frame_id_witness"))

    def test_untracked_controller_is_in_source_identity(self):
        names = ["Eval/Proactive_Eval/OneStreamer/src/onestreamer/" + name
                 for name in ("state_readout.py", "action_readout.py", "frame_evidence.py", "question_router.py")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in names:
                controller = root / name
                controller.parent.mkdir(parents=True, exist_ok=True)
                controller.write_text("controller_v1\n")
            with patch.object(runner, "ROOT", root), patch.object(runner.subprocess, "check_output", return_value=b""):
                original = runner.source_hashes()
                for name in names:
                    (root / name).write_text("controller_v2\n")
                    changed = runner.source_hashes()
                    self.assertIn(name, original)
                    self.assertNotEqual(original[name], changed[name])
                    self.assertNotEqual(runner.digest(original), runner.digest(changed))
                    original = changed

    def test_readout_options_participate_in_prepared_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.mp4").write_bytes(b"video")
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "weights.bin"}}))
            (root / "weights.bin").write_bytes(b"weights")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"samples": [sample("0")]}))
            args = SimpleNamespace(manifest=manifest, video_root=root, model_path=root,
                                   python=sys.executable, history_read_mode="fill",
                                   state_history_budgets=[1024, 4096, 16384])

            def output(command, **kwargs):
                return "test-commit\n" if command[:2] == ["git", "rev-parse"] else '{"python":"test", "packages":{}}'

            with patch.object(runner, "source_hashes", return_value={"state_readout.py": "test-sha"}), \
                    patch.object(runner.subprocess, "check_output", side_effect=output):
                baseline = runner.prepare(args)
                args.history_read_mode = "state"
                state = runner.prepare(args)
                args.history_read_mode = "action"
                action = runner.prepare(args)
                args.state_history_budgets = [512, 2048]
                different_budget = runner.prepare(args)
                args.history_read_mode = "fill"
                fill_budget = runner.prepare(args)
                args.answer_prompt = "frame_id_witness"
                witness = runner.prepare(args)
            self.assertEqual(state["identity"]["config"]["read_mode"], "state")
            self.assertEqual(action["identity"]["config"]["read_mode"], "action")
            self.assertEqual(different_budget["identity"]["config"]["state_history_budgets"], [512, 2048])
            self.assertEqual(witness["identity"]["config"]["answer_prompt"], "frame_id_witness")
            self.assertNotEqual(baseline["fingerprint"], state["fingerprint"])
            self.assertNotEqual(state["fingerprint"], action["fingerprint"])
            self.assertNotEqual(action["fingerprint"], different_budget["fingerprint"])
            self.assertNotEqual(fill_budget["fingerprint"], witness["fingerprint"])

    def test_grouping_and_balance_are_order_independent(self):
        rows = [sample("0", end=30), sample("1", end=40), sample("2", "b.mp4", end=30),
                sample("3", "c.mp4", end=10), sample("4", start=3, end=8)]
        shards, loads = runner.balanced_shards(rows)
        self.assertEqual((shards, loads), runner.balanced_shards(list(reversed(rows))))
        self.assertTrue(any({"0", "1"}.issubset(s) for s in map(set, shards)))
        self.assertEqual(sum(loads), 85)
        self.assertLessEqual(abs(loads[0] - loads[1]), 5)

    def test_retry_only_terminal_io_errors(self):
        for terminal in ["RuntimeError: Failed to decode source frame 632", "ValueError: Cannot open video: /nas/x.mp4",
                         "OSError: [Errno 5] Input/output error", "OSError: [Errno 116] Stale file handle"]:
            self.assertTrue(runner.retryable(1, "Traceback (most recent call last):\n" + terminal))
        for code, text in [(1, "torch.OutOfMemoryError: CUDA out of memory"),
                           (1, "ValueError: context budget exceeded"),
                           (1, "RuntimeError: Failed to decode source frame 1\nTypeError: later failure"),
                           (-15, "RuntimeError: Failed to decode source frame 1"),
                           (137, "RuntimeError: Failed to decode source frame 1"),
                           (1, "ValueError: Invalid video metadata: x"),
                           (1, "RuntimeError: generic failure")]:
            self.assertFalse(runner.retryable(code, text))

    def test_resume_rejects_changed_identity_and_corrupted_manifest(self):
        original = {"identity": {"code": "a", "config": runner.CONFIG}}
        original["fingerprint"] = runner.digest(original["identity"])
        runner.verify_identity(original, original)
        changed = copy.deepcopy(original)
        changed["identity"]["code"] = "b"
        changed["fingerprint"] = runner.digest(changed["identity"])
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            runner.verify_identity(original, changed)
        changed["fingerprint"] = "broken"
        with self.assertRaisesRegex(ValueError, "fingerprint is invalid"):
            runner.verify_identity(changed, original)

    def test_merge_keeps_action_decisions_and_executor_stops_separate(self):
        rows = [dict(sample(str(i)), model_alias="OneStreamer-4B", variant="fake_visual",
                     status="ok", prediction="A", hit=True, observation_seconds_delta=1,
                     runtime_seconds=2, total_runtime_seconds=3) for i in range(3)]
        rows[0].update(action_rounds=[{"action": "STOP"}], action_probe_seconds=.5,
                       action_invalid_count=0, action_stop_reason="model_stop")
        rows[1].update(action_rounds=[{"action": None}, {"action": "READ_MORE"}],
                       action_probe_seconds=1.5, action_invalid_count=1,
                       action_stop_reason="history_budget_exhausted")
        # The executor may finish without invoking the model when no history
        # is available. It must not fabricate a STOP action for that sample.
        rows[2].update(action_rounds=[], action_probe_seconds=0,
                       action_invalid_count=0, action_stop_reason="history_exhausted")
        manifest = dict(identity=dict(samples=rows, shards=[["0", "2"], ["1"]]), fingerprint="test")
        state = {}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "collect", return_value=({r["sample_id"]: r for r in rows}, {})):
            output = Path(directory)
            runner.merge(output, manifest, state)
            summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["action_probe_seconds"], 2)
        self.assertEqual(summary["action_probe_count"], 3)
        self.assertEqual(summary["action_invalid_count"], 1)
        self.assertEqual(summary["action_counts"], {"STOP": 1, "READ_MORE": 1, "invalid_action": 1})
        self.assertEqual(summary["action_stop_reasons"], {
            "model_stop": 1, "history_budget_exhausted": 1, "history_exhausted": 1})
        self.assertEqual(summary["state_probe_count"], 0)
        self.assertEqual(summary["state_stop_reasons"], {})


# This child stands in for run_visual_memory.sh, exercises real subprocess and
# signal cleanup, and never imports torch or accesses a GPU.
FAKE_WORKER = r'''
import hashlib, json, pathlib, sys, time
samples, out, mode, config, job = sys.argv[1:]
out = pathlib.Path(out)
out.mkdir()
rows = json.loads(pathlib.Path(samples).read_text())["samples"]
manifest = dict(samples=rows, config=json.loads(config), job=json.loads(job))
fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
manifest["fingerprint"] = fingerprint
(out / "manifest.json").write_text(json.dumps(manifest))
if mode == "sleep":
    time.sleep(30)
first = out.name.endswith("attempt-001")
if mode == "context":
    print("ValueError: Recent frames exceed context budget", flush=True)
    sys.exit(1)
if mode == "ioalways":
    print("OSError: [Errno 5] Input/output error", flush=True)
    sys.exit(1)
with (out / "predictions.jsonl").open("w") as f:
    chosen = rows[:1] if mode == "retry" and first else rows
    for r in chosen:
        r.update(model_alias="OneStreamer-4B",variant="fake_visual",status="ok",fingerprint=fingerprint,
                 prediction="A",hit=True,observation_seconds_delta=1,runtime_seconds=1,total_runtime_seconds=2,
                 state_probe_seconds=.25*(r["index"]+1),state_rounds=[{}]*(1+r["index"]%2),
                 state_stop_reason="response" if r["index"]%2 == 0 else "history_exhausted")
        f.write(json.dumps(r)+"\n")
    if mode == "retry" and first:
        f.write('{"sample_id":')
if mode == "retry" and first:
    print("RuntimeError: Failed to decode source frame 7", flush=True)
    sys.exit(1)
if mode == "progress":
    time.sleep(30)
'''


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        for name in ("inputs", "attempts", "logs"):
            (self.output / name).mkdir()
        self.rows = [sample(str(i), f"{i}.mp4") for i in range(4)]
        self.job = dict(model="OneStreamer-4B", model_path="/fake", device="cuda", dtype="bfloat16",
                        attn_implementation="sdpa", video_root="/video")
        identity = dict(samples=self.rows, shards=[["0", "2"], ["1", "3"]], config=runner.CONFIG, job=self.job)
        self.manifest = dict(identity=identity, fingerprint=runner.digest(identity))
        self.state = dict(attempts=[], run_fingerprint=self.manifest["fingerprint"])
        self.args = type("Args", (), dict(output=self.output))()
        self.lock = (self.output / ".lock").open("a+")
        self.addCleanup(self.lock.close)

    def run_fake(self, mode):
        def command(args, samples, output):
            return [sys.executable, "-c", FAKE_WORKER, str(samples), str(output), mode,
                    json.dumps(runner.CONFIG), json.dumps(self.job)]
        with patch.object(runner, "command", command), patch.object(runner, "environment", lambda *_: os.environ.copy()), \
                patch.object(runner, "RETRY_DELAY", 0):
            return runner.supervise(self.args, self.manifest, self.state, self.lock.fileno())

    def test_partial_io_retry_preserves_original_predictions_and_fingerprints(self):
        self.assertEqual(self.run_fake("retry"), 0)
        attempts = self.state["attempts"]
        self.assertEqual([a["number"] for a in attempts], [1, 1, 2, 2])
        self.assertTrue(all(len(a["sample_ids"]) == 1 for a in attempts if a["number"] == 2))
        rows = [json.loads(line) for line in (self.output / "predictions.jsonl").read_text().splitlines()]
        self.assertEqual([r["sample_id"] for r in rows], ["0", "1", "2", "3"])
        provenance = json.loads((self.output / "prediction_sources.json").read_text())
        self.assertTrue(provenance["0"]["attempt"].endswith("attempt-001"))
        self.assertTrue(provenance["2"]["attempt"].endswith("attempt-002"))
        self.assertTrue(all(r["fingerprint"] == provenance[r["sample_id"]]["fingerprint"] for r in rows))
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["rows"], 4)
        self.assertEqual(summary["state_probe_seconds"], 2.5)
        self.assertEqual(summary["state_probe_count"], 6)
        self.assertEqual(summary["state_stop_reasons"], {"response": 2, "history_exhausted": 2})
        self.assertEqual(summary["action_probe_count"], 0)
        self.assertEqual(summary["action_invalid_count"], 0)
        self.assertEqual(summary["action_counts"], {})
        self.assertEqual(summary["action_stop_reasons"], {})
        self.assertEqual(self.state["per_shard_completed"], [2, 2])
        # Tampering with a source attempt is rejected on the next resume/merge.
        path = self.output / attempts[0]["directory"] / "predictions.jsonl"
        text = path.read_text().replace('"fingerprint": "', '"fingerprint": "bad')
        path.write_text(text)
        with self.assertRaisesRegex(ValueError, "Invalid prediction identity"):
            runner.collect(self.output, self.manifest, self.state)

    def test_context_failure_never_retries(self):
        with self.assertRaisesRegex(RuntimeError, "failed; inspect"):
            self.run_fake("context")
        self.assertTrue(all(a["number"] == 1 for a in self.state["attempts"]))
        self.assertEqual(self.state["status"], "failed")
        self.assertTrue(all(a["status"] != "running" for a in self.state["attempts"]))

    def test_manual_interruptions_do_not_exhaust_io_retry_budget(self):
        # Each shard was intentionally interrupted three times before producing
        # a record. A fourth unique attempt must still be allowed on resume.
        self.state["attempts"] = [dict(shard=shard, number=number, status="interrupted",
                                      directory=f"attempts/shard-{shard:03d}-attempt-{number:03d}",
                                      sample_ids=self.manifest["identity"]["shards"][shard])
                                  for number in range(1, 4) for shard in range(2)]
        self.assertEqual(self.run_fake("success"), 0)
        self.assertEqual([a["number"] for a in self.state["attempts"][-2:]], [4, 4])
        self.assertEqual(self.state["completed"], 4)

    def test_third_io_failure_stops_further_attempts_including_resume(self):
        with self.assertRaisesRegex(RuntimeError, "failed; inspect"):
            self.run_fake("ioalways")
        failures = [a for a in self.state["attempts"] if a["shard"] == 0 and a.get("retryable")]
        self.assertEqual(len(failures), 3)
        count = len(self.state["attempts"])
        with self.assertRaisesRegex(RuntimeError, "exhausted its budget"):
            self.run_fake("success")
        self.assertEqual(len(self.state["attempts"]), count)

    def test_sigterm_cleans_own_worker_groups(self):
        timer = threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM))
        timer.start()
        try:
            self.assertEqual(self.run_fake("sleep"), 143)
        finally:
            timer.cancel()
        self.assertEqual(self.state["status"], "interrupted")
        for attempt in self.state["attempts"]:
            self.assertEqual(attempt["status"], "interrupted")
            with self.assertRaises(ProcessLookupError):
                os.kill(attempt["pid"], 0)

    def test_progress_updates_while_workers_are_still_running(self):
        timer = threading.Timer(1.5, lambda: os.kill(os.getpid(), signal.SIGTERM))
        timer.start()
        try:
            with patch.object(runner, "PROGRESS_INTERVAL", 0.1):
                self.assertEqual(self.run_fake("progress"), 143)
        finally:
            timer.cancel()
        self.assertEqual(self.state["completed"], 4)
        self.assertEqual(self.state["per_shard_completed"], [2, 2])
        self.assertTrue(all(a["status"] == "interrupted" for a in self.state["attempts"]))


if __name__ == "__main__":
    unittest.main()

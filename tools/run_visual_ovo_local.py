#!/usr/bin/env python3
"""Two-GPU local OVO runner for fill, state, or action visual-memory readout.

Use --dry-run to validate, then run with --output results/RUN. Restart with
--resume and the same output. Successful answers retain their attempt fingerprint.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Eval/Proactive_Eval/OneStreamer/src"
sys.path.insert(0, str(SOURCE))
from onestreamer.schema import Sample
from onestreamer.visual_memory import MemoryConfig

DEFAULT_MANIFEST = Path("/mnt/truenas_nfs/dataset/znh1302/OneStreamer-OVO-eval/outputs/OneStreamer-4B/ovobench/ovo-full-memory-2gpu-20261004/memory/ovobench/manifest.json")
DEFAULT_MODEL = Path("/mnt/truenas_nfs/dataset/znh1302/OneStreamer")
DEFAULT_VIDEOS = Path("/mnt/truenas_nfs/dataset/ovo-bench")
DEFAULT_PYTHON = "/home/zhangnuohua/miniconda3/envs/vllm/bin/python"
OVERLAY = "/home/zhangnuohua/medhorizon/OneStreamer/.runtime/ovo"
GPUS = ("GPU-82943a1a-f610-7e0b-e236-e55e9ba80eb4", "GPU-ce291af9-8e98-caf1-7ab5-91911cef71cd")
MAX_IO_FAILURES = 3
RETRY_DELAY = 30
PROGRESS_INTERVAL = 30
DEFAULT_STATE_HISTORY_BUDGETS = (1024, 4096, 16384)
# JSON-native values also match the per-attempt manifests before a first resume.
CONFIG = json.loads(json.dumps(asdict(MemoryConfig(
    capacity=16384, read_mode="fill", context_tokens=131072,
    history_compression="progressive_pool", recent_frames=16))))


def configuration(args):
    values = dict(CONFIG)
    values.update(read_mode=getattr(args, "history_read_mode", "fill"),
                  state_history_budgets=tuple(getattr(
                      args, "state_history_budgets", DEFAULT_STATE_HISTORY_BUDGETS)))
    return json.loads(json.dumps(asdict(MemoryConfig(**values))))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "Eval", "run_visual_memory.sh"], cwd=ROOT).decode().split("\0")
    sources = {name: file_sha(ROOT / name) for name in tracked if name and (ROOT / name).is_file()}
    # New local modules must participate in resume identity before a git add.
    for name in ("state_readout.py", "action_readout.py"):
        controller = "Eval/Proactive_Eval/OneStreamer/src/onestreamer/" + name
        sources[controller] = file_sha(ROOT / controller)
    sources["tools/run_visual_ovo_local.py"] = file_sha(Path(__file__).resolve())
    return sources


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def balanced_shards(rows):
    """Keep each resolved video/start stream together, using longest-first packing."""
    groups = {}
    for row in rows:
        key = (row["media"], row["start"])
        groups.setdefault(key, []).append(row)
    weighted = [(max(r["end"] for r in items) - key[1], key, items)
                for key, items in groups.items()]
    shards, loads = [[], []], [0.0, 0.0]
    for cost, key, items in sorted(weighted, key=lambda x: (-x[0], x[1])):
        target = min(range(2), key=lambda i: (loads[i], i))
        shards[target].extend(r["sample_id"] for r in sorted(items, key=lambda r: (r["end"], r["sample_id"])))
        loads[target] += cost
    return shards, loads


def prepare(args):
    config = configuration(args)
    raw = args.manifest.read_bytes()
    original = json.loads(raw)
    rows = []
    for row in original["samples"]:
        sample = Sample.from_dict(row)
        if not all(math.isfinite(x) for x in (sample.start, sample.end)) or sample.start < 0:
            raise ValueError(f"Invalid stream bounds: {sample.sample_id}")
        path = (args.video_root / sample.video).resolve()
        if not path.is_file():
            raise ValueError(f"Missing video: {path}")
        rows.append(dict(row, **asdict(sample), media=str(path)))
    ids = [r["sample_id"] for r in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Manifest must contain nonempty, unique sample IDs")
    shards, loads = balanced_shards(rows)
    sources = source_hashes()
    model_index = args.model_path / "model.safetensors.index.json"
    weights = sorted(set(json.loads(model_index.read_text())["weight_map"].values()))
    model_files = sorted(set(weights + [p.name for p in args.model_path.glob("*.json")] + [p.name for p in args.model_path.glob("*.jinja")]))
    model_stats = {}
    for name in model_files:
        path = args.model_path / name
        stat = path.stat()
        model_stats[name] = [stat.st_size, stat.st_mtime_ns,
                             file_sha(path) if stat.st_size < 20 * 1024**2 else None]
    media_stats = {}
    for path in sorted({r["media"] for r in rows}):
        stat = Path(path).stat()
        media_stats[path] = [stat.st_size, stat.st_mtime_ns]
    job = dict(model=original.get("model", args.model_path.name), model_path=str(args.model_path),
               device="cuda", dtype="bfloat16", attn_implementation="sdpa", video_root=str(args.video_root))
    versions = json.loads(subprocess.check_output([
        args.python, "-c", "import json,sys; from importlib.metadata import version; "
        "print(json.dumps(dict(python=sys.version, packages={p: version(p) for p in "
        "['torch','torchvision','transformers','accelerate','numpy','Pillow','opencv-python-headless','pandas']})))"
    ], env=environment(args, 0), text=True))
    identity = dict(version=1, repo_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    code_sha256=digest(sources), source_files=sources,
                    source_manifest=str(args.manifest), source_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                    model_files=model_stats, media_metadata_sha256=digest(media_stats), config=config, job=job,
                    python=args.python, dependency_versions=versions, dependency_overlay=OVERLAY, gpu_uuids=list(GPUS),
                    environment={"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
                                 "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1"},
                    shards=shards, shard_video_seconds=loads, samples=rows,
                    retry_budget={"max_io_failures_per_shard": MAX_IO_FAILURES,
                                  "scope": "entire run including resume",
                                  "manual_interruptions_consume_budget": False})
    return dict(identity=identity, fingerprint=digest(identity), created_at=utc_now(),
                results={"predictions": "predictions.jsonl", "summary": "summary.json",
                         "prediction_sources": "prediction_sources.json", "status": "status.json",
                         "attempts": "attempts/shard-NNN-attempt-NNN", "logs": "logs/shard-NNN.log"})


def verify_identity(existing, planned):
    if existing.get("fingerprint") != digest(existing.get("identity")):
        raise ValueError("Stored manifest fingerprint is invalid")
    if existing["fingerprint"] != planned["fingerprint"]:
        raise ValueError("Resume identity mismatch: source, code, model, media, configuration or shard assignment changed")


def retryable(code, text):
    if not 0 < code < 128:
        return False
    if re.search(r"out of memory|OutOfMemoryError|context|fingerprint|KeyboardInterrupt|SystemExit|segmentation fault", text, re.I):
        return False
    last = text.rstrip().splitlines()[-1] if text.strip() else ""
    return bool(re.search(r"^(?:RuntimeError|ValueError|OSError|IOError|TimeoutError): .*"
                          r"(?:failed to (?:decode|read|open).*(?:frame|video)|cannot open video|"
                          r"(?:input/output|I/O) error|stale file handle|(?:connection|read|operation) timed out|stream timeout)", last, re.I))


def read_tail(path):
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 128 * 1024))
        return handle.read().decode("utf-8", errors="replace")


def collect(output, manifest, state):
    """Validate every accepted answer against its untouched attempt manifest."""
    source = {r["sample_id"]: r for r in manifest["identity"]["samples"]}
    completed, provenance = {}, {}
    for attempt in state["attempts"]:
        directory = output / attempt["directory"]
        predictions = directory / "predictions.jsonl"
        if not predictions.exists():
            continue
        metadata = json.loads((directory / "manifest.json").read_text())
        fingerprint = metadata.pop("fingerprint")
        if fingerprint != digest(metadata):
            raise ValueError(f"Attempt manifest fingerprint mismatch: {directory}")
        if metadata["config"] != manifest["identity"]["config"] or metadata["job"] != manifest["identity"]["job"]:
            raise ValueError(f"Attempt configuration mismatch: {directory}")
        attempt_ids = {r["sample_id"] for r in metadata["samples"]}
        if attempt_ids != set(attempt["sample_ids"]):
            raise ValueError(f"Attempt sample assignment mismatch: {directory}")
        for row in metadata["samples"]:
            if row != source.get(row["sample_id"]):
                raise ValueError(f"Attempt source sample changed: {row['sample_id']}")
        lines = predictions.read_bytes().splitlines(keepends=True)
        for i, line in enumerate(lines):
            # An interrupted write can leave one incomplete final record. Keep
            # the original file for audit and retry that sample in a new attempt.
            if not line.endswith(b"\n") and i == len(lines) - 1 and attempt["status"] != "complete":
                continue
            row = json.loads(line)
            sid = row["sample_id"]
            if sid not in attempt_ids or sid in completed:
                raise ValueError(f"Unexpected or duplicate completed sample: {sid}")
            if row.get("status") != "ok" or row.get("fingerprint") != fingerprint:
                raise ValueError(f"Invalid prediction identity: {sid}")
            if any(row.get(k) != v for k, v in source[sid].items()):
                raise ValueError(f"Prediction source sample changed: {sid}")
            completed[sid] = row
            provenance[sid] = {"attempt": attempt["directory"], "fingerprint": fingerprint}
    return completed, provenance


def command(args, samples, output):
    config = configuration(args)
    return ["bash", str(ROOT / "run_visual_memory.sh"), "--manifest", str(samples),
            "--model-path", str(args.model_path), "--video-root", str(args.video_root),
            "--output", str(output), "--context-tokens", str(config["context_tokens"]),
            "--attn-implementation", "sdpa", "--history-read-mode", config["read_mode"],
            "--state-history-budgets", *map(str, config["state_history_budgets"])]


def environment(args, shard):
    env = os.environ.copy()
    env.update(EVAL_PYTHON=args.python, PYTHONPATH=OVERLAY, CUDA_VISIBLE_DEVICES=GPUS[shard],
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    return env


def merge(output, manifest, state):
    rows, provenance = collect(output, manifest, state)
    expected = [r["sample_id"] for r in manifest["identity"]["samples"]]
    if set(rows) != set(expected):
        raise ValueError(f"Cannot aggregate incomplete run: {len(rows)}/{len(expected)}")
    ordered = [rows[sid] for sid in expected]
    sys.path.insert(0, OVERLAY)
    from onestreamer.metrics import aggregate
    summary = aggregate(ordered)
    summary.update(observation_seconds=sum(r["observation_seconds_delta"] for r in ordered),
                   query_seconds=sum(r["runtime_seconds"] for r in ordered),
                   total_measured_seconds=sum(r["total_runtime_seconds"] for r in ordered),
                   state_probe_seconds=sum(r.get("state_probe_seconds", 0.0) for r in ordered),
                   state_probe_count=sum(len(r.get("state_rounds", [])) for r in ordered),
                   state_stop_reasons=dict(Counter(r["state_stop_reason"] for r in ordered
                                                   if r.get("state_stop_reason") is not None)),
                   action_probe_seconds=sum(r.get("action_probe_seconds", 0.0) for r in ordered),
                   action_probe_count=sum(len(r.get("action_rounds", [])) for r in ordered),
                   action_invalid_count=sum(r.get("action_invalid_count", 0) for r in ordered),
                   action_counts=dict(Counter(step.get("action") or "invalid_action"
                                              for r in ordered for step in r.get("action_rounds", []))),
                   action_stop_reasons=dict(Counter(r["action_stop_reason"] for r in ordered
                                                    if r.get("action_stop_reason") is not None)),
                   run_fingerprint=manifest["fingerprint"], completed_at=utc_now())
    temporary = output / "predictions.jsonl.tmp"
    with temporary.open("w") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output / "predictions.jsonl")
    write_json(output / "prediction_sources.json", provenance)
    write_json(output / "summary.json", summary)
    state.update(status="complete", completed=len(rows), completed_at=utc_now(),
                 per_shard_completed=[len(ids) for ids in manifest["identity"]["shards"]])
    write_json(output / "status.json", state)
    print(f"Completed and aggregated {len(rows)} predictions: {output / 'summary.json'}", flush=True)


def supervise(args, manifest, state, lock_fd):
    output = args.output
    source = {r["sample_id"]: r for r in manifest["identity"]["samples"]}
    active = {}
    stopping = [0]
    old_handlers = {}

    def signal_group(process, signum):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    def stop(signum, frame):
        if not stopping[0]:
            stopping[0] = signum
            for process, attempt, handle in active.values():
                signal_group(process, signal.SIGTERM)

    def persist():
        state["updated_at"] = utc_now()
        write_json(output / "status.json", state)

    def progress(completed):
        state["completed"] = len(completed)
        state["per_shard_completed"] = [sum(sid in completed for sid in ids)
                                         for ids in manifest["identity"]["shards"]]

    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, stop)
    try:
        completed, _ = collect(output, manifest, state)
        for attempt in state["attempts"]:
            if attempt["status"] == "running":
                attempt["status"] = "interrupted"
        state.update(status="running", supervisor_pid=os.getpid())
        progress(completed)
        persist()
        ready_at = [0.0, 0.0]
        last_progress_at = time.monotonic()
        while not stopping[0]:
            for shard, (process, attempt, handle) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del active[shard]
                attempt.update(returncode=code, ended_at=utc_now(), status="complete" if code == 0 else "failed")
                completed, _ = collect(output, manifest, state)
                progress(completed)
                remaining = set(attempt["sample_ids"]) - set(completed)
                if code == 0 and remaining:
                    raise RuntimeError(f"Shard {shard} exited successfully with {len(remaining)} missing predictions")
                if code != 0:
                    can_retry = retryable(code, read_tail(output / attempt["log"]))
                    attempt["retryable"] = can_retry
                    print(f"Shard {shard} attempt {attempt['number']} exit {code}; retryable={can_retry}", flush=True)
                    failures = sum(a["shard"] == shard and a["status"] == "failed" and a.get("retryable", False)
                                   for a in state["attempts"])
                    if not can_retry or failures >= MAX_IO_FAILURES:
                        raise RuntimeError(f"Shard {shard} failed; inspect {output / attempt['log']}")
                    ready_at[shard] = time.monotonic() + RETRY_DELAY
                persist()
            if time.monotonic() - last_progress_at >= PROGRESS_INTERVAL:
                completed, _ = collect(output, manifest, state)
                progress(completed)
                persist()
                last_progress_at = time.monotonic()
            if len(completed) == len(source) and not active:
                merge(output, manifest, state)
                return 0
            for shard, ids in enumerate(manifest["identity"]["shards"]):
                remaining = [sid for sid in ids if sid not in completed]
                if shard in active or not remaining or time.monotonic() < ready_at[shard]:
                    continue
                previous = [a for a in state["attempts"] if a["shard"] == shard]
                if previous and previous[-1]["status"] == "failed" and not previous[-1].get("retryable"):
                    raise RuntimeError(f"Shard {shard} has a non-retryable failure; no automatic relaunch")
                number = len(previous) + 1
                failures = sum(a["status"] == "failed" and a.get("retryable", False) for a in previous)
                if failures >= MAX_IO_FAILURES:
                    raise RuntimeError(f"Shard {shard} exhausted its budget of {MAX_IO_FAILURES} I/O failures")
                name = f"shard-{shard:03d}-attempt-{number:03d}"
                inputs = output / "inputs" / (name + ".json")
                directory = output / "attempts" / name
                log = output / "logs" / (name + ".log")
                write_json(inputs, {"model": manifest["identity"]["job"]["model"], "samples": [source[sid] for sid in remaining]})
                attempt = dict(shard=shard, number=number, directory=str(directory.relative_to(output)),
                               log=str(log.relative_to(output)), sample_ids=remaining, started_at=utc_now(), status="running")
                state["attempts"].append(attempt)
                persist()
                latest = output / "logs" / f"shard-{shard:03d}.log"
                if latest.is_symlink():
                    latest.unlink()
                latest.symlink_to(log.name)
                handle = log.open("x")
                try:
                    process = subprocess.Popen(command(args, inputs, directory), cwd=ROOT, env=environment(args, shard),
                                               stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                                               pass_fds=(lock_fd,))
                except BaseException:
                    handle.close()
                    attempt["status"] = "failed"
                    persist()
                    raise
                active[shard] = (process, attempt, handle)
                attempt["pid"] = process.pid
                persist()
                print(f"Shard {shard}: {len(remaining)} samples, attempt {number}, prior I/O failures={failures}/{MAX_IO_FAILURES}, pid={process.pid}, GPU={GPUS[shard]}, log={log}", flush=True)
            time.sleep(1)
        state["status"] = "interrupted"
        return 128 + stopping[0]
    except BaseException as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        for process, attempt, handle in active.values():
            signal_group(process, signal.SIGTERM)
        deadline = time.monotonic() + 20
        for process, attempt, handle in active.values():
            try:
                process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                signal_group(process, signal.SIGKILL)
                process.wait()
            handle.close()
            attempt.update(status="interrupted", ended_at=utc_now(), returncode=process.returncode)
        persist()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--python", default=os.environ.get("EVAL_PYTHON", DEFAULT_PYTHON))
    parser.add_argument("--history-read-mode", choices=("fill", "state", "action"), default="fill",
                        help="Fill context, use native response states, or explicitly decide whether to read more history")
    parser.add_argument("--state-history-budgets", type=int, nargs="+",
                        default=DEFAULT_STATE_HISTORY_BUDGETS, metavar="TOKENS",
                        help="Strictly increasing positive historical visual-token budgets after the recent-only probe")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        configuration(args)
    except ValueError as error:
        parser.error(str(error))
    for name in ("output", "manifest", "model_path", "video_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    planned = prepare(args)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        verify_identity(existing, planned)
        if not args.resume and not args.dry_run:
            raise ValueError("Output already contains a run; pass --resume")
    elif args.output.exists() and any(p.name != ".supervisor.lock" for p in args.output.iterdir()):
        raise ValueError("Output is nonempty and has no valid run manifest")
    if args.dry_run:
        validation = subprocess.run(command(args, args.manifest, args.output) + ["--dry-run"],
                                    cwd=ROOT, env=environment(args, 0), check=True, capture_output=True, text=True)
        print(validation.stdout.strip())
        print(json.dumps(dict(fingerprint=planned["fingerprint"], repo_commit=planned["identity"]["repo_commit"],
                              shard_samples=list(map(len, planned["identity"]["shards"])),
                              shard_video_seconds=planned["identity"]["shard_video_seconds"],
                              gpu_uuids=list(GPUS), output=str(args.output)), indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".supervisor.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another supervisor or its workers hold this run's lock") from None
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            verify_identity(manifest, planned)
            state = json.loads((args.output / "status.json").read_text())
        else:
            manifest = planned
            write_json(manifest_path, manifest)
            state = dict(status="prepared", run_fingerprint=manifest["fingerprint"], attempts=[], completed=0)
            write_json(args.output / "status.json", state)
        if state["run_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Status/manifest fingerprint mismatch")
        for directory in ("inputs", "attempts", "logs"):
            (args.output / directory).mkdir(exist_ok=True)
        return supervise(args, manifest, state, lock.fileno())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Visual OVO supervisor failed: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)

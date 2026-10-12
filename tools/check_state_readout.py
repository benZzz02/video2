#!/usr/bin/env python3
"""Compare legacy state classification with the native response-prefix probe.

This small diagnostic loads one local checkpoint and replays the same samples
with fresh visual banks for each protocol. It never trains, downloads weights,
or generates caption memory. Exit 2 means the revised protocol is suspect;
state diversity alone does not establish correct memory routing.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Eval/Proactive_Eval/OneStreamer/src"
sys.path.insert(0, str(SOURCE))
from onestreamer import visual_memory as vm
from onestreamer.schema import Sample

LEGACY_SYSTEM = (
    "Decide whether the supplied timestamped visual evidence is sufficient to answer the user's question. "
    "HISTORICAL_VISUAL_MEMORY contains sparse earlier observations; RECENT_VISUAL_CONTEXT contains "
    "the recent scene. Respect timestamps and distinguish earlier states from the current state. "
    "Judge the evidence needed by this particular question. A question about the current scene can "
    "be answerable from recent observations alone. A question about earlier events may require "
    "earlier observations. A prediction asks for an inference from observed evidence, not for "
    "observation of future frames. A supported negative answer or zero count is still an answer. "
    "Missing historical frames alone do not establish that an event never happened. "
    "Output exactly one native state token: </Response> if the available information is sufficient "
    "to answer; </Standby> if relevant evidence is present but incomplete; </Silence> if the supplied "
    "observations contain no relevant evidence. Output only that token. Do not answer the question, "
    "describe the scene, or generate a caption. The question's answer-format instructions apply "
    "to the subsequent answer, not to this state decision."
)
LEGACY_SUFFIX = (
    "Do not answer the quoted question yet. Its letter, Yes/No, or number format applies "
    "only when an answer is requested later. For this turn, assess whether the visual "
    "evidence already supplied is enough to answer it. Emit exactly one state token: "
    "</Response> for sufficient evidence, </Standby> for relevant but incomplete evidence, "
    "or </Silence> for no relevant evidence. State token:"
)


def legacy_state_chat(self, history, recent, question, direct=False):
    chat = self.build_chat(history, recent, question, direct)
    chat[0]["content"] = [dict(type="text", text=LEGACY_SYSTEM)]
    chat[1]["content"][-1] = dict(type="text", text=(
        "<QUESTION_TO_ASSESS>\n" + question + "\n</QUESTION_TO_ASSESS>\n" + LEGACY_SUFFIX))
    return chat


@contextmanager
def probe_protocol(name):
    original = vm.NativeFeatureAdapter.build_state_chat
    original_version = vm.STATE_PROMPT_VERSION
    if name == "legacy_v2":
        vm.NativeFeatureAdapter.build_state_chat = legacy_state_chat
        vm.STATE_PROMPT_VERSION = "native_readiness_v2"
    elif name != "native_prefix_v3":
        raise ValueError(f"Unknown protocol: {name}")
    try:
        yield
    finally:
        vm.NativeFeatureAdapter.build_state_chat = original
        vm.STATE_PROMPT_VERSION = original_version


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample-ids", nargs="+")
    selection.add_argument("--limit", type=int, default=12,
                           help="First N manifest rows when --sample-ids is absent (default: 12)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and selection without loading a model")
    return parser


def prepare(args):
    for name in ("manifest", "model_path", "video_root", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output.exists():
        raise FileExistsError(f"Choose a new --output; existing paths are never overwritten: {args.output}")
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if vm.STATE_PROMPT_VERSION != "native_response_prefix_v3":
        raise ValueError("This comparison requires the native_response_prefix_v3 adapter")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Input manifest is unavailable (check the data mount): {args.manifest}")
    rows, original = vm._load_rows(args.manifest)
    ids = [Sample.from_dict(row).sample_id for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Manifest must contain nonempty, unique sample IDs")
    if args.sample_ids is not None:
        if len(args.sample_ids) != len(set(args.sample_ids)):
            raise ValueError("--sample-ids must be unique")
        lookup = dict(zip(ids, rows))
        missing = set(args.sample_ids) - set(lookup)
        if missing:
            raise ValueError(f"Unknown sample IDs: {sorted(missing)}")
        rows = [lookup[sid] for sid in args.sample_ids]
    else:
        rows = rows[:args.limit]
    normalized = []
    for row in rows:
        sample = Sample.from_dict(row)
        if not all(math.isfinite(x) for x in (sample.start, sample.end)) or sample.start < 0:
            raise ValueError(f"Invalid stream bounds: {sample.sample_id}")
        media = (args.video_root / sample.video).resolve()
        if not media.is_file():
            raise FileNotFoundError(f"Video is unavailable (check --video-root or the data mount): {media}")
        normalized.append(dict(row, **asdict(sample), media=str(media)))
    required = [args.model_path / name for name in ("config.json", "tokenizer_config.json")]
    index = args.model_path / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if not shards:
            raise ValueError("Checkpoint index has no weight shards")
        required.extend(args.model_path / name for name in shards)
    else:
        required.append(args.model_path / "model.safetensors")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Local checkpoint file is unavailable; downloads are disabled: {path}")
    config = vm.MemoryConfig(capacity=16384, read_mode="state", context_tokens=131072,
                             history_compression="progressive_pool", recent_frames=16)
    job = dict(model=original.get("model", args.model_path.name), model_path=str(args.model_path),
               video_root=str(args.video_root), device=args.device, dtype=args.dtype,
               attn_implementation=args.attn_implementation)
    sources = [Path(__file__), *sorted((SOURCE / "onestreamer").glob("*.py")),
               *sorted((ROOT / "Eval/Proactive_Eval/shared").glob("*.py"))]
    model_files = {}
    for path in sorted(set(required + list(args.model_path.glob("*.json")) + list(args.model_path.glob("*.jinja")))):
        stat = path.stat()
        model_files[path.name] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                     sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                                     if stat.st_size < 20 * 1024**2 else None)
    manifest = dict(kind="state_readout_protocol_diagnostic", config=asdict(config), job=job,
                    source_manifest=str(args.manifest),
                    source_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                    samples=normalized, model_files=model_files,
                    source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in sources},
                    protocols={"legacy_v2": "native_readiness_v2", "native_prefix_v3": vm.STATE_PROMPT_VERSION},
                    caption_generation=False, controller_task_labels=False,
                    caveat="Selected diagnostic samples; state diversity does not prove routing quality or RT protection.")
    manifest["fingerprint"] = digest(manifest)
    return manifest, config


def summarize(rows):
    probes = [probe for row in rows for probe in row.get("state_rounds", [])]
    states = Counter(probe.get("state") or "invalid_state" for probe in probes)
    invalid = sum(count for state, count in states.items() if state not in vm.STATE_TOKENS)
    warnings = []
    if not probes:
        warnings.append("no_state_probes")
    if probes and states.get("</Silence>", 0) == len(probes):
        warnings.append("all_silence")
    if invalid:
        warnings.append("invalid_state_predictions")
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(bool(row["hit"]))
    return dict(sample_count=len(rows), correct=sum(bool(row["hit"]) for row in rows),
                sample_micro_accuracy=sum(bool(row["hit"]) for row in rows) / len(rows) if rows else None,
                by_task={task: dict(n=len(hits), correct=sum(hits), accuracy=sum(hits)/len(hits))
                         for task, hits in by_task.items()},
                state_probe_count=len(probes), state_counts=dict(states),
                raw_token_counts=dict(Counter(str(probe.get("predicted_token")) for probe in probes)),
                stop_reasons=dict(Counter(row.get("state_stop_reason") for row in rows)),
                state_probe_seconds=sum(row.get("state_probe_seconds", 0.0) for row in rows),
                query_seconds=sum(row.get("runtime_seconds", 0.0) for row in rows),
                warnings=warnings, diagnostic_status="suspect" if warnings else "completed",
                interpretation="Protocol diagnostics only; state diversity is not evidence of routing correctness.")


def compare(protocol_rows):
    legacy = {row["sample_id"]: row for row in protocol_rows["legacy_v2"]}
    revised = {row["sample_id"]: row for row in protocol_rows["native_prefix_v3"]}
    if set(legacy) != set(revised):
        raise ValueError("Protocols did not produce the same sample ID set")
    pairs = [dict(sample_id=sid, task=legacy[sid]["task"], legacy_hit=bool(legacy[sid]["hit"]),
                  revised_hit=bool(revised[sid]["hit"]),
                  prediction_changed=legacy[sid]["prediction"] != revised[sid]["prediction"],
                  legacy_stop_reason=legacy[sid].get("state_stop_reason"),
                  revised_stop_reason=revised[sid].get("state_stop_reason")) for sid in legacy]
    return dict(n=len(pairs), per_sample=pairs,
                revised_gains=sum(row["revised_hit"] and not row["legacy_hit"] for row in pairs),
                revised_losses=sum(row["legacy_hit"] and not row["revised_hit"] for row in pairs))


def run(args, manifest, config):
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        # Force offline before importing/loading the checkpoint implementation.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        vm._ensure_paths()
        from onestreamer.engine import OneStreamerEngine
        engine = OneStreamerEngine(**{k: manifest["job"][k] for k in
                                      ("model_path", "device", "dtype", "attn_implementation")})
        protocol_rows, summaries = {}, {}
        for protocol, version in manifest["protocols"].items():
            directory = args.output / protocol
            directory.mkdir()
            metadata = dict(parent_fingerprint=manifest["fingerprint"], protocol=protocol,
                            prompt_version=version, config=manifest["config"], job=manifest["job"])
            metadata["fingerprint"] = digest(metadata)
            write_json(directory / "manifest.json", metadata)
            rows = []
            with probe_protocol(protocol), (directory / "predictions.jsonl").open("x", encoding="utf-8") as handle:
                for row in vm.infer(manifest["job"], manifest["samples"], config, engine=engine):
                    row = dict(row, protocol=protocol, prompt_version=version, status="ok",
                               fingerprint=metadata["fingerprint"])
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    rows.append(row)
                    print(f"{protocol}: {row['sample_id']} stop={row.get('state_stop_reason')}", flush=True)
            if {row["sample_id"] for row in rows} != {row["sample_id"] for row in manifest["samples"]} or len(rows) != len(manifest["samples"]):
                raise ValueError(f"Incomplete or duplicate predictions for {protocol}")
            protocol_rows[protocol] = rows
            summaries[protocol] = summarize(rows)
            write_json(directory / "summary.json", summaries[protocol])
        report = dict(fingerprint=manifest["fingerprint"], summaries=summaries,
                      paired=compare(protocol_rows), wall_seconds=time.perf_counter()-started,
                      completed_at=datetime.now(timezone.utc).isoformat(),
                      caveat=manifest["caveat"])
        write_json(args.output / "comparison.json", report)
        print(json.dumps({name: summary for name, summary in summaries.items()}, indent=2), flush=True)
        return 2 if summaries["native_prefix_v3"]["warnings"] else 0
    except Exception as error:
        write_json(args.output / "failure.json", dict(error_type=type(error).__name__, error=str(error),
                   fingerprint=manifest["fingerprint"], elapsed_seconds=time.perf_counter()-started))
        raise


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest, config = prepare(args)
    if args.dry_run:
        print(json.dumps(dict(samples=len(manifest["samples"]), sample_ids=[r["sample_id"] for r in manifest["samples"]],
                              protocols=manifest["protocols"], config=manifest["config"],
                              fingerprint=manifest["fingerprint"], output=str(args.output)), indent=2))
        return 0
    return run(args, manifest, config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"State-readout diagnostic failed: {error}", file=sys.stderr)
        raise SystemExit(1)

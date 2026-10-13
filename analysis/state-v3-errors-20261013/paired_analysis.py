#!/usr/bin/env python3
"""Reproduce paired v3-versus-fill error counts without running inference.

Only paired.json and paired_samples.csv are written, under --output-dir.
Example: /home/zhangnuohua/miniconda3/envs/vllm/bin/python paired_analysis.py
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "Eval/Proactive_Eval/OneStreamer/src"
sys.path.insert(0, str(SOURCE))
from onestreamer.metrics import OVO_TASKS, aggregate
from onestreamer.ovobench import build_question_text, score_prediction
from onestreamer.schema import Sample

SOURCE_FIELDS = ("sample_id", "index", "task", "subtask", "video", "start", "end",
                 "question", "options", "answer", "answer_text", "eval_mode")
RECENT_FIELDS = ("recent_timestamps", "recent_frames", "recent_read_frames",
                 "recent_image_sizes", "recent_sampling")
HISTORY_FIELDS = ("history_timestamps", "history_read_frames", "history_visual_tokens",
                  "history_frame_tokens", "history_pool_scales")
INPUT_FIELDS = ("input_tokens", "visual_tokens", "text_tokens", "image_grid_thw",
                "input_image_sizes", "context_limit", "answer_reserve_tokens")
WRITER_FIELDS = ("bank", "stream_encoded_frames_cumulative", "pooled_input_tokens_cumulative",
                 "pooled_output_tokens_cumulative")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_predictions(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    lookup = {r["sample_id"]: r for r in rows}
    if len(lookup) != len(rows) or not rows:
        raise ValueError(f"Empty/duplicate predictions: {path}")
    if any(r.get("status") != "ok" for r in rows):
        raise ValueError(f"Non-successful predictions: {path}")
    return rows, lookup


def method_ast(source, name):
    module = ast.parse(source)
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "NativeFeatureAdapter")
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.dump(node, include_attributes=False)


def verify_answer_implementation(state_path, fill_path):
    """Check the recorded fill source via its git object, without checking out."""
    relative = "Eval/Proactive_Eval/OneStreamer/src/onestreamer/visual_memory.py"
    state_manifest = json.loads((state_path.parent / "manifest.json").read_text())["identity"]
    fill_manifest = json.loads((fill_path.parent / "manifest.json").read_text())["identity"]
    current = (REPO / relative).read_bytes()
    original_repo = fill_path.parents[2]
    previous = subprocess.check_output(
        ["git", "show", f"{fill_manifest['repo_commit']}:{relative}"], cwd=original_repo)
    state_matches = hashlib.sha256(current).hexdigest() == state_manifest["source_files"][relative]
    fill_matches = hashlib.sha256(previous).hexdigest() == fill_manifest["source_files"][relative]
    if not state_matches or not fill_matches:
        raise ValueError("Cannot verify comparison code against the recorded run source hashes")
    return dict(state_source_matches_manifest=state_matches, fill_git_source_matches_manifest=fill_matches,
                final_chat_builder_ast_equal=method_ast(current.decode(), "build_chat") == method_ast(previous.decode(), "build_chat"),
                final_answer_method_ast_equal=method_ast(current.decode(), "answer") == method_ast(previous.decode(), "answer"))


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return dict(n=len(ordered), mean=statistics.mean(ordered), median=statistics.median(ordered),
                min=ordered[0], max=ordered[-1])


def summarize(records):
    outcomes = Counter(r["outcome"] for r in records)
    count = len(records)
    old_correct = sum(r["fill_hit"] for r in records)
    new_correct = sum(r["state_hit"] for r in records)
    return dict(n=count, gains=outcomes["gain"], losses=outcomes["loss"],
                both_correct=outcomes["both_correct"], both_wrong=outcomes["both_wrong"],
                fill_correct=old_correct, state_correct=new_correct,
                fill_micro_accuracy=old_correct / count if count else None,
                state_micro_accuracy=new_correct / count if count else None,
                micro_delta_pp=100 * (new_correct-old_correct) / count if count else None,
                full_benchmark_macro_contribution_pp=sum(r["macro_weight"] *
                    (r["state_hit"]-r["fill_hit"]) for r in records) * 100,
                raw_prediction_changed=sum(not r["raw_prediction_equal"] for r in records),
                parsed_prediction_changed=sum(not r["parsed_prediction_equal"] for r in records))


def grouped(records, key):
    groups = defaultdict(list)
    for row in records:
        groups[str(row[key])].append(row)
    return {name: summarize(rows) for name, rows in sorted(groups.items())}


def analyze(state_path, fill_path):
    state_rows, states = read_predictions(state_path)
    fill_rows, fills = read_predictions(fill_path)
    if set(states) != set(fills):
        raise ValueError("The two full prediction files have different sample ID sets")
    code_audit = verify_answer_implementation(state_path, fill_path)
    denominators = Counter((r["task"], r["subtask"]) for r in state_rows)
    records = []
    agreements = {field: [] for field in (*RECENT_FIELDS, *HISTORY_FIELDS, *INPUT_FIELDS, *WRITER_FIELDS)}
    for row in state_rows:
        old = fills[row["sample_id"]]
        for field in SOURCE_FIELDS:
            if row.get(field) != old.get(field):
                raise ValueError(f"Source mismatch {row['sample_id']}: {field}")
        sample, old_sample = Sample.from_dict(row), Sample.from_dict(old)
        if build_question_text(sample) != build_question_text(old_sample):
            raise ValueError(f"Rendered question differs: {sample.sample_id}")
        for prediction in (row, old):
            parsed, hit = score_prediction(sample, prediction["prediction"])
            if parsed != prediction["parsed_output"] or hit != prediction["hit"]:
                raise ValueError(f"Current scorer disagrees with saved result: {sample.sample_id}")
        for field, mismatches in agreements.items():
            if row.get(field) != old.get(field):
                mismatches.append(sample.sample_id)
        same_recent = all(row.get(k) == old.get(k) for k in RECENT_FIELDS)
        same_history = all(row.get(k) == old.get(k) for k in HISTORY_FIELDS)
        same_input = all(row.get(k) == old.get(k) for k in INPUT_FIELDS)
        rounds = row["state_rounds"]
        initial, final = rounds[0], rounds[-1]
        new_hit, old_hit = int(row["hit"]), int(old["hit"])
        outcome = ("gain" if new_hit and not old_hit else "loss" if old_hit and not new_hit
                   else "both_correct" if new_hit else "both_wrong")
        available = row["history_selection_meta"]["eligible_history_frames"]
        final_stage = ("zero" if not row["history_read_frames"] else "all_eligible"
                       if row["history_read_frames"] == available else "partial")
        stop_stage = ("response_recent_only" if not row["history_read_frames"] else "response_after_history") \
            if row["state_stop_reason"] == "response" else row["state_stop_reason"]
        old_frames = dict(zip(old["history_timestamps"], zip(old["history_frame_tokens"], old["history_pool_scales"])))
        new_frames = dict(zip(row["history_timestamps"], zip(row["history_frame_tokens"], row["history_pool_scales"])))
        subset = all(timestamp in old_frames and old_frames[timestamp] == data for timestamp, data in new_frames.items())
        record = dict(sample_id=sample.sample_id, task=sample.task, subtask=sample.subtask,
                      eval_mode=sample.eval_mode, question=sample.question, options=sample.options,
                      answer=sample.answer, answer_text=sample.answer_text,
                      fill_prediction=old["prediction"], state_prediction=row["prediction"],
                      fill_parsed=old["parsed_output"], state_parsed=row["parsed_output"],
                      fill_hit=old_hit, state_hit=new_hit, outcome=outcome,
                      macro_weight=1/(3*len(OVO_TASKS[sample.task])*denominators[(sample.task, sample.subtask)]),
                      state_stop_reason=row["state_stop_reason"], stop_stage=stop_stage,
                      task_stop_stage=sample.task+"|"+stop_stage, subtask_stop_stage=sample.subtask+"|"+stop_stage,
                      final_budget_tokens=final["history_budget_tokens"], final_history_stage=final_stage,
                      state_round_count=len(rounds), available_history_frames=available,
                      fill_history_frames=old["history_read_frames"], state_history_frames=row["history_read_frames"],
                      fill_history_tokens=old["history_visual_tokens"], state_history_tokens=row["history_visual_tokens"],
                      history_token_fraction=row["history_visual_tokens"]/old["history_visual_tokens"] if old["history_visual_tokens"] else None,
                      fill_input_tokens=old["input_tokens"], state_input_tokens=row["input_tokens"],
                      same_recent_metadata=same_recent, same_bank_metadata=row["bank"]==old["bank"],
                      same_history_metadata=same_history, same_input_metadata=same_input,
                      same_final_evidence_metadata=same_recent and same_history and same_input,
                      state_history_is_fill_subset_with_equal_frame_metadata=subset,
                      raw_prediction_equal=row["prediction"]==old["prediction"],
                      parsed_prediction_equal=row["parsed_output"]==old["parsed_output"],
                      initial_state=initial["state"], final_state=final["state"],
                      initial_response_probability=initial["state_probabilities"]["</Response>"],
                      final_response_probability=final["state_probabilities"]["</Response>"],
                      initial_silence_probability=initial["state_probabilities"]["</Silence>"],
                      final_silence_probability=final["state_probabilities"]["</Silence>"],
                      initial_standby_probability=initial["state_probabilities"]["</Standby>"],
                      final_standby_probability=final["state_probabilities"]["</Standby>"],
                      history_context_clipped=row["history_selection_meta"]["state_context_clipped"],
                      final_history_timestamps=row["history_timestamps"], fill_history_timestamps=old["history_timestamps"],
                      state_probe_seconds=row["state_probe_seconds"], state_query_seconds=row["runtime_seconds"],
                      fill_query_seconds=old["runtime_seconds"])
        records.append(record)
    official_state = aggregate(state_rows)["metrics"][0]
    official_fill = aggregate(fill_rows)["metrics"][0]
    if any(official_state["missing_subtasks"].values()) or any(official_fill["missing_subtasks"].values()):
        raise ValueError("Full benchmark is incomplete")
    tasks = grouped(records, "task")
    subtasks = grouped(records, "subtask")
    for task, metric in tasks.items():
        metric.update(fill_macro_accuracy=official_fill["task_accuracy"][task],
                      state_macro_accuracy=official_state["task_accuracy"][task],
                      macro_delta_pp=100*(official_state["task_accuracy"][task]-official_fill["task_accuracy"][task]))
    same_input_records = [r for r in records if r["same_final_evidence_metadata"]]
    no_history = [r for r in records if r["available_history_frames"] == 0]
    response_records = [r for r in records if r["state_stop_reason"] == "response"]
    groups = {key: grouped(records, key) for key in ("state_stop_reason", "stop_stage", "final_budget_tokens",
               "final_history_stage", "state_round_count", "task_stop_stage", "subtask_stop_stage")}
    report = dict(
        sources={"state": dict(path=str(state_path), sha256=sha(state_path)),
                 "fill": dict(path=str(fill_path), sha256=sha(fill_path)),
                 "analysis_script": dict(path=str(Path(__file__).resolve()), sha256=sha(Path(__file__)))},
        validation=dict(n=len(records), paired_source_fields=list(SOURCE_FIELDS),
                        rendered_questions_equal=True, current_scorer_reproduces_all_saved_parsed_outputs_and_hits=True,
                        scoring_source_sha256={name:sha(SOURCE/"onestreamer"/name) for name in ("ovobench.py", "schema.py", "metrics.py")},
                        answer_implementation=code_audit,
                        agreement_by_field={field:dict(equal=len(records)-len(ids),different=len(ids),different_ids=ids)
                                            for field,ids in agreements.items()},
                        feature_tensor_hashes_available=False,
                        evidence_equality_limit="Exact equality of saved timing/grid/token/pooling metadata; feature tensors were not serialized or hashed."),
        overall=summarize(records), task=tasks, subtask=subtasks, groups=groups,
        official_macro=dict(fill=official_fill["overall_accuracy"],state=official_state["overall_accuracy"],
                            delta_pp=100*(official_state["overall_accuracy"]-official_fill["overall_accuracy"]),
                            definition="Mean of 3 task-category scores; each category is the unweighted mean of its subtask accuracies."),
        same_final_evidence_metadata=dict(summary=summarize(same_input_records),
             changed_prediction_ids=[r["sample_id"] for r in same_input_records if not r["raw_prediction_equal"]]),
        no_available_history=dict(summary=summarize(no_history),
             nonzero_fill_history_ids=[r["sample_id"] for r in no_history if r["fill_history_frames"]],
             changed_prediction_ids=[r["sample_id"] for r in no_history if not r["raw_prediction_equal"]]),
        state_history_not_fill_subset_ids=[r["sample_id"] for r in records if not r["state_history_is_fill_subset_with_equal_frame_metadata"]],
        response_confidence_by_outcome={outcome:dict(
            n=sum(r["outcome"]==outcome for r in response_records),
            final_response_probability=distribution([r["final_response_probability"] for r in response_records if r["outcome"]==outcome]))
            for outcome in ("gain", "loss", "both_correct", "both_wrong")},
        caveats=["Correctness changes are paired against visual fill, not against an independently executed matched recent-only baseline.",
                 "Macro task/benchmark deltas differ from micro sample accuracy deltas. Group macro contributions use full-benchmark weights, not reweighted subset scores.",
                 "Native state probabilities are uncalibrated; error-conditioned summaries are descriptive and are not a validated routing threshold.",
                 "Metadata equality cannot prove bitwise equality of unrecorded visual tensors."])
    return report, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=REPO/"results/ovo-state-memory-v3-2gpu-20261012/predictions.jsonl")
    parser.add_argument("--fill", type=Path, default=REPO.parent/"video2/results/ovo-visual-memory-2gpu-20261011/predictions.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--overwrite", action="store_true", help="Replace only this script's two derived analysis outputs")
    args = parser.parse_args()
    outputs = [args.output_dir/"paired.json", args.output_dir/"paired_samples.csv"]
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("Derived analysis exists; pass --overwrite to reproduce it")
    report, records = analyze(args.state.resolve(), args.fill.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    with outputs[1].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in record.items()})
    print(json.dumps(dict(overall=report["overall"],official_macro=report["official_macro"],
                          task=report["task"],stop_stage=report["groups"]["stop_stage"],
                          no_history=report["no_available_history"],same_evidence=report["same_final_evidence_metadata"]),indent=2))


if __name__ == "__main__":
    main()

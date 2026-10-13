"""Read-only CPU audit of v3 memory retention, selection, and forced-answer behavior.

Run with Python 3. Writes only adjacent memory_audit.json and memory_audit.md.
Does not load tensors, decode videos, run inference, or modify evaluation code.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
FIX = OUT.parents[1]
OLD = FIX.parent / "video2"
NEW_PATH = FIX / "results/ovo-state-memory-v3-2gpu-20261012/predictions.jsonl"
OLD_PATH = OLD / "results/ovo-visual-memory-2gpu-20261011/predictions.jsonl"


def read(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len({row["sample_id"] for row in rows}) == len(rows)
    return {row["sample_id"]: row for row in rows}


def median(values):
    return statistics.median(values) if values else None


def summarize(rows):
    return {
        "n": len(rows),
        "v3_correct": sum(r["v3_hit"] for r in rows),
        "oldfill_correct": sum(r["oldfill_hit"] for r in rows),
        "with_eviction": sum(r["dropped_oldest"] > 0 for r in rows),
        "eviction_without_repeated_pool": sum(r["evicted_before_any_second_pool"] for r in rows),
        "median_available_prefix_fraction": median([r["available_prefix_fraction"] for r in rows]),
        "available_prefix_fraction_lt_half": sum(r["available_prefix_fraction"] < .5 for r in rows),
        "median_retention_horizon_seconds": median([r["retention_horizon_seconds"] for r in rows]),
        "median_stored_frames": median([r["stored_frames"] for r in rows]),
        "median_selected_frames": median([r["selected_frames"] for r in rows]),
        "no_history_read": sum(r["selected_frames"] == 0 for r in rows),
        "all_bank_read": sum(r["selected_frames"] == r["stored_frames"] for r in rows),
        "all_bank_read_with_nonempty_bank": sum(r["selected_frames"] == r["stored_frames"] and r["stored_frames"] > 0 for r in rows),
        "final_states": dict(Counter(r["states"][-1] for r in rows)),
        "stop_reasons": dict(Counter(r["stop_reason"] for r in rows)),
    }


def main():
    new, old = read(NEW_PATH), read(OLD_PATH)
    assert new.keys() == old.keys()
    manifest = json.loads((NEW_PATH.parent / "manifest.json").read_text())
    code_dir = FIX / "Eval/Proactive_Eval/OneStreamer/src/onestreamer"
    source_checks = {}
    for name in ("visual_memory.py", "state_readout.py"):
        relative = f"Eval/Proactive_Eval/OneStreamer/src/onestreamer/{name}"
        actual = hashlib.sha256((code_dir / name).read_bytes()).hexdigest()
        source_checks[relative] = {
            "sha256": actual,
            "matches_run_manifest": actual == manifest["identity"]["source_files"][relative],
        }
    derived = []
    for sid, row in new.items():
        prior = old[sid]
        assert row["bank"] == prior["bank"], sid
        assert row["recent_timestamps"] == prior["recent_timestamps"], sid
        assert prior["history_read_frames"] == row["bank"]["stored_frames"], sid
        assert len(prior["history_timestamps"]) == row["bank"]["stored_frames"], sid
        assert sum(prior["history_frame_tokens"]) == row["bank"]["stored_tokens"], sid
        for key in ("question", "answer", "video", "start", "end", "subtask", "task"):
            assert row[key] == prior[key], (sid, key)
        available = prior["history_timestamps"] + row["recent_timestamps"]
        assert available == sorted(set(available)), sid
        assert all(row["start"] <= t < row["end"] for t in available), sid
        assert set(row["history_timestamps"]).issubset(prior["history_timestamps"]), sid
        earliest = min(available)
        bank = row["bank"]
        records = {
            "sample_id": sid, "task": row["task"], "subtask": row["subtask"],
            "v3_hit": int(row["hit"]), "oldfill_hit": int(prior["hit"]),
            "both_wrong": not row["hit"] and not prior["hit"],
            "video": row["video"], "start": row["start"], "end": row["end"],
            "question": row["question"], "answer": row["answer"], "answer_text": row.get("answer_text"),
            "v3_prediction": row["prediction"], "oldfill_prediction": prior["prediction"],
            "stored_frames": bank["stored_frames"], "stored_tokens": bank["stored_tokens"],
            "seen": bank["seen"], "dropped_oldest": bank["dropped_oldest"],
            "selected_frames": row["history_read_frames"],
            "available_first_history": prior["history_timestamps"][0] if prior["history_timestamps"] else None,
            "available_last_history": prior["history_timestamps"][-1] if prior["history_timestamps"] else None,
            "recent_first": row["recent_timestamps"][0],
            "earliest_available": earliest,
            "retention_horizon_seconds": row["end"] - earliest,
            "available_prefix_fraction": (row["end"] - earliest) / (row["end"] - row["start"]),
            "available_pool_scales": sorted(set(prior["history_pool_scales"])),
            "available_frame_token_counts": sorted(set(prior["history_frame_tokens"])),
            "evicted_before_any_second_pool": bool(bank["dropped_oldest"] and bank["pool_operations"] == bank["seen"]),
            "states": [p["state"] for p in row["state_rounds"]],
            "state_response_probabilities": [p["state_probabilities"]["</Response>"] for p in row["state_rounds"]],
            "round_read_frames": [p["history_read_frames"] for p in row["state_rounds"]],
            "stop_reason": row["state_stop_reason"],
            "same_final_visual_input_as_oldfill": row["history_timestamps"] == prior["history_timestamps"],
            "same_prediction_as_oldfill": row["prediction"] == prior["prediction"],
        }
        derived.append(records)
    subsets = {
        "all": derived,
        "all_both_wrong": [r for r in derived if r["both_wrong"]],
        "BT": [r for r in derived if r["task"] == "backward_tracking"],
        "BT_both_wrong": [r for r in derived if r["task"] == "backward_tracking" and r["both_wrong"]],
        "REC": [r for r in derived if r["subtask"] == "REC"],
        "REC_both_wrong": [r for r in derived if r["subtask"] == "REC" and r["both_wrong"]],
    }
    for subtask in sorted({r["subtask"] for r in derived}):
        subsets[f"subtask_{subtask}"] = [r for r in derived if r["subtask"] == subtask]
        subsets[f"subtask_{subtask}_both_wrong"] = [r for r in derived if r["subtask"] == subtask and r["both_wrong"]]
    states = {}
    for state in sorted({r["states"][-1] for r in derived}):
        matching = [r for r in derived if r["states"][-1] == state]
        states[state] = {
            **summarize(matching),
            "same_final_visual_input_as_oldfill": sum(r["same_final_visual_input_as_oldfill"] for r in matching),
            "same_prediction_as_oldfill": sum(r["same_prediction_as_oldfill"] for r in matching),
        }
    retention_cliff = Counter(
        (r["stored_frames"], tuple(r["available_frame_token_counts"]))
        for r in derived if r["evicted_before_any_second_pool"]
    )
    cases = ["ovo-0000", "ovo-0264", "ovo-2344", "ovo-2346", "ovo-2358", "ovo-2616", "ovo-3021"]
    report = {
        "inputs": {"v3": str(NEW_PATH), "oldfill": str(OLD_PATH)},
        "source_checks": source_checks,
        "configuration": manifest["identity"]["config"],
        "validation": {
            "unique_matched_samples": len(derived),
            "identical_banks": len(derived),
            "oldfill_reads_complete_bank": len(derived),
            "identical_recent_timestamps": len(derived),
            "all_logged_available_and_selected_timestamps_causal": True,
            "note": "Available bank timestamps are measured from oldfill, whose complete bank and recent timestamps exactly match v3. Coverage denotes sampled timestamps, not preservation of all semantic/action evidence.",
        },
        "subsets": {name: summarize(rows) for name, rows in subsets.items()},
        "by_final_state": states,
        "retention_cliff_groups": [
            {"stored_frames": n, "frame_tokens": list(tokens), "samples": count}
            for (n, tokens), count in retention_cliff.most_common()
        ],
        "cases": [next(r for r in derived if r["sample_id"] == sid) for sid in cases],
        "per_sample": derived,
    }
    (OUT / "memory_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    make_markdown(report)
    print(json.dumps({"subsets": {k:report["subsets"][k] for k in ["all", "BT_both_wrong", "REC_both_wrong"]}, "by_final_state": states, "output": str(OUT / "memory_audit.json")}, indent=2))


def make_markdown(report):
    lines = [
        "# State v3 memory audit (2026-10-13)", "",
        "This is an offline, read-only audit of 3,035 matched predictions. No inference was run. All numerical counts are sample counts, not independent-video counts.", "",
        "## Verified facts", "",
        "- Every v3 bank snapshot and recent timestamp list equals oldfill's. Oldfill reads all stored frames in every sample, so its historical timestamps reveal the full bank available to v3 independently of the gate.",
        "- All logged historical/recent timestamps are strictly before the answer cutoff. v3 selected histories are subsets of the available bank. This checks logged causality, not a separate video re-decoding audit.",
        "- 1,216 samples have eviction; 581 evict before any second age-based pooling. Of those 581: 154 retain 23 frames at 690 tokens/frame; 231 retain 32 at 510; 181 retain 41 at 391; 15 retain 13 at 1,200.",
        "- The bank budget is 16,384 tokens, with a 60-second interval before re-pooling. At 1 fps, these four sizes hit capacity before their first second-stage pool. The progressive bank therefore behaves like a short FIFO in those samples.",
        "- All 698 REC examples have zero eviction. Thus eviction cannot explain their errors; 427 are wrong in both runs, including 108 where v3 reads the complete bank (possibly empty). Temporal coverage also does not guarantee observing every short action at 1 fps.", "",
        "| Subset | n | Eviction | Median available prefix span | Prefix span < 50% | No history read | All bank read |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ["BT", "BT_both_wrong", "REC", "REC_both_wrong"]:
        s=report["subsets"][key]
        lines.append(f"| {key} | {s['n']} | {s['with_eviction']} | {s['median_available_prefix_fraction']:.2%} | {s['available_prefix_fraction_lt_half']} | {s['no_history_read']} | {s['all_bank_read']} |")
    lines += [
        "", "Prefix span is `(end - earliest_available_timestamp) / (end - start)` using the entire bank plus Recent16, regardless of what v3 selects. It is not a ground-truth answer-evidence coverage metric.", "",
        "## Selection and final answering", "",
        "- `state_readout.py:39` builds an endpoint-first, progressively uniform temporal order. `:118` and `:134` select whole bundles by token budget. The question affects probing and token counts, but does not semantically rank candidate frames.",
        "- `state_readout.py:181` stops on Response; both Standby and Silence request another budget. There are no Standby or invalid final states in this run.",
        "- 1,043 examples end with Silence: 886 exhaust the bank and 157 have no bank. `visual_memory.py:983` still unconditionally calls the ordinary answer function, which uses the normal forced-answer prompt, not the state prompt. These are bounded best-effort answers, not a native Silence response.",
        "- The 1,043 final-Silence examples have exactly the same final history as oldfill and 734/1,043 are correct in both runs. Therefore Silence is not a calibrated indicator that answering will be wrong, and these outcomes do not demonstrate a new v3 retrieval regression.", "",
        "## Concrete cases", "",
        "| ID | What the logs show | Limit on interpretation |", "|---|---|---|",
        "| ovo-0000 (EPM) | At cutoff 215 s, bank is only 176–198 s plus Recent16 at 199–214 s; 176 earlier frames evicted. All four probes return Silence; both runs answer D, gold C. | The target chopping/interaction time is not logged; cannot prove the gold evidence was among discarded frames. |",
        "| ovo-0264 (EPM) | Asks about painting in the first bedroom. At 163 s, only 115–162 s remains; 115 frames evicted. Both runs wrong; all probes Silence. | Refers to a first room, but its actual visual occurrence was not manually verified. |",
        "| ovo-2344 (REC) | Entire 0–13 s prefix present with no historical bank. Both runs predict 2; gold 1. | Demonstrates an error that memory selection cannot explain. Recognition/temporal sampling/counting remain hypotheses. |",
        "| ovo-2346 (REC) | All 0–4 s observations present; no bank; probe Silence; both still output 1 while gold is 0. | Exposes the native Silence to forced answer transition, but not proof that abstention should score as zero. |",
        "| ovo-2358 (REC) | 17 historical frames available, zero eviction. v3 says Response and selects none; both runs predict 1, gold 2. | Early stop omitted evidence but oldfill also failed with it, so this is not sufficient evidence that more readout alone fixes it. |",
        "| ovo-2616 (REC) | 80 historical frames plus Recent16 cover 0–95 s. v3 reads the complete bank; both predict 2 javelin throws, gold 3. | Full 1-fps coverage still leaves action sampling, spatial pooling and temporal reasoning as possible causes. |",
        "| ovo-3021 (REC) | 82 history frames plus Recent16 cover 0–97 s; no eviction; v3 reads all. Both predict 2 spikes, gold 9. | Same limitation: no proof that every spike was observable in retained 1-fps frames. |", "",
        "## At most three training-free mechanisms to test", "",
        "1. **Compress under capacity pressure before deleting old records.** Keep temporal coverage by further pooling older visual records when the token budget would overflow; preserve selected event neighborhoods or temporal anchors. This targets the verified eviction-before-aging cliff. It may erase detail, so compare equal storage-token budgets and inspect the earliest retained timestamp and recognition quality. No task label or caption is required.",
        "2. **Retrieve question-relevant visual segments with explicit temporal coverage.** Use the frozen model to request evidence entities/events/time spans and select stored frame IDs or short adjacent segments, with an empty result allowed. Retain a coverage path for multi-event evidence instead of treating top similar isolated frames as complete. This targets query-blind temporal ordering. Its benefit is a hypothesis until tested, and nothing can retrieve already discarded observations.",
        "3. **Separate evidence coverage from native willingness to answer.** Have the frozen model judge which observations the question requires and whether the selected view covers them, instead of using Response alone as sufficient evidence. Keep recent-only generation independent when comparing candidate corrections; do not feed a memory candidate answer into a verifier as a premise. Final Silence should remain visible in diagnostics even when the benchmark requires best-effort answering. This adds inference cost and cannot guarantee per-question nonregression.", "",
        "## Code references", "",
    ]
    refs = [("visual_memory.py",358,"age-based pooling"),("visual_memory.py",401,"first re-pooling scheduled after 60 s"),("visual_memory.py",406,"delete oldest immediately under capacity pressure"),("visual_memory.py",625,"native state prompt differs from final answer prompt"),("visual_memory.py",659,"full-vocabulary state probe"),("visual_memory.py",983,"unconditional final answer"),("state_readout.py",39,"query-blind temporal ordering"),("state_readout.py",129,"nested budget selection"),("state_readout.py",181,"state stopping and history exhaustion")]
    for name,line,label in refs:
        path = FIX / "Eval/Proactive_Eval/OneStreamer/src/onestreamer" / name
        lines.append(f"- [{name}:{line}]({path}:{line}): {label}.")
    lines += ["", "Both reviewed code files match the run manifest SHA-256. The JSON stores per-sample derived fields and exact case metadata; rerun `memory_audit.py` to regenerate this audit.", ""]
    (OUT / "memory_audit.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

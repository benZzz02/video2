"""Summarize the hard-case trial against official Recent16 and the old Fill run."""
import collections, json, statistics
from pathlib import Path

M = Path("/home/zhangnuohua/medhorizon")
sel = json.load(open("selection.json"))["groups"]
R = json.load(open(M / "video2-action-readout-20261013/analysis/recent16-memory-errors-20261014/recent16_predictions.json"))
if isinstance(R, dict):
    R = R.get("predictions") or R.get("samples") or list(R.values())
R = {r["sample_id"]: r for r in R}
F = {json.loads(l)["sample_id"]: json.loads(l) for l in open(M / "video2/results/ovo-visual-memory-2gpu-20261011/predictions.jsonl")}
N = {}
for p in sorted(Path(".").glob("shard-*/predictions.jsonl")):
    N.update({json.loads(l)["sample_id"]: json.loads(l) for l in open(p)})
by = collections.defaultdict(list)
for i, r in N.items():
    by[sel[i]].append(i)
print("group       done/sel  new_correct  R_correct  oldFill_correct  old_hist_span_s  new_hist_span_s")
for g in ("EPM", "SSR", "ASI", "CRR", "REC", "RT_control"):
    ids = by.get(g, [])
    total = sum(v == g for v in sel.values())
    if not ids:
        print(f"{g:11s} 0/{total}"); continue
    span = lambda run, i: (run[i]["recent_timestamps"][0] - run[i]["history_timestamps"][0]) if run[i]["history_timestamps"] else 0
    print(f"{g:11s} {len(ids):3d}/{total:<3d}   {sum(N[i]['hit'] for i in ids):5d}       {sum(R[i]['hit'] for i in ids):5d}       "
          f"{sum(F[i]['hit'] for i in ids):5d}         {statistics.median(span(F, i) for i in ids):8.0f}      "
          f"{statistics.median(span(N, i) for i in ids):8.0f}")
x = [r["input_tokens"] for r in N.values()]; q = [r["qa_seconds"] for r in N.values()]
print("input_tokens max", max(x), "median", statistics.median(x), "| qa_seconds median", round(statistics.median(q), 1), "max", round(max(q), 1))
print("budget violations", sum(r["bank"]["budget_cost"] > r["bank"]["token_budget"] for r in N.values()),
      "| earliest!=first", sum(r["bank"]["earliest_timestamp"] is not None and r["bank"]["earliest_timestamp"] > r["start"] + 1.5 for r in N.values()))
print("\nchanged answers (new vs old Fill):")
for i in sorted(N, key=lambda i: (sel[i], i)):
    if N[i]["parsed_output"] != F[i]["parsed_output"]:
        print(f"  {i} {sel[i]:10s} gt={N[i]['answer']!s:5.5} R={R[i]['parsed_output']!s:5.5} oldF={F[i]['parsed_output']!s:5.5} new={N[i]['parsed_output']!s:5.5} hit={N[i]['hit']} hist_frames {F[i]['history_read_frames']}->{N[i]['history_read_frames']}")

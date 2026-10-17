"""Recompute official OVO-Bench scores for every full run from its predictions.

Overall = mean over BT/RT/FT of the mean subtask accuracy (official macro).
The frame-ID run is also rescored with the official parser on its raw output,
because its witness parser scored answers without an "ANSWER:" line as empty.
"""
import glob, json, re, sys
from pathlib import Path

M = Path("/home/zhangnuohua/medhorizon")
sys.path.insert(0, str(M / "video2-fill-full-context-20261015/Eval/Proactive_Eval/OneStreamer/src"))
from onestreamer.ovobench import score_prediction
from onestreamer.schema import Sample

GROUP = {"EPM": "BT", "HLD": "BT", "ASI": "BT", "STU": "RT", "OJR": "RT", "ATR": "RT",
         "FPD": "RT", "ACR": "RT", "OCR": "RT", "CRR": "FT", "SSR": "FT", "REC": "FT"}
RUNS = [
    ("Recent16 (official)", "/mnt/truenas_nfs/dataset/znh1302/OneStreamer-OVO-eval/outputs/OneStreamer-4B/ovobench/ovo-full-recent16-2gpu-20261012/recent16/ovobench/predictions.json"),
    ("Fill (progressive_pool, 2026-10-11)", "video2/results/ovo-visual-memory-2gpu-20261011/predictions.jsonl"),
    ("State v3", "video2-state-fix-20261012/results/ovo-state-memory-v3-2gpu-20261012/predictions.jsonl"),
    ("Action", "video2-action-readout-20261013/results/ovo-action-memory-2gpu-20261013/predictions.jsonl"),
    ("Frame-ID witness (as scored)", "video2-action-readout-20261013/results/ovo-frame-id-witness-fill-2gpu-20261014/predictions.jsonl"),
    ("Frame-ID witness (official parser)", "video2-action-readout-20261013/results/ovo-frame-id-witness-fill-2gpu-20261014/predictions.jsonl"),
    ("Full-context budget_pool v2", "video2-fill-full-context-20261015/results/ovo-fill-full-context-v2-2gpu-20261015/predictions.jsonl"),
]


def load(path):
    path = Path(path) if Path(path).is_absolute() else M / path
    if path.suffix == ".json":
        data = json.load(open(path))
        return data.get("predictions") or data.get("samples") if isinstance(data, dict) else data
    return [json.loads(line) for line in open(path)]


def rescore(row):
    raw = row["raw_prediction"]
    match = re.search(r"(?m)^ANSWER:[ \t]*(.+)$", raw)
    answer = match.group(1) if match else (raw.strip().splitlines() or [""])[0]
    return score_prediction(Sample.from_dict(row), answer)[1]


table = []
for name, path in RUNS:
    rows = load(path)
    assert len(rows) == 3035 and len({r["sample_id"] for r in rows}) == 3035, (name, len(rows))
    hit = rescore if "official parser" in name else (lambda r: r["hit"])
    acc = {s: 100 * sum(hit(r) for r in rows if r["subtask"] == s) / sum(r["subtask"] == s for r in rows) for s in GROUP}
    g = {k: sum(acc[s] for s in GROUP if GROUP[s] == k) / sum(GROUP[s] == k for s in GROUP) for k in ("BT", "RT", "FT")}
    table.append(dict(run=name, source=str(path), overall=sum(g.values()) / 3, **g, subtasks=acc))

json.dump(table, open("full_run_scores.json", "w"), indent=1)
print("| Run | Overall | BT | RT | FT |\n|---|---:|---:|---:|---:|")
for t in table:
    print(f"| {t['run']} | {t['overall']:.2f} | {t['BT']:.2f} | {t['RT']:.2f} | {t['FT']:.2f} |")
print("\n| Run | " + " | ".join(GROUP) + " |\n|---|" + "---:|" * len(GROUP))
for t in table:
    print(f"| {t['run']} | " + " | ".join(f"{t['subtasks'][s]:.1f}" for s in GROUP) + " |")

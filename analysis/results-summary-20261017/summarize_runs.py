"""Recompute official OVO-Bench scores for every full run from its predictions.

Overall = mean over BT/RT/FT of the mean subtask accuracy (official macro).
The frame-ID run is also rescored with the official parser on its raw output,
because its witness parser scored answers without an "ANSWER:" line as empty.
"""
import gzip, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Eval/Proactive_Eval/OneStreamer/src"))
from onestreamer.ovobench import score_prediction
from onestreamer.schema import Sample

GROUP = {"EPM": "BT", "HLD": "BT", "ASI": "BT", "STU": "RT", "OJR": "RT", "ATR": "RT",
         "FPD": "RT", "ACR": "RT", "OCR": "RT", "CRR": "FT", "SSR": "FT", "REC": "FT"}
HERE = Path(__file__).resolve().parent / "predictions"
RUNS = [
    ("Recent16 (official)", HERE / "recent16-20261012.json.gz"),
    ("Fill (progressive_pool, 2026-10-11)", HERE / "fill-20261011.jsonl.gz"),
    ("State v3", HERE / "state-v3-20261012.jsonl.gz"),
    ("Action", HERE / "action-20261013.jsonl.gz"),
    ("Frame-ID witness (as scored)", HERE / "frame-id-witness-20261014.jsonl.gz"),
    ("Frame-ID witness (official parser)", HERE / "frame-id-witness-20261014.jsonl.gz"),
    ("Full-context budget_pool v2", HERE / "full-context-v2-20261015.jsonl.gz"),
]


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        if path.name.endswith(".json.gz"):
            data = json.load(handle)
            return data.get("predictions") or data.get("samples") if isinstance(data, dict) else data
        return [json.loads(line) for line in handle]


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
    table.append(dict(run=name, source="predictions/" + path.name, overall=sum(g.values()) / 3, **g, subtasks=acc))

json.dump(table, open(Path(__file__).resolve().parent / "full_run_scores.json", "w"), indent=1)
print("| Run | Overall | BT | RT | FT |\n|---|---:|---:|---:|---:|")
for t in table:
    print(f"| {t['run']} | {t['overall']:.2f} | {t['BT']:.2f} | {t['RT']:.2f} | {t['FT']:.2f} |")
print("\n| Run | " + " | ".join(GROUP) + " |\n|---|" + "---:|" * len(GROUP))
for t in table:
    print(f"| {t['run']} | " + " | ".join(f"{t['subtasks'][s]:.1f}" for s in GROUP) + " |")

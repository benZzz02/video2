"""Fixed hash selection of Recent16 hard cases for the full-context memory trial."""
import hashlib, json
from pathlib import Path

M = Path("/home/zhangnuohua/medhorizon")
SEED = "fullctx-trial-20261015"
R = json.load(open(M / "video2-action-readout-20261013/analysis/recent16-memory-errors-20261014/recent16_predictions.json"))
if isinstance(R, dict):
    R = R.get("predictions") or R.get("samples") or list(R.values())
R = {r["sample_id"]: r for r in R}
F = {json.loads(l)["sample_id"]: json.loads(l) for l in open(M / "video2/results/ovo-visual-memory-2gpu-20261011/predictions.jsonl")}
excluded = set(json.load(open(M / "video2-action-readout-20261013/analysis/heldout-panel-20261014/heldout_panel.audit.json"))["excluded_sample_ids"])
rows = {}
for p in sorted((M / "video2-action-readout-20261013/results/ovo-frame-id-witness-fill-2gpu-20261014/inputs").glob("shard-*.json")):
    for r in json.load(open(p))["samples"]:
        rows[r["sample_id"]] = r
key = lambda i: hashlib.sha256(f"{SEED}|{i}".encode()).hexdigest()
trunc = lambda i: F[i]["bank"]["dropped_oldest"] > 0
both_wrong = lambda i: not R[i]["hit"] and not F[i]["hit"]
groups = [
    ("EPM", 30, lambda i, s: s == "EPM" and both_wrong(i) and trunc(i), None),
    ("SSR", 20, lambda i, s: s == "SSR" and both_wrong(i) and trunc(i), None),
    ("ASI", 10, lambda i, s: s == "ASI" and both_wrong(i), trunc),
    ("CRR", 15, lambda i, s: s == "CRR" and both_wrong(i), None),
    ("REC", 15, lambda i, s: s == "REC" and both_wrong(i), None),
    ("RT_control", 20, lambda i, s: s in ("STU", "OJR") and R[i]["hit"] and not F[i]["hit"], None),
]
selected = []
for name, n, cond, prefer in groups:
    pool = [i for i in R if i not in excluded and cond(i, R[i]["subtask"])]
    pool.sort(key=lambda i: ((not prefer(i)) if prefer else 0, key(i)))
    chosen = pool[:n]
    print(name, "eligible", len(pool), "chosen", len(chosen))
    selected += [(name, i) for i in chosen]
# Balance two shards by video length (streams are per-sample chunks).
order = sorted(selected, key=lambda x: -rows[x[1]]["end"])
shards, load = [[], []], [0.0, 0.0]
for name, i in order:
    k = 0 if load[0] <= load[1] else 1
    shards[k].append(rows[i]); load[k] += rows[i]["end"] - rows[i]["start"]
for tag, items in zip("ab", shards):
    Path(f"samples-{tag}.json").write_text(json.dumps(dict(samples=items), indent=1))
Path("selection.json").write_text(json.dumps(dict(seed=SEED, groups={i: g for g, i in selected},
                                                  shard_video_seconds=load), indent=1))
print("shard seconds", [round(x) for x in load], "sizes", [len(s) for s in shards])

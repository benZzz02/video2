"""Qwen3-VL-8B: score V-CAST storage under the main method without threshold, against the same method with
budget_pool storage (stitched 63.41) and the recent-16 s baseline (62.85).

NOW-labeled questions use Qwen's recent-16 s answers in both; SHORT/LONG questions use each storage's run.
"""
import glob, json, sys
from collections import defaultdict
import numpy as np
F = "/home/zhangnuohua/medhorizon/video2-fill-full-context-20261015"
sys.path.insert(0, f"{F}/analysis/paper-analysis-20261018")
import common

ROOT = "/home/zhangnuohua/medhorizon/video2-vcast-20261020/results/ovo-vcast-qwen3vl8b-short-long-2gpu-20261021"
load = lambda pattern: {json.loads(l)["sample_id"]: json.loads(l) for p in glob.glob(pattern) for l in open(p) if l.endswith("\n")}
R = load(f"{F}/analysis/qwen3vl8b-20261019/recent16-[01].jsonl")
S = load(f"{F}/results/ovo-routed-qwen3vl8b-2gpu-20261019/attempts/*/predictions.jsonl")
V = load(f"{ROOT}/attempts/*/predictions.jsonl")
ids = sorted(R); sub = {i: R[i]["subtask"] for i in ids}
label = {i: S[i]["route"]["label"] for i in ids}
mem = [i for i in ids if label[i] != "NOW"]
done = [i for i in mem if i in V]
print(f"finished {len(done)} / {len(mem)} SHORT/LONG questions")
print("router label equal:", sum(V[i]["route"]["label"] == label[i] for i in done), "/", len(done))
by = defaultdict(list)
for i in done:
    by[sub[i]].append(i)
print("\nsame questions, per subtask (recent16 / budget_pool / vcast):")
for s in common.GROUP:
    q = by.get(s, [])
    if q:
        f = lambda X: 100 * np.mean([X[i]["hit"] for i in q])
        print(f"  {s:4s} n={len(q):4d}  {f(R):5.1f} / {f(S):5.1f} / {f(V):5.1f}")
print(f"  vcast vs budget_pool: fixes {sum(V[i]['hit'] and not S[i]['hit'] for i in done)}, "
      f"breaks {sum(S[i]['hit'] and not V[i]['hit'] for i in done)}")

if len(done) == len(mem):
    hits = {"recent16": {i: R[i]["hit"] for i in ids},
            "budget_pool": {i: (R[i] if label[i] == "NOW" else S[i])["hit"] for i in ids},
            "vcast": {i: (R[i] if label[i] == "NOW" else V[i])["hit"] for i in ids}}
    for name, h in hits.items():
        c, o, acc = common.macro(h, ids, sub)
        print(f"{name:12s} BT {c['BT']:.2f} RT {c['RT']:.2f} FT {c['FT']:.2f} Overall {o:.2f}  "
              + " ".join(f"{s} {acc[s]:.1f}" for s in common.GROUP))
    src = common.source_videos(R); videos = sorted(set(src.values())); vi = {v: k for k, v in enumerate(videos)}
    subs = list(common.GROUP); si = {s: k for k, s in enumerate(subs)}
    n = np.zeros((len(videos), len(subs))); H = {k: np.zeros_like(n) for k in hits}
    for i in ids:
        a, b = vi[src[i]], si[sub[i]]; n[a, b] += 1
        for k in hits:
            H[k][a, b] += hits[k][i]
    groups = {g: [si[s] for s in subs if common.GROUP[s] == g] for g in ("BT", "RT", "FT")}
    def score(w, h):
        acc = 100 * (w @ h) / np.maximum(w @ n, 1e-9)
        cat = {g: np.nanmean(acc[:, c], 1) for g, c in groups.items()}
        return cat, sum(cat.values()) / 3
    rng = np.random.default_rng(20261018)
    w = rng.multinomial(len(videos), np.full(len(videos), 1 / len(videos)), size=10000).astype(float)
    one = np.ones((1, len(videos)))
    for other in ("budget_pool", "recent16"):
        pc, po = score(one, H["vcast"]); qc, qo = score(one, H[other]); bc, bo = score(w, H["vcast"]); cc, co = score(w, H[other])
        for metric, p, d in [("Overall", po - qo, bo - co)] + [(g, pc[g] - qc[g], bc[g] - cc[g]) for g in ("BT", "RT", "FT")]:
            lo, hi = np.nanpercentile(d, [2.5, 97.5]); pv = min(1, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
            print(f"vcast vs {other:12s} {metric:7s} {p[0]:+.2f} [{lo:+.2f}, {hi:+.2f}] p={pv:.3f}")

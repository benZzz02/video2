"""Score V-CAST storage under the fixed main method against the stitched 72.99 (budget_pool storage).

NOW questions use the official Recent16 answers in both; SHORT/LONG questions use each storage's run.
On a partial run, compares only finished SHORT/LONG questions (per-subtask accuracy on the same questions).
"""
import glob, json, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, "/home/zhangnuohua/medhorizon/video2-fill-full-context-20261015/analysis/paper-analysis-20261018")
import common

ROOT = "/home/zhangnuohua/medhorizon/video2-vcast-20261020/results/ovo-vcast-succ-short-long-2gpu-20261020"
runs = common.load_runs(); R, S = runs["Recent16"], runs["Ours (routed)"]
V = {json.loads(l)["sample_id"]: json.loads(l) for p in glob.glob(f"{ROOT}/attempts/*/predictions.jsonl")
     for l in open(p) if l.endswith("\n")}
ids = sorted(R); sub = {i: R[i]["subtask"] for i in ids}
label = {i: S[i]["route"]["label"] for i in ids}
mem = [i for i in ids if label[i] != "NOW"]
done = [i for i in mem if i in V]
print(f"finished {len(done)} / {len(mem)} SHORT/LONG questions")
print("router label equal:", sum(V[i]["route"]["label"] == label[i] for i in done), "/", len(done))

by = defaultdict(list)
for i in done:
    by[sub[i]].append(i)
print("\nsame questions, per subtask (budget_pool -> vcast):")
for s in common.GROUP:
    q = by.get(s, [])
    if q:
        a = 100 * np.mean([S[i]["hit"] for i in q]); b = 100 * np.mean([V[i]["hit"] for i in q])
        print(f"  {s:4s} n={len(q):4d}  {a:5.1f} -> {b:5.1f}  ({b - a:+.1f})")
fix = sum(V[i]["hit"] and not S[i]["hit"] for i in done); hurt = sum(S[i]["hit"] and not V[i]["hit"] for i in done)
print(f"  vcast fixes {fix}, breaks {hurt}")

if len(done) == len(mem):
    main = {i: (R[i] if label[i] == "NOW" else S[i])["hit"] for i in ids}
    vc = {i: (R[i] if label[i] == "NOW" else V[i])["hit"] for i in ids}
    for name, h in (("main (budget_pool)", main), ("main (vcast)", vc)):
        c, o, _ = common.macro(h, ids, sub)
        print(f"{name:20s} BT {c['BT']:.2f} RT {c['RT']:.2f} FT {c['FT']:.2f} Overall {o:.2f}")
    src = common.source_videos(R); videos = sorted(set(src.values())); vi = {v: k for k, v in enumerate(videos)}
    subs = list(common.GROUP); si = {s: k for k, s in enumerate(subs)}
    n = np.zeros((len(videos), len(subs))); H = {k: np.zeros_like(n) for k in ("m", "v", "p")}
    P = runs["PHCM (caption memory)"]
    for i in ids:
        a, b = vi[src[i]], si[sub[i]]; n[a, b] += 1
        H["m"][a, b] += main[i]; H["v"][a, b] += vc[i]; H["p"][a, b] += P[i]["hit"]
    groups = {g: [si[s] for s in subs if common.GROUP[s] == g] for g in ("BT", "RT", "FT")}
    def score(w, h):
        acc = 100 * (w @ h) / np.maximum(w @ n, 1e-9)
        cat = {g: np.nanmean(acc[:, c], 1) for g, c in groups.items()}
        return cat, sum(cat.values()) / 3
    rng = np.random.default_rng(20261018)
    w = rng.multinomial(len(videos), np.full(len(videos), 1 / len(videos)), size=10000).astype(float)
    one = np.ones((1, len(videos)))
    for other, key in (("budget_pool main", "m"), ("PHCM", "p")):
        pc, po = score(one, H["v"]); qc, qo = score(one, H[key]); bc, bo = score(w, H["v"]); cc, co = score(w, H[key])
        for metric, p, b in [("Overall", po - qo, bo - co)] + [(g, pc[g] - qc[g], bc[g] - cc[g]) for g in ("BT", "RT", "FT")]:
            lo, hi = np.nanpercentile(b, [2.5, 97.5])
            print(f"vcast vs {other:16s} {metric:7s} {p[0]:+.2f} [{lo:+.2f}, {hi:+.2f}]")

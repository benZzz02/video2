"""Router + visual check: NOW questions with low Recent16 confidence escalate to history.

Threshold and escalation target are chosen on half0 only; half1 is held out.
"""
import glob, hashlib, json
from pathlib import Path

M = Path("/home/zhangnuohua/medhorizon")
R = json.load(open(M / "video2-action-readout-20261013/analysis/recent16-memory-errors-20261014/recent16_predictions.json"))
if isinstance(R, dict):
    R = R.get("predictions") or R.get("samples") or list(R.values())
R = {r["sample_id"]: r for r in R}
L = lambda g: {json.loads(l)["sample_id"]: json.loads(l) for p in glob.glob(str(g)) for l in open(p) if l.endswith("\n")}
F = L(M / "video2/results/ovo-visual-memory-2gpu-20261011/predictions.jsonl")
N = L(M / "video2-fill-full-context-20261015/results/ovo-fill-full-context-v2-2gpu-20261015/attempts/*/predictions.jsonl")
route = L("routes-*.jsonl")
cf = route.pop("__content_free__")["mean_logprob"]
label = {i: max(("NOW", "SHORT", "LONG"), key=lambda l: r["mean_logprob"][l] - cf[l]) for i, r in route.items()}
V = L("vcheck-*.jsonl")
SRC = {"NOW": R, "SHORT": F, "LONG": N}
grp = {'EPM':'BT','HLD':'BT','ASI':'BT','STU':'RT','OJR':'RT','ATR':'RT','FPD':'RT','ACR':'RT','OCR':'RT','CRR':'FT','SSR':'FT','REC':'FT'}
half = {i: int(hashlib.sha256(R[i]["video"].encode()).hexdigest(), 16) % 2 for i in R}
missing = [i for i in R if label[i] == "NOW" and i not in V]
assert not missing, f"{len(missing)} NOW questions lack a visual check"
print("visual check reproduces official answer:", sum(V[i]["prediction"] == R[i]["prediction"] for i in V), "/", len(V))

def pick(i, tau, target):
    lab = label[i]
    if lab == "NOW" and V[i]["p_top1"] < tau:
        lab = target
    return SRC[lab][i]

def score(ids, f):
    acc = {s: 100 * sum(f(i)["hit"] for i in ids if R[i]["subtask"] == s) / sum(R[i]["subtask"] == s for i in ids) for s in grp}
    g = {k: sum(acc[s] for s in grp if grp[s] == k) / sum(grp[s] == k for s in grp) for k in ("BT", "RT", "FT")}
    return g, sum(g.values()) / 3, acc

dev = [i for i in R if half[i] == 0]
test = [i for i in R if half[i] == 1]
grid = [x / 100 for x in range(30, 100, 2)] + [0.99, 0.995, 1.01]
best = max(((score(dev, lambda i: pick(i, t, tg))[1], t, tg) for t in grid for tg in ("SHORT", "LONG")))
_, tau, target = best
print(f"chosen on half0: tau={tau} target={target} (half0 Overall {best[0]:.2f})")
for name, ids in (("half0 (dev)", dev), ("half1 (held out)", test), ("ALL", list(R))):
    print(f"\n{name} n={len(ids)}")
    for m, f in (("Recent16", lambda i: R[i]), ("oldFill", lambda i: F[i]), ("fullctx", lambda i: N[i]),
                 ("router", lambda i: pick(i, -1, "SHORT")), ("router+vcheck", lambda i: pick(i, tau, target))):
        g, o, _ = score(ids, f)
        print(f"  {m:14s} BT {g['BT']:.2f} RT {g['RT']:.2f} FT {g['FT']:.2f} Overall {o:.2f}")
esc = [i for i in R if label[i] == "NOW" and V[i]["p_top1"] < tau]
print("\nescalated per subtask:", {s: sum(R[i]["subtask"] == s for i in esc) for s in grp})
_, _, acc = score(test, lambda i: pick(i, tau, target))
print("half1 router+vcheck per subtask:", {s: round(v, 1) for s, v in acc.items()})

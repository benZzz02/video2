"""Combine existing predictions by router label; report dev/test halves split by video."""
import collections, hashlib, json, glob
from pathlib import Path

M = Path("/home/zhangnuohua/medhorizon")
R = json.load(open(M / "video2-action-readout-20261013/analysis/recent16-memory-errors-20261014/recent16_predictions.json"))
if isinstance(R, dict):
    R = R.get("predictions") or R.get("samples") or list(R.values())
R = {r["sample_id"]: r for r in R}
L = lambda g: {json.loads(l)["sample_id"]: json.loads(l) for p in glob.glob(str(g)) for l in open(p) if l.endswith("\n")}
F = L(M / "video2/results/ovo-visual-memory-2gpu-20261011/predictions.jsonl")
N = L(M / "video2-fill-full-context-20261015/results/ovo-fill-full-context-v2-2gpu-20261015/attempts/*/predictions.jsonl")
import sys
route = L("routes-*.jsonl")
cf = route.pop("__content_free__")["mean_logprob"]
if "--calibrate" in sys.argv:
    for r in route.values():
        r["label"] = max(("NOW", "SHORT", "LONG"), key=lambda l: r["mean_logprob"][l] - cf[l])
print("content-free logprobs:", {k: round(v, 2) for k, v in cf.items()}, "| calibrate:", "--calibrate" in sys.argv)
SRC = {"NOW": R, "SHORT": F, "LONG": N}
grp = {'EPM':'BT','HLD':'BT','ASI':'BT','STU':'RT','OJR':'RT','ATR':'RT','FPD':'RT','ACR':'RT','OCR':'RT','CRR':'FT','SSR':'FT','REC':'FT'}
half = lambda i: int(hashlib.sha256(R[i]["video"].encode()).hexdigest(), 16) % 2

def score(ids, pick):
    acc = {}
    for s in grp:
        x = [i for i in ids if R[i]["subtask"] == s]
        acc[s] = 100 * sum(pick(i)["hit"] for i in x) / len(x)
    g = {k: sum(acc[s] for s in grp if grp[s] == k) / sum(grp[s] == k for s in grp) for k in ("BT", "RT", "FT")}
    return g, sum(g.values()) / 3, acc

print("label distribution per subtask (NOW/SHORT/LONG):")
for s in grp:
    c = collections.Counter(route[i]["label"] for i in R if R[i]["subtask"] == s)
    print(f"  {s}: {c['NOW']:4d} {c['SHORT']:4d} {c['LONG']:4d}")
methods = {"Recent16": lambda i: R[i], "oldFill": lambda i: F[i], "fullctx": lambda i: N[i],
           "router": lambda i: SRC[route[i]["label"]][i]}
for name, ids in (("ALL", list(R)), ("half0", [i for i in R if half(i) == 0]), ("half1", [i for i in R if half(i) == 1])):
    print(f"\n{name} (n={len(ids)})")
    for m, pick in methods.items():
        g, o, _ = score(ids, pick)
        print(f"  {m:9s} BT {g['BT']:.2f} RT {g['RT']:.2f} FT {g['FT']:.2f} Overall {o:.2f}")
_, _, acc = score(list(R), methods["router"])
print("\nrouter per subtask:", {s: round(v, 1) for s, v in acc.items()})

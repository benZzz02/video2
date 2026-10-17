"""Per-question accounting for the routed run: what memory added, cost, and left unused."""
import collections, glob, json
from pathlib import Path

M = Path("/home/zhangnuohua/medhorizon")
B = M / "video2-fill-full-context-20261015/results/ovo-routed-vcheck-2gpu-20261017"
L = lambda g: {json.loads(l)["sample_id"]: json.loads(l) for p in glob.glob(str(g)) for l in open(p) if l.endswith("\n")}
S = L(B / "attempts/*/predictions.jsonl")
R = json.load(open(M / "video2-action-readout-20261013/analysis/recent16-memory-errors-20261014/recent16_predictions.json"))
R = {r["sample_id"]: r for r in (R.get("predictions") or R.get("samples") if isinstance(R, dict) else R)}
F = L(M / "video2/results/ovo-visual-memory-2gpu-20261011/predictions.jsonl")
N = L(M / "video2-fill-full-context-20261015/results/ovo-fill-full-context-v2-2gpu-20261015/predictions.jsonl")
GROUP = {'EPM':'BT','HLD':'BT','ASI':'BT','STU':'RT','OJR':'RT','ATR':'RT','FPD':'RT','ACR':'RT','OCR':'RT','CRR':'FT','SSR':'FT','REC':'FT'}

def path(i):
    r = S[i]["route"]
    if r["label"] == "NOW":
        return "NOW-kept" if r["final_scope"] == "NOW" else "NOW-escalated"
    return r["label"]

print(f"finished {len(S)}\n")
print("path            n   R16  sys  | sys vs R16 +/-  | any-version-right  oracle-gap")
rows = collections.defaultdict(list)
for i in S:
    rows[(S[i]["subtask"], path(i))].append(i)
by_path = collections.defaultdict(list)
for (s, p), ids in rows.items():
    by_path[p] += ids
for p in ("NOW-kept", "NOW-escalated", "SHORT", "LONG"):
    ids = by_path[p]
    pct = lambda f: 100 * sum(map(f, ids)) / len(ids)
    anyr = lambda i: R[i]["hit"] or F[i]["hit"] or N[i]["hit"] or S[i]["hit"]
    print(f"{p:14s} {len(ids):4d} {pct(lambda i: R[i]['hit']):5.1f} {pct(lambda i: S[i]['hit']):5.1f} | "
          f"+{sum(S[i]['hit'] and not R[i]['hit'] for i in ids):3d} -{sum(R[i]['hit'] and not S[i]['hit'] for i in ids):3d}"
          f"        | {pct(anyr):5.1f}  {pct(anyr) - pct(lambda i: S[i]['hit']):5.1f}")

print("\nper subtask x path: n, R16 -> sys, (best alternative among R16/Fill/fullctx)")
for s in GROUP:
    parts = []
    for p in ("NOW-kept", "NOW-escalated", "SHORT", "LONG"):
        ids = rows.get((s, p), [])
        if not ids:
            continue
        f = lambda X: round(100 * sum(X[i]["hit"] for i in ids) / len(ids))
        parts.append(f"{p} {len(ids)}: {f(R)}->{f(S)} (F {f(F)}, N {f(N)})")
    if parts:
        print(f"  {s}: " + " | ".join(parts))

# Missed gains: system wrong, but a memory version right while R16 wrong (history needed, not used well)
missed = [i for i in S if not S[i]["hit"] and not R[i]["hit"] and (F[i]["hit"] or N[i]["hit"])]
# Losses: R16 right, system wrong
lost = [i for i in S if R[i]["hit"] and not S[i]["hit"]]
print("\nmissed history gains by path:", collections.Counter(path(i) for i in missed))
print("losses vs R16 by path:", collections.Counter(path(i) for i in lost))
json.dump(dict(missed=missed, lost=lost), open("case_lists.json", "w"))

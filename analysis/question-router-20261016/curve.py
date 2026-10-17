exec(open("combine_vcheck.py").read().split("dev = [")[0])
dev = [i for i in R if half[i] == 0]; test = [i for i in R if half[i] == 1]
for tg in ("SHORT", "LONG"):
    print("target", tg)
    for t in (0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98):
        row = []
        for ids in (dev, test):
            g, o, acc = score(ids, lambda i: pick(i, t, tg))
            row.append(f"RT {g['RT']:.2f} FT {g['FT']:.2f} BT {g['BT']:.2f} O {o:.2f}")
        n = sum(label[i] == "NOW" and V[i]["p_top1"] < t for i in R)
        print(f"  tau {t:.2f} esc {n:4d} | half0 {row[0]} | half1 {row[1]}")
# Separability: among NOW questions, does low confidence predict that history helps?
import statistics as st
for s in ("CRR", "STU", "OJR", "SSR", "EPM"):
    x = [i for i in R if label[i] == "NOW" and R[i]["subtask"] == s]
    lo = [i for i in x if V[i]["p_top1"] < 0.8]; hi = [i for i in x if V[i]["p_top1"] >= 0.8]
    f = lambda ids, X: 100 * sum(X[i]["hit"] for i in ids) / max(len(ids), 1)
    print(f"{s}: conf<0.8 n={len(lo)} R16 {f(lo, R):.0f} oldF {f(lo, F):.0f} | conf>=0.8 n={len(hi)} R16 {f(hi, R):.0f} oldF {f(hi, F):.0f}")

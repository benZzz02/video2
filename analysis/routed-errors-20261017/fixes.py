"""Quantify specific failure modes and two cheap fixes on finished questions."""
import collections
exec(open("accounting.py").read().split('print(f"finished')[0])
hit = lambda X, ids: sum(X[i]["hit"] for i in ids)

# 1. Memory path but no history actually available (short video): system answers with the
#    memory prompt + FIFO frames instead of the official Recent16 path.
z = [i for i in S if path(i) in ("SHORT", "LONG", "NOW-escalated") and S[i]["history_read_frames"] == 0]
print(f"[1] memory path with zero history: n={len(z)} R16 {hit(R, z)} vs system {hit(S, z)}  by subtask {collections.Counter(S[i]['subtask'] for i in z)}")

# 2. Yes/No direction of CRR/SSR flips after reading history
for s in ("CRR", "SSR"):
    ids = [i for i in S if S[i]["subtask"] == s and S[i]["history_read_frames"] > 0]
    flips = collections.Counter((R[i]["parsed_output"], S[i]["parsed_output"], S[i]["answer"].lower()) for i in ids
                                if R[i]["parsed_output"] != S[i]["parsed_output"])
    print(f"[2] {s} R16->system flips (R16, sys, gt):", dict(flips))

# 3. Escalation target from the router's own SHORT-vs-LONG preference instead of always SHORT
cf = None
esc = [i for i in S if path(i) == "NOW-escalated"]
def second(i):
    r = S[i]["route"]; lp, c = r["mean_logprob"], r["content_free_logprob"]
    return "LONG" if lp["LONG"] - c["LONG"] > lp["SHORT"] - c["SHORT"] else "SHORT"
alt = {i: (N if second(i) == "LONG" else F)[i] for i in esc}
print(f"[3] escalated n={len(esc)}: always-SHORT(sys) {hit(S, esc)}  router-second-choice {sum(alt[i]['hit'] for i in esc)}  "
      f"oldFill {hit(F, esc)} fullctx {hit(N, esc)} R16 {hit(R, esc)}")
for s in sorted({S[i]['subtask'] for i in esc}):
    x = [i for i in esc if S[i]["subtask"] == s]
    print(f"     {s}: n={len(x)} sys {hit(S, x)} second-choice {sum(alt[i]['hit'] for i in x)} R16 {hit(R, x)} share LONG {sum(second(i)=='LONG' for i in x)}")

# 4. SHORT coverage in seconds depends on resolution (cap is in tokens, not time)
import statistics as st
for s in ("EPM", "ATR", "ACR", "SSR", "CRR"):
    x = [i for i in S if S[i]["subtask"] == s and S[i]["route"]["final_scope"] == "SHORT" and S[i]["history_timestamps"]]
    if x:
        span = [S[i]["recent_timestamps"][0] - S[i]["history_timestamps"][0] for i in x]
        print(f"[4] {s} SHORT lookback seconds median {st.median(span):.0f} (min {min(span):.0f}, max {max(span):.0f})")

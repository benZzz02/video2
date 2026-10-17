import json, sys
exec(open("accounting.py").read().split('print(f"finished')[0])
c = json.load(open("case_lists.json"))
def show(i, tag):
    s, r = S[i], S[i]["route"]
    h = s["history_timestamps"]
    print(f"{tag} {i} {s['subtask']} path={path(i)} conf={r.get('recent_confidence', float('nan')):.2f} "
          f"gt={s['answer']} R16={R[i]['parsed_output']} sys={s['parsed_output']} F={F[i]['parsed_output']} N={N[i]['parsed_output']} "
          f"hist={len(h)}fr/{s['history_visual_tokens']}tok span={(h[0] if h else 0):.0f}-{(h[-1] if h else 0):.0f}s end={s['end']:.0f} "
          f"| {s['question'].strip().splitlines()[-1][:90]}")
for i in sorted(c["lost"], key=lambda i: (path(i), S[i]["subtask"])):
    show(i, "LOST")
for i in sorted(c["missed"], key=lambda i: (path(i), S[i]["subtask"])):
    show(i, "MISS")

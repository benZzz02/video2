"""Print question/options/answers and evidence timing for the early full-run cases."""
import glob, json, sys
from pathlib import Path

M = Path("/home/zhangnuohua/medhorizon")
CASES = sys.argv[1:]
R = json.load(open(M / "video2-action-readout-20261013/analysis/recent16-memory-errors-20261014/recent16_predictions.json"))
if isinstance(R, dict):
    R = R.get("predictions") or R.get("samples") or list(R.values())
R = {r["sample_id"]: r for r in R}
load = lambda p: {json.loads(l)["sample_id"]: json.loads(l) for l in open(p) if l.endswith("\n")}
F = load(M / "video2/results/ovo-visual-memory-2gpu-20261011/predictions.jsonl")
N = {}
for p in glob.glob(str(M / "video2-fill-full-context-20261015/results/ovo-fill-full-context-2gpu-20261015/attempts/*/predictions.jsonl")):
    N.update(load(p))
for i in CASES:
    n = N[i]
    print(f"== {i} {n['subtask']} end={n['end']} video={Path(n['media']).name}")
    print("Q:", n["question"].strip().splitlines()[-1][:200])
    print("options:", n["options"], "| GT", n["answer"])
    print(f"R={R[i]['parsed_output']} oldF={F[i]['parsed_output']} new={n['parsed_output']}")
    print(f"recent R {R[i]['recent_timestamps'][0]:.2f}-{R[i]['recent_timestamps'][-1]:.2f} | new {n['recent_timestamps'][0]:.2f}-{n['recent_timestamps'][-1]:.2f}")
    h = n["history_timestamps"]
    print(f"history oldF {len(F[i]['history_timestamps'])} frames from {F[i]['history_timestamps'][0] if F[i]['history_timestamps'] else '-'} | new {len(h)} frames {h[0] if h else '-'}-{h[-1] if h else '-'}, tokens/frame {min(n['history_frame_tokens'], default=0)}-{max(n['history_frame_tokens'], default=0)}")

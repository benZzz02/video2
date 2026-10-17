"""Visual check for NOW-routed questions: rerun the official Recent16 answer path
and record the full-vocabulary probability of its first answer token.

The answer must reproduce official Recent16; the confidence decides whether the
current frames suffice or the question escalates to stored history.
"""
import argparse, glob, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Eval/Proactive_Eval/OneStreamer/src"))
sys.path.insert(0, str(ROOT / "Eval/Proactive_Eval"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, nargs="+", required=True)
    ap.add_argument("--routes", default="routes-*.jsonl")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    import torch
    from onestreamer.engine import OneStreamerEngine
    from onestreamer.ovobench import build_question_text, score_prediction
    from onestreamer.prompts import MEMORY_DIRECT_QA_SYSTEM, MEMORY_QA_SYSTEM
    from onestreamer.schema import Sample
    from onestreamer.video import read_recent_frames

    routes = {json.loads(l)["sample_id"]: json.loads(l) for p in glob.glob(args.routes) for l in open(p)}
    cf = routes.pop("__content_free__")["mean_logprob"]
    label = lambda r: max(("NOW", "SHORT", "LONG"), key=lambda l: r["mean_logprob"][l] - cf[l])
    rows = [r for p in args.samples for r in json.load(open(p))["samples"]]
    rows = sorted((r for r in rows if label(routes[r["sample_id"]]) == "NOW"), key=lambda r: r["sample_id"])
    rows = rows[args.shard::args.num_shards]
    engine = OneStreamerEngine(model_path="/mnt/truenas_nfs/dataset/znh1302/OneStreamer", dtype="bfloat16",
                               device="cuda", attn_implementation="sdpa")
    original = engine.model.generate
    captured = {}

    def generate(**kwargs):
        out = original(**kwargs, output_scores=True, return_dict_in_generate=True)
        captured["scores"] = out.scores
        return out.sequences

    engine.model.generate = generate
    tok = engine.processor.tokenizer
    done = set()
    if args.output.exists():
        done = {json.loads(l)["sample_id"] for l in open(args.output)}
    with args.output.open("a") as out:
        for n, row in enumerate(rows):
            if row["sample_id"] in done:
                continue
            sample = Sample.from_dict(row)
            t0 = time.time()
            frames = read_recent_frames(row["media"], sample.start, sample.end, recent_frames=16, sample_fps=1.0)
            prediction = engine.answer(
                system=MEMORY_QA_SYSTEM if sample.eval_mode == "mcq_letter" else MEMORY_DIRECT_QA_SYSTEM,
                memory_text="", recent_frames=frames, question_text=build_question_text(sample),
                max_tokens=16, recent_resize_policy="source_pixels")
            probs = captured["scores"][0][0].float().softmax(-1)
            top = probs.topk(2)
            _, hit = score_prediction(sample, prediction)
            out.write(json.dumps(dict(
                sample_id=row["sample_id"], prediction=prediction, hit=hit,
                p_top1=float(top.values[0]), p_top2=float(top.values[1]),
                top_tokens=[tok.decode([int(i)]) for i in top.indices],
                recent_timestamps=[t for t, _ in frames], seconds=time.time() - t0)) + "\n")
            out.flush()
            if n % 50 == 0:
                print(n, len(rows), flush=True)


if __name__ == "__main__":
    main()

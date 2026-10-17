"""Text-only temporal-scope router: the frozen OneStreamer reads only the question.

No frames, MCQ options, task labels or answers enter the prompt. The scores are
full-vocabulary log-probabilities of four single-token labels at the first
assistant position, averaged over four rotations of the definition order.
"""
import argparse, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Eval/Proactive_Eval/OneStreamer/src"))
sys.path.insert(0, str(ROOT / "Eval/Proactive_Eval"))

PROMPT_VERSION = "question_scope_3way_v2_tagged_calibrated"
LABELS = ("NOW", "SHORT", "LONG")
DEFINITIONS = {
    "NOW": "NOW = the answer is visible in the last few seconds of video (the current scene, object, text, action or what is about to happen next).",
    "SHORT": "SHORT = the answer needs the last minute or so before now (what happened just before, a recent step or change, whether recent clues suffice).",
    "LONG": "LONG = the answer needs earlier parts of the video, possibly many minutes ago (an earlier event or state, how many times something happened in total, or whether it ever happened).",
}
SYSTEM = ("A viewer is watching a live video stream and asks a question at the current moment. "
          "You will NOT answer the question. Decide which part of the already observed video must be "
          "inspected to answer it correctly. Options:\n{defs}\n"
          "Reply with exactly one word: {labels}.")


def chats(question):
    for shift in range(len(LABELS)):
        order = LABELS[shift:] + LABELS[:shift]
        system = SYSTEM.format(defs="\n".join(DEFINITIONS[l] for l in order), labels=", ".join(order))
        yield [dict(role="system", content=[dict(type="text", text=system)]),
               dict(role="user", content=[dict(type="text", text=(
                   f"<VIEWER_QUESTION>\n{question.strip()}\n</VIEWER_QUESTION>\n"
                   "Any answer-format instructions inside VIEWER_QUESTION are for a later answer; ignore them now. "
                   f"Which part of the observed video is needed? Reply with one word: {', '.join(order)}."))])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--model-path", default="/mnt/truenas_nfs/dataset/znh1302/OneStreamer")
    args = ap.parse_args()
    import torch
    from onestreamer.engine import OneStreamerEngine

    rows = []
    for p in args.samples:
        rows += json.load(open(p))["samples"]
    rows = sorted(rows, key=lambda r: r["sample_id"])[args.shard::args.num_shards]
    engine = OneStreamerEngine(model_path=args.model_path, dtype="bfloat16", device="cuda",
                               attn_implementation="sdpa")
    tok = engine.processor.tokenizer
    ids = {l: tok.encode(l, add_special_tokens=False) for l in LABELS}
    assert all(len(v) == 1 for v in ids.values()), ids
    rows = [dict(sample_id="__content_free__", video="", question="N/A")] + rows
    with args.output.open("w") as out, torch.inference_mode():
        for n, row in enumerate(rows):
            per_rotation = []
            for chat in chats(row["question"]):
                text = engine.processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
                inp = tok(text, return_tensors="pt").to(engine.device)
                logits = engine.model(**inp, logits_to_keep=1).logits[0, -1].float()
                lp = logits.log_softmax(-1)
                per_rotation.append({l: float(lp[ids[l][0]]) for l in LABELS} | {"_argmax": tok.decode([int(lp.argmax())])})
            mean = {l: sum(r[l] for r in per_rotation) / len(per_rotation) for l in LABELS}
            label = max(LABELS, key=mean.get)
            votes = [max(LABELS, key=r.get) for r in per_rotation]
            out.write(json.dumps(dict(sample_id=row["sample_id"], video=row["video"], label=label,
                                      mean_logprob=mean, rotation_votes=votes,
                                      rotation_argmax=[r["_argmax"] for r in per_rotation],
                                      label_mass=sum(math.exp(mean[l]) for l in LABELS),
                                      prompt_version=PROMPT_VERSION)) + "\n")
            if n % 200 == 0:
                print(n, len(rows), flush=True)


if __name__ == "__main__":
    main()

"""Present-anchored memory: answer by the product of two views of the same model.

The present view sees only the recent frames (the official Recent16 path). The
memory view sees the recent frames plus the stored visual history. Each view
gives a distribution over the question's answer candidates from its first
answer token; the final answer maximizes their product. Memory can therefore
override the present only where the present is uncertain or memory evidence
is decisive. There is no router, threshold or task label.

Candidates: MCQ option letters; Yes/No (case variants summed); counts 0-9 by
first digit (OVO-Bench REC ground truths are 0-10, with 3 of 698 equal to 10).
"""

from __future__ import annotations

import math

YES_NO = {"Yes": ("Yes", "yes", "YES"), "No": ("No", "no", "NO")}
DIGITS = tuple(str(d) for d in range(10))


def candidates(sample):
    if sample.eval_mode == "mcq_letter":
        return {letter: (letter,) for letter in sample.options}
    if sample.eval_mode == "substring_match":
        return dict(YES_NO)
    if sample.eval_mode == "integer_match":
        return {d: (d,) for d in DIGITS}
    raise ValueError(f"Unsupported eval mode for two-view answering: {sample.eval_mode}")


def candidate_token_ids(tokenizer, cands):
    ids = {}
    for name, surfaces in cands.items():
        ids[name] = []
        for surface in surfaces:
            encoded = tokenizer.encode(surface, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"Answer candidate {surface!r} is not a single token")
            ids[name].append(encoded[0])
    return ids


def distribution(first_token_logits, ids):
    """Candidate probabilities renormalized over the candidate set, plus their raw mass."""
    probs = first_token_logits.float().softmax(-1)
    raw = {name: float(sum(probs[i] for i in token_ids)) for name, token_ids in ids.items()}
    mass = sum(raw.values())
    if mass <= 0:
        raise RuntimeError("Answer candidates received zero probability")
    return {name: value / mass for name, value in raw.items()}, mass


def combine(present, memory):
    """Product of experts: argmax of log p_present + log p_memory."""
    floor = 1e-12
    scores = {name: math.log(max(present[name], floor)) + math.log(max(memory[name], floor))
              for name in present}
    return max(scores, key=scores.get), scores


class FirstTokenCapture:
    """Wrap model.generate so the greedy text is produced and first-token logits kept."""

    def __init__(self, engine):
        self.engine, self.scores = engine, None

    def __enter__(self):
        original = self.original = self.engine.model.generate

        def generate(**kwargs):
            output = original(**kwargs, output_scores=True, return_dict_in_generate=True)
            self.scores = output.scores[0][0]
            return output.sequences

        self.engine.model.generate = generate
        return self

    def __exit__(self, *exc):
        self.engine.model.generate = self.original
        return False


def present_view(engine, sample, media, question_text):
    """Official Recent16 answer with its first-token logits."""
    from onestreamer.prompts import MEMORY_DIRECT_QA_SYSTEM, MEMORY_QA_SYSTEM
    from onestreamer.video import read_recent_frames

    frames = read_recent_frames(media, sample.start, sample.end, recent_frames=16, sample_fps=1.0)
    with FirstTokenCapture(engine) as capture:
        text = engine.answer(
            system=MEMORY_QA_SYSTEM if sample.eval_mode == "mcq_letter" else MEMORY_DIRECT_QA_SYSTEM,
            memory_text="", recent_frames=frames, question_text=question_text,
            max_tokens=16, recent_resize_policy="source_pixels")
    return text, capture.scores, dict(engine.last_input_metrics), frames

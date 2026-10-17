"""Query-time routing between recent-only answering and stored visual history.

Memory writing never sees the question. When a question arrives:

1. The frozen model reads only the question text (no frames, options or task
   labels) and scores three single-token scopes: NOW, SHORT, LONG. Scores are
   full-vocabulary log-probabilities averaged over rotations of the definition
   order, minus those of a content-free question ("N/A").
2. NOW questions are answered by the official Recent16 path. If the
   probability of its first answer token is below a threshold, the question
   escalates to stored history.
3. SHORT reads the newest stored records up to a visual-token cap; LONG fills
   the remaining context with the whole stored history.
"""

from __future__ import annotations

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
CONTENT_FREE_QUESTION = "N/A"


def scope_chats(question):
    for shift in range(len(LABELS)):
        order = LABELS[shift:] + LABELS[:shift]
        system = SYSTEM.format(defs="\n".join(DEFINITIONS[l] for l in order), labels=", ".join(order))
        yield [dict(role="system", content=[dict(type="text", text=system)]),
               dict(role="user", content=[dict(type="text", text=(
                   f"<VIEWER_QUESTION>\n{question.strip()}\n</VIEWER_QUESTION>\n"
                   "Any answer-format instructions inside VIEWER_QUESTION are for a later answer; ignore them now. "
                   f"Which part of the observed video is needed? Reply with one word: {', '.join(order)}."))])]


def choose_scope(mean_logprob, content_free):
    return max(LABELS, key=lambda label: mean_logprob[label] - content_free[label])


class QuestionRouter:
    def __init__(self, engine):
        tokenizer = engine.processor.tokenizer
        self.engine = engine
        self.label_ids = {}
        for label in LABELS:
            encoded = tokenizer.encode(label, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"Scope label {label} is not a single token")
            self.label_ids[label] = encoded[0]
        self.content_free = self._score(CONTENT_FREE_QUESTION)

    def _score(self, question):
        import torch

        engine, tokenizer = self.engine, self.engine.processor.tokenizer
        total = dict.fromkeys(LABELS, 0.0)
        rotations = 0
        with torch.inference_mode():
            for chat in scope_chats(question):
                text = engine.processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
                inputs = tokenizer(text, return_tensors="pt").to(engine.device)
                logprobs = engine.model(**inputs, logits_to_keep=1).logits[0, -1].float().log_softmax(-1)
                for label, token_id in self.label_ids.items():
                    total[label] += float(logprobs[token_id])
                rotations += 1
        return {label: value / rotations for label, value in total.items()}

    def route(self, question):
        scores = self._score(question)
        return dict(label=choose_scope(scores, self.content_free), mean_logprob=scores,
                    content_free_logprob=self.content_free, prompt_version=PROMPT_VERSION)


def official_recent_answer(engine, sample, media, question_text):
    """Official Recent16 answer (end-aligned frames, empty memory) and its first-token probability."""
    from onestreamer.prompts import MEMORY_DIRECT_QA_SYSTEM, MEMORY_QA_SYSTEM
    from onestreamer.video import read_recent_frames

    frames = read_recent_frames(media, sample.start, sample.end, recent_frames=16, sample_fps=1.0)
    original = engine.model.generate
    captured = {}

    def generate(**kwargs):
        output = original(**kwargs, output_scores=True, return_dict_in_generate=True)
        captured["scores"] = output.scores
        return output.sequences

    engine.model.generate = generate
    try:
        prediction = engine.answer(
            system=MEMORY_QA_SYSTEM if sample.eval_mode == "mcq_letter" else MEMORY_DIRECT_QA_SYSTEM,
            memory_text="", recent_frames=frames, question_text=question_text,
            max_tokens=16, recent_resize_policy="source_pixels")
    finally:
        engine.model.generate = original
    confidence = float(captured["scores"][0][0].float().softmax(-1).max())
    return prediction, dict(engine.last_input_metrics), confidence, frames


def newest_history(candidates, max_visual_tokens):
    """Newest stored records whose visual tokens fit the cap, in time order."""
    selected, used = [], 0
    for frame in sorted(candidates, key=lambda f: f.timestamp, reverse=True):
        if used + frame.token_count > max_visual_tokens:
            break
        selected.append(frame)
        used += frame.token_count
    return selected[::-1]

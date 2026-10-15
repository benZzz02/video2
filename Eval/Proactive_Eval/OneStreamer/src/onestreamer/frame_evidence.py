"""Frame-ID witness prompt and structural audit for the exploratory full run.

The prompt is the six-case v2 pilot prompt for MCQ and integer questions.
Frame IDs identify positions in the stored, eligible history at this query
(not positions in every frame of the source video). A valid citation only
means that the named image was supplied; it does not certify visual support.
"""

from __future__ import annotations

import re


PROMPT_VERSION = "frame_id_witness_v2"
ANSWER_RESERVE_TOKENS = 128
SYSTEM = (
    "Answer the video question using only the supplied visual frames. Each "
    "image is immediately preceded by a unique frame ID and timestamp. H IDs "
    "are sparse earlier observations; R IDs are the recent observations. "
    "First locate the particular entity, action, and time interval asked about. "
    "Cite only IDs of supplied images that directly support the chosen answer "
    "for that same entity or event; never infer what an unseen frame contains. "
    "For a before/after relation, cite at least one frame for each side, with "
    "IDs ordered by time. For a count, cite one representative frame near each "
    "distinct completed occurrence, in time order; repeated views of one "
    "occurrence do not create new occurrences. An older attribute of a different "
    "object cannot override a newer observation of the queried object. "
    "An earlier observation may establish an earlier action or a distinct "
    "event within the requested interval. Missing sparse frames do not prove "
    "absence. If no supplied image supports the answer, cite NONE. Do not "
    "describe frames or invent unseen events. Follow the two-line pilot response "
    "format even if the question asks for an answer-only response."
)
FOOTER = (
    "\n\nFor this evidence pilot, give exactly two lines:\n"
    "ANSWER: <one uppercase option letter, or one integer when the question asks for a count>\n"
    "EVIDENCE: <comma-separated exact frame IDs in chronological order, or NONE>\n"
    "Use only the IDs immediately preceding the supplied images; do not use timestamps or ranges."
)
SUBSTRING_FOOTER = (
    "\n\nFor this evidence pilot, give exactly two lines:\n"
    "ANSWER: <a short answer to the question>\n"
    "EVIDENCE: <comma-separated exact frame IDs in chronological order, or NONE>\n"
    "Use only the IDs immediately preceding the supplied images; do not use timestamps or ranges."
)
FRAME_ID = re.compile(r"[HR]\d{2,}")


def frame_specs(history, recent, history_indices):
    """Give selected frames stable IDs from the query's eligible history list."""
    if not recent:
        raise ValueError("Frame-ID prompt requires at least one recent image")
    if len({f.timestamp for f in history}) != len(history):
        raise ValueError("Historical timestamps must be unique")
    specs = []
    for frame in history:
        if frame.timestamp not in history_indices:
            raise ValueError("Selected historical frame has no eligible-history ID")
        index = history_indices[frame.timestamp]
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("Historical frame ID must be a nonnegative integer")
        specs.append(dict(id=f"H{index:02d}", timestamp=float(frame.timestamp),
                          section="history", source_index=index))
    specs.extend(dict(id=f"R{i:02d}", timestamp=float(frame.timestamp),
                      section="recent", source_index=i)
                 for i, frame in enumerate(recent))
    if (len({spec["id"] for spec in specs}) != len(specs)
            or any(a["timestamp"] >= b["timestamp"] for a, b in zip(specs, specs[1:]))):
        raise ValueError("Frame-ID mapping is nonunique or nonchronological")
    return specs


def build_chat(adapter, history, recent, question, direct, specs, eval_mode):
    """Port v2's image-adjacent IDs and answer footer without changing options."""
    if eval_mode not in {"mcq_letter", "integer_match", "substring_match"}:
        raise ValueError(f"Unsupported frame-evidence answer mode: {eval_mode}")
    chat = adapter.build_chat(history, recent, question, direct=direct)
    chat[0]["content"] = [{"type": "text", "text": SYSTEM}]
    content = chat[1]["content"]
    image_positions = [i for i, part in enumerate(content) if part["type"] == "image"]
    if len(image_positions) != len(specs):
        raise RuntimeError("Frame-ID map does not match prompt images")
    for i, spec in zip(image_positions, specs):
        expected = f'<{spec["timestamp"]:.3f}s>'
        if i == 0 or content[i - 1] != {"type": "text", "text": expected}:
            raise RuntimeError("Timestamp/image adjacency changed")
        content[i - 1] = {"type": "text", "text": f'[{spec["id"]}] {expected}'}
    if content[-1] != {"type": "text", "text": question}:
        raise RuntimeError("Original question/options changed")
    content.append({"type": "text", "text":
                    SUBSTRING_FOOTER if eval_mode == "substring_match" else FOOTER})
    return chat


def evidence_requirement(eval_mode, question):
    """Structural audit only: this does not choose a readout view."""
    if eval_mode == "integer_match":
        return "one_id_per_claimed_occurrence"
    if re.search(r"\b(?:before|after)\b", question, flags=re.I):
        return "ordered_relation_pair"
    return "at_least_one_id"


def parse_witness(text, specs, eval_mode, question, valid_letters=None):
    """Port the strict v2 audit; never infer whether cited pixels support an answer."""
    lines = text.strip().splitlines()
    format_valid = (len(lines) == 2 and bool(re.fullmatch(r"ANSWER: .+", lines[0]))
                    and bool(re.fullmatch(r"EVIDENCE: .+", lines[1])))
    answer_match = re.search(r"(?m)^ANSWER:[ \t]*([^\n\r]+)$", text)
    evidence_match = re.search(r"(?m)^EVIDENCE:[ \t]*([^\n\r]+)$", text)
    answer = answer_match.group(1).strip() if answer_match else None
    if eval_mode == "mcq_letter":
        allowed = valid_letters or "ABCD"
        answer_valid = bool(answer and len(answer) == 1 and answer in allowed)
    elif eval_mode == "integer_match":
        answer_valid = bool(answer and re.fullmatch(r"\d+", answer))
    elif eval_mode == "substring_match":
        answer_valid = bool(answer)
    else:
        raise ValueError(f"Unsupported frame-evidence answer mode: {eval_mode}")
    evidence = evidence_match.group(1).strip() if evidence_match else None
    if evidence == "NONE":
        ids, syntax_valid = [], True
    elif evidence is None:
        ids, syntax_valid = [], False
    else:
        parts = [part.strip() for part in evidence.split(",")]
        syntax_valid = bool(parts) and all(FRAME_ID.fullmatch(part) for part in parts)
        ids = parts if syntax_valid else []
    frame_by_id = {spec["id"]: spec for spec in specs}
    unknown_ids = [fid for fid in ids if fid not in frame_by_id]
    all_citations_supplied = syntax_valid and not unknown_ids
    duplicate_ids = len(ids) != len(set(ids))
    cited_times = [frame_by_id[fid]["timestamp"] for fid in ids if fid in frame_by_id]
    chronology_valid = all_citations_supplied and all(
        a < b for a, b in zip(cited_times, cited_times[1:]))
    requirement = evidence_requirement(eval_mode, question)
    if requirement == "one_id_per_claimed_occurrence":
        cardinality_valid = bool(answer_valid and ids and len(ids) == int(answer))
    elif requirement == "ordered_relation_pair":
        cardinality_valid = len(ids) >= 2
    else:
        cardinality_valid = len(ids) >= 1
    protocol_valid = bool(format_valid and answer_valid and all_citations_supplied
                          and not duplicate_ids and chronology_valid)
    return {
        "answer": answer, "answer_valid": answer_valid, "format_valid": format_valid,
        "evidence_raw": evidence, "cited_ids": ids, "cited_times": cited_times,
        "evidence_syntax_valid": syntax_valid, "unknown_ids": unknown_ids,
        "all_citations_supplied": all_citations_supplied,
        "duplicate_ids": duplicate_ids, "chronology_valid": chronology_valid,
        "evidence_requirement": requirement,
        "required_cardinality_valid": cardinality_valid,
        "citation_protocol_valid": protocol_valid,
        "witness_structure_valid": protocol_valid and cardinality_valid,
        "cited_history_ids": [fid for fid in ids if fid in frame_by_id
                              and frame_by_id[fid]["section"] == "history"],
        "cited_recent_ids": [fid for fid in ids if fid in frame_by_id
                             and frame_by_id[fid]["section"] == "recent"],
    }


def answer_for_scoring(witness):
    """Score the valid ANSWER line independently of citation format."""
    return witness["answer"] if witness["answer_valid"] else ""

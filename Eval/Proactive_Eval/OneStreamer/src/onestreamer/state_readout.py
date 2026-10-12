"""Bounded, caption-free historical feature reading controlled by native states.

The controller never receives an evaluation task label.  It asks the frozen
model whether the current visual evidence is sufficient, beginning with the
recent window and progressively exposing nested historical feature bundles.
Neither probe outputs nor descriptions are written into the visual memory.
"""

from __future__ import annotations

import math
from numbers import Integral
from time import perf_counter


RESPONSE = "</Response>"
STANDBY = "</Standby>"
SILENCE = "</Silence>"
NATIVE_STATES = (RESPONSE, STANDBY, SILENCE)
DEFAULT_HISTORY_BUDGETS = (1024, 4096, 16384)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_frames(frames, name):
    previous = -math.inf
    for frame in frames:
        timestamp = float(frame.timestamp)
        if not math.isfinite(timestamp) or timestamp <= previous:
            raise ValueError(f"{name} timestamps must be finite, unique and strictly increasing")
        _positive_integer(frame.token_count, f"{name} frame token_count")
        previous = timestamp


def _temporal_order(length):
    """Visit endpoints, then progressively fill the temporal gaps."""
    result, seen = [], set()
    count = min(2, length)
    while count:
        indices = ([(length - 1) // 2] if count == 1 else
                   [round(i * (length - 1) / (count - 1)) for i in range(count)])
        for index in indices:
            if index not in seen:
                result.append(index)
                seen.add(index)
        if count == length:
            break
        count = min(count * 2, length)
    return result


def run_state_readout(adapter, candidates, recent, question, direct=False,
                      history_budgets=DEFAULT_HISTORY_BUDGETS, max_tokens=16,
                      context_tokens=None):
    """Return ``(history, metadata)``; the caller generates the final answer.

    ``probe_state`` returns a dict whose ``state`` is an exact native state
    literal, or None for any other model prediction.  Response stops reading;
    Standby and Silence both permit the next budget.  Invalid predictions stop
    at the current view and are recorded, rather than forced into a state.

    Budgets count historical visual tokens, not frames.  Selection preserves
    all previously exposed frames.  It tries whole bundles in a deterministic
    temporal-spread order, skipping bundles that exceed a visual budget.  If
    a proposed extension exceeds either prompt's context cap, exact token
    counts and binary search retain its largest fitting prefix.  This bounded
    prefix policy avoids quadratic prompt tokenization for large banks; it
    does not promise to maximize context utilization after such clipping.
    """
    started = perf_counter()
    recent = list(recent)
    candidates = sorted(candidates, key=lambda frame: frame.timestamp)
    if not recent:
        raise ValueError("State readout requires a nonempty recent window")
    _validate_frames(recent, "Recent")
    _validate_frames(candidates, "Historical")
    if candidates and candidates[-1].timestamp >= recent[0].timestamp:
        raise ValueError("Historical candidates must strictly precede the recent window")
    if not isinstance(question, str):
        raise ValueError("question must be a string")
    budgets = tuple(_positive_integer(value, "history budget") for value in history_budgets)
    if not budgets or any(right <= left for left, right in zip(budgets, budgets[1:])):
        raise ValueError("history_budgets must be a nonempty strictly increasing sequence")
    reserve = _positive_integer(max_tokens, "max_tokens")
    limit = _positive_integer(adapter.context_limit(context_tokens), "context limit")
    if reserve >= limit:
        raise ValueError("Answer reserve must be smaller than the context limit")

    count_calls = 0
    count_cache = {}

    def measure(indices):
        nonlocal count_calls
        key = tuple(sorted(indices))
        if key not in count_cache:
            frames = [candidates[index] for index in key]
            answer_count = _positive_integer(adapter.count_input_tokens(
                frames, recent, question, direct=direct), "answer input count")
            state_count = _positive_integer(adapter.count_state_input_tokens(
                frames, recent, question, direct=direct), "state input count")
            count_calls += 2
            count_cache[key] = (answer_count, state_count)
        return count_cache[key]

    def fits(counts):
        return counts[0] + reserve <= limit and counts[1] + 1 <= limit

    base_counts = measure(())
    if not fits(base_counts):
        raise ValueError(
            f"Recent context/question exceeds context limit {limit}: answer input "
            f"{base_counts[0]} + reserve {reserve}; state input {base_counts[1]} + reserve 1")

    order = _temporal_order(len(candidates))
    selected = set()
    used_tokens = 0
    counts = base_counts
    rounds = []
    selection_stages = []
    probe_seconds = 0.0
    selection_seconds = perf_counter() - started
    stop_reason = "history_budget_exhausted"
    context_clipped = False

    for budget in (0, *budgets):
        if budget:
            selection_start = perf_counter()
            extension = []
            planned_tokens = used_tokens
            for index in order:
                if index not in selected and planned_tokens + candidates[index].token_count <= budget:
                    extension.append(index)
                    planned_tokens += candidates[index].token_count
            clipped = False
            if extension:
                trial_counts = measure(selected.union(extension))
                if not fits(trial_counts):
                    clipped = context_clipped = True
                    # Native prompt counts increase with every added complete
                    # timestamp/image bundle.  Every accepted prefix is also
                    # checked exactly; generated state text is never included.
                    low, high = 0, len(extension)
                    while low < high:
                        middle = (low + high + 1) // 2
                        if fits(measure(selected.union(extension[:middle]))):
                            low = middle
                        else:
                            high = middle - 1
                    extension = extension[:low]
                if extension:
                    selected.update(extension)
                    used_tokens = sum(candidates[index].token_count for index in selected)
                    counts = measure(selected)
            selection_seconds += perf_counter() - selection_start
            selection_stages.append(dict(history_budget_tokens=budget,
                                         added_frames=len(extension), context_clipped=clipped))
            if not extension:
                # A later budget may admit a bundle skipped by this budget.
                # Never probe an unchanged visual view twice.
                continue

        history = [candidates[index] for index in sorted(selected)]
        probe_start = perf_counter()
        probe = adapter.probe_state(history, recent, question, direct=direct,
                                    context_tokens=context_tokens)
        elapsed = perf_counter() - probe_start
        probe_seconds += elapsed
        if not isinstance(probe, dict):
            raise TypeError("probe_state must return a metadata dictionary")
        state = probe.get("state")
        rounds.append(dict(probe, round_index=len(rounds), history_budget_tokens=budget,
                           history_timestamps=[float(frame.timestamp) for frame in history],
                           history_frame_tokens=[int(frame.token_count) for frame in history],
                           history_read_frames=len(history), history_visual_tokens=used_tokens,
                           answer_input_tokens=counts[0], state_input_tokens=counts[1],
                           state_probe_seconds=elapsed))
        if state == RESPONSE:
            stop_reason = "response"
            break
        if state not in (STANDBY, SILENCE):
            stop_reason = "invalid_state"
            break
        if not candidates:
            stop_reason = "no_history"
            break
        if len(selected) == len(candidates):
            stop_reason = "history_exhausted"
            break
    else:
        if context_clipped:
            stop_reason = "context_limit"

    history = [candidates[index] for index in sorted(selected)]
    return history, dict(
        read_policy="native_state_progressive_visual_history_v1",
        selection_policy="nested_temporal_spread_prefix",
        context_limit=limit, answer_reserve_tokens=reserve, state_reserve_tokens=1,
        history_budgets=list(budgets), state_rounds=rounds,
        state_probe_count=len(rounds), state_stop_reason=stop_reason,
        state_probe_seconds=probe_seconds, state_selection_seconds=selection_seconds,
        state_selection_stages=selection_stages,
        recent_only_input_tokens=base_counts[0], recent_only_state_input_tokens=base_counts[1],
        selected_input_tokens=counts[0], state_selected_input_tokens=counts[1],
        history_extra_input_tokens=counts[0] - base_counts[0],
        remaining_input_tokens=limit - reserve - counts[0],
        remaining_state_input_tokens=limit - 1 - counts[1],
        eligible_history_frames=len(candidates), budget_count_calls=count_calls,
        state_context_clipped=context_clipped,
    )

"""Training-free decisions about reading retained historical visual evidence.

The action probe is separate from answer generation.  It receives only the
question, the current visual view, and truthful bookkeeping about unread
records.  Neither task labels nor probe text are written into the memory bank.
"""

from __future__ import annotations

import math
from time import perf_counter

from .state_readout import (
    DEFAULT_HISTORY_BUDGETS,
    _positive_integer,
    _temporal_order,
    _validate_frames,
)


STOP = "STOP"
READ_MORE = "READ_MORE"


def _finite_time(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite timestamp")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite timestamp") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite timestamp")
    return value


def _history_status(candidates, recent, selected, query_time, stream_start):
    """Summarize observation records, never infer continuous visual coverage."""
    unread = [frame for index, frame in enumerate(candidates) if index not in selected]
    bins = []
    if unread:
        count = min(8, len(candidates))
        # Fixed contiguous groups of retained records avoid implying that the
        # bank covers the full elapsed interval (including dropped records).
        for group in range(count):
            left = group * len(candidates) // count
            right = (group + 1) * len(candidates) // count
            unread_count = sum(index not in selected for index in range(left, right))
            if unread_count:
                bins.append(dict(start=float(candidates[left].timestamp),
                                 end=float(candidates[right - 1].timestamp),
                                 unread_records=unread_count))
    return dict(
        query_time=query_time, stream_start=stream_start,
        earliest_retained_timestamp=float((candidates or recent)[0].timestamp),
        recent_start=float(recent[0].timestamp), recent_end=float(recent[-1].timestamp),
        retained_history_records=len(candidates), selected_history_records=len(selected),
        unread_history_records=len(unread), unread_time_bins=bins,
        history_observations="sparse_retained_records_not_complete_coverage",
        timestamp_origin_note=("Source timestamps may slightly precede the requested stream_start "
                               "because the existing sampler quantizes to source video frames."),
    )


def run_action_readout(adapter, candidates, recent, question, *, query_time,
                       stream_start, direct=False,
                       history_budgets=DEFAULT_HISTORY_BUDGETS, max_tokens=16,
                       context_tokens=None):
    """Return a nested historical view and a reproducible action trace.

    The adapter returns a metadata dict with ``action`` equal to STOP,
    READ_MORE, or None.  Invalid actions advance under a fixed fallback and
    remain visible in the trace.  The executor handles exhausted resources;
    reaching a bound never certifies that the visual evidence is sufficient.

    Each budget tries whole records in the original temporal-spread order.
    Context clipping scans candidate prefixes in descending length because
    the metadata prompt's token count need not increase monotonically.  Both
    prompts must fit where another action is required.  Terminal views need
    only fit the final answer, and do not run an unused action probe.
    """
    started = perf_counter()
    recent = list(recent)
    candidates = sorted(candidates, key=lambda frame: frame.timestamp)
    if not recent:
        raise ValueError("Action readout requires a nonempty recent window")
    _validate_frames(recent, "Recent")
    _validate_frames(candidates, "Historical")
    if candidates and candidates[-1].timestamp >= recent[0].timestamp:
        raise ValueError("Historical candidates must strictly precede the recent window")
    query_time = _finite_time(query_time, "query_time")
    stream_start = _finite_time(stream_start, "stream_start")
    if stream_start > query_time:
        raise ValueError("stream_start must not exceed query_time")
    if recent[-1].timestamp >= query_time:
        raise ValueError("Visual evidence must strictly precede query_time")
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
            history = [candidates[index] for index in key]
            status = _history_status(candidates, recent, set(key), query_time, stream_start)
            answer_count = _positive_integer(adapter.count_input_tokens(
                history, recent, question, direct=direct), "answer input count")
            action_count = _positive_integer(adapter.count_read_action_input_tokens(
                history, recent, question, status, direct=direct), "action input count")
            count_calls += 2
            count_cache[key] = (answer_count, action_count, status)
        return count_cache[key]

    def boundary(indices, budget):
        if not candidates:
            return "no_history"
        if len(indices) == len(candidates):
            return "history_exhausted"
        used = sum(candidates[index].token_count for index in indices)
        if budget == budgets[-1] or not any(
                index not in indices and used + frame.token_count <= budgets[-1]
                for index, frame in enumerate(candidates)):
            return "history_budget_exhausted"
        return None

    def fits(indices, budget, counts):
        return (counts[0] + reserve <= limit and
                (boundary(indices, budget) is not None or counts[1] + 1 <= limit))

    selected = set()
    used_tokens = 0
    base_counts = counts = measure(selected)
    if not fits(selected, 0, counts):
        raise ValueError(
            f"Recent context/question exceeds context limit {limit}: answer input "
            f"{counts[0]} + reserve {reserve}; action input {counts[1]} + reserve 1")

    order = _temporal_order(len(candidates))
    rounds, selection_stages = [], []
    probe_seconds = 0.0
    selection_seconds = perf_counter() - started
    invalid_count = 0
    context_clipped = False
    stop_reason = None

    for budget in (0, *budgets):
        clipped = False
        if budget:
            selection_start = perf_counter()
            extension = []
            planned_tokens = used_tokens
            for index in order:
                if index not in selected and planned_tokens + candidates[index].token_count <= budget:
                    extension.append(index)
                    planned_tokens += candidates[index].token_count
            accepted = []
            if extension:
                # Metadata shrinks and tokenizes differently as records become
                # selected.  Never assume monotonicity for a binary search.
                for length in range(len(extension), 0, -1):
                    trial = selected.union(extension[:length])
                    trial_counts = measure(trial)
                    if fits(trial, budget, trial_counts):
                        accepted = extension[:length]
                        selected = trial
                        counts = trial_counts
                        break
                    clipped = context_clipped = True
                used_tokens = sum(candidates[index].token_count for index in selected)
            selection_seconds += perf_counter() - selection_start
            selection_stages.append(dict(
                history_budget_tokens=budget, proposed_frames=len(extension),
                added_frames=len(accepted), context_clipped=clipped,
                answer_input_tokens=counts[0], action_input_tokens=counts[1],
            ))
            if not accepted:
                # Larger budgets may admit a previously skipped record.  At
                # the final budget return the unchanged answerable view.
                if budget == budgets[-1]:
                    stop_reason = "context_limit" if clipped else "history_budget_exhausted"
                    break
                continue

        stop_reason = boundary(selected, budget)
        if stop_reason is not None:
            if clipped and stop_reason == "history_budget_exhausted":
                stop_reason = "context_limit"
            break

        history = [candidates[index] for index in sorted(selected)]
        probe_start = perf_counter()
        probe = adapter.probe_read_action(
            history, recent, question, counts[2], direct=direct,
            context_tokens=context_tokens)
        elapsed = perf_counter() - probe_start
        probe_seconds += elapsed
        if not isinstance(probe, dict):
            raise TypeError("probe_read_action must return a metadata dictionary")
        action = probe.get("action")
        invalid = action not in (STOP, READ_MORE)
        invalid_count += int(invalid)
        rounds.append(dict(
            probe, round_index=len(rounds), history_budget_tokens=budget,
            history_timestamps=[float(frame.timestamp) for frame in history],
            history_frame_tokens=[int(frame.token_count) for frame in history],
            history_read_frames=len(history), history_visual_tokens=used_tokens,
            history_status=counts[2], answer_input_tokens=counts[0],
            action_input_tokens=counts[1], action_probe_seconds=elapsed,
            invalid_action=invalid, effective_action=READ_MORE if invalid else action,
        ))
        if action == STOP:
            stop_reason = "model_stop"
            break

    history = [candidates[index] for index in sorted(selected)]
    return history, dict(
        read_policy="explicit_action_progressive_visual_history_v1",
        selection_policy="nested_temporal_spread_prefix",
        context_limit=limit, answer_reserve_tokens=reserve, action_reserve_tokens=1,
        history_budgets=list(budgets), action_rounds=rounds,
        action_probe_count=len(rounds), action_stop_reason=stop_reason,
        action_invalid_count=invalid_count, action_probe_seconds=probe_seconds,
        action_selection_seconds=selection_seconds, action_selection_stages=selection_stages,
        recent_only_input_tokens=base_counts[0], recent_only_action_input_tokens=base_counts[1],
        selected_input_tokens=counts[0], action_selected_input_tokens=counts[1],
        history_extra_input_tokens=counts[0] - base_counts[0],
        remaining_input_tokens=limit - reserve - counts[0],
        remaining_action_input_tokens=limit - 1 - counts[1],
        eligible_history_frames=len(candidates), budget_count_calls=count_calls,
        action_context_clipped=context_clipped, final_history_status=counts[2],
    )

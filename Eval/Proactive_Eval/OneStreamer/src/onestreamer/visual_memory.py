r"""Caption-free visual memory prototype for the released OneStreamer/Qwen3-VL.

This is an independent entry point; it does not change the official eval queue.
Example (from the repository root, using the official inference environment)::

    python Eval/Proactive_Eval/OneStreamer/src/onestreamer/visual_memory.py \
        --manifest /path/to/official/manifest.json \
        --model-path /path/to/OneStreamer-4B --video-root /path/to/OVOBench \
        --output /path/to/new-visual-run --limit 20 --dry-run

Remove --dry-run to infer. --samples accepts the official sample JSON/JSONL
format instead of a manifest. No caption writer is imported or executed.

--history-compression avg_pool encodes each 1-fps stream observation once,
keeps the newest N full feature bundles (--recent-frames, default 16), and spatially pools evicted features
(including all DeepStack branches) into a bounded CPU history bank.
progressive_pool additionally pools each archived frame every configured age
interval, down to one token, and drops the oldest records when caps are exceeded.
vcast is an opt-in, training-free alternative: preserve original tokens until
budget pressure, then allocate by temporal curvature and select spatial tokens
with V-CAST's dual-anchor score. Original coordinates and all DeepStack branches
remain aligned. See VCAST_MEMORY.md for the streaming adaptation and limits.
The default
resize mode instead encodes lower-resolution historical images and replays the
official recent-frame decoder. resize/avg_pool use a seeded reservoir; readout
uses fixed-budget temporal selection, or --history-read-mode fill to use the
remaining context after the configured recent window, question/formatting and answer reservation.
--history-read-mode state first probes the recent window using native readiness
tokens, then exposes nested visual history only while the model reports
insufficient evidence. --state-history-budgets sets the increasing historical
visual-token caps. State decisions never enter the final answer or memory bank.
--history-read-mode action instead asks whether to STOP or READ_MORE using
ordinary A/B tokens and truthful unread-history metadata. It uses the same
budget schedule; the program enforces exhaustion and records invalid actions.
--answer-prompt frame_id_witness runs an exploratory two-line answer and
frame-citation prompt with --history-read-mode fill. It retains the raw answer
and structurally audits cited frame IDs; citations are not proof of support.
--context-tokens sets a total cap (default 32768), bounded by the model config.
The historical writer and temporal frame ordering are model independent;
the optional state readout policy depends on the frozen checkpoint's decisions.
the feature adapter is deliberately limited to Qwen3-VL/Transformers 4.57.6.

Frames are observed causally from the evaluated stream start; discarded evidence
is never reconstructed. Pooling mode's recent window is a fixed-origin streaming FIFO,
which has a different sampling phase from the official end-aligned recent frames.
Use the same pooling mode with --history-capacity 0 as its recent-only control.
Source pixels are preserved when encoding recent frames in both modes.
All ingestion, recent decoding/encoding, readout and QA time is reported.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, deque
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any


VERSION = "visual_memory_v7_context_budget_pool"
DEFAULT_CONTEXT_TOKENS = 32768
POOLED_COMPRESSIONS = frozenset({"avg_pool", "progressive_pool", "budget_pool"})
FEATURE_STREAM_COMPRESSIONS = POOLED_COMPRESSIONS | {"vcast"}
VCAST_VERSION = "visual_memory_v10_vcast_streaming"
# Safety margin for the per-stream history budget; the fill reader still
# measures the exact prompt and drops records if this estimate is exceeded.
HISTORY_BUDGET_MARGIN_TOKENS = 64
STATE_TOKENS = ("</Response>", "</Standby>", "</Silence>")
STATE_PROMPT_VERSION = "native_response_prefix_v3"
STATE_SYSTEM_PROMPT = (
    "You are a helpful assistant specializing in streaming video question answering. "
    "Answer the user's question using only the supplied timestamped visual evidence. "
    "HISTORICAL_VISUAL_MEMORY contains sparse earlier observations; RECENT_VISUAL_CONTEXT contains "
    "the recent scene. Respect timestamps and distinguish earlier states from the current state. "
    "Use the native response protocol at each update. When the available information is sufficient "
    "to answer the question, begin with </Response> followed by the answer in the requested format. "
    "Use </Standby> alone when relevant evidence is present but the question cannot yet be answered. "
    "Use </Silence> alone when the observations provide no relevant evidence for the question. "
    "Whenever you deliver an answer, begin with </Response>. The requested letter, Yes/No, or number "
    "format applies to the answer after that prefix. A question about the current scene can be "
    "answered from recent observations alone; a new event boundary is not required. "
    "For a prediction question, infer from observed evidence without requiring future frames. "
    "A supported negative answer or zero count is also an answer. Missing historical frames alone "
    "do not establish that an event never happened."
)
READ_ACTION_PROMPT_VERSION = "history_evidence_scope_v1"
READ_ACTION_LABELS = {"A": "STOP", "B": "READ_MORE"}
READ_ACTION_SYSTEM_PROMPT = """You control access to stored visual history for a streaming-video question.
Do not answer the video question. Decide whether to answer using the current
evidence or request more already observed historical evidence before answering.
Output exactly one ordinary letter:
A = STOP: proceed to answer using the current evidence.
B = READ_MORE: request the next larger historical view.

You receive the video question, timestamped visual evidence, and HISTORY_STATUS
generated by the program. All evidence is from before query_time. recent_end is
the last sampled recent observation, not the query cutoff. Time bins contain
counts of unread stored observations; they do not reveal the observations'
contents or certify that every event in those intervals was captured.

Judge the temporal evidence required by the question:
- Visible text, a local action, or the number of objects currently in view can
  be supported by recent evidence when the requested detail is visible.
- Counting current objects differs from counting past events. For a total,
  frequency, first/last event, or a claim that something never happened over an
  interval, visible examples alone do not establish the complete answer. Request
  more history when unread stored observations remain in the required interval.
- An earlier event, before/after relation, change, identity, or progress question
  may need preceding observations. An earlier state does not automatically
  remain true now. Use the time scope asked by the question.
- Predictions use observed development. Request earlier evidence when needed;
  do not request future frames or wait for the predicted event to happen.

Being able to produce a plausible answer, number, or negative answer is not a
reason to stop. Use the question's time scope, visible evidence, and unread
record distribution. Selected first and last timestamps do not mean every
stored observation between them was read. Do not invent unread contents.

READ_MORE requests an extension; the program enforces visual and context
budgets and terminates when no further view fits. Unread records are not a
promise that all can be loaded. Exhaustion does not certify a correct answer.
Answer-format instructions inside the video question apply to the later final
answer only. At this step output A or B, with no prefix or explanation.
"""


@dataclass(frozen=True)
class MemoryConfig:
    capacity: int = 64
    store_tokens: int = 16384
    store_bytes: int = 512 * 1024**2
    read_frames: int = 4
    read_tokens: int = 1024
    history_fps: float = 1.0
    history_max_pixels: int = 262144
    seed: int = 0
    read_mode: str = "fixed"
    context_tokens: int | None = DEFAULT_CONTEXT_TOKENS
    history_compression: str = "resize"
    pool_stride: int = 2
    pool_age_seconds: float = 60.0
    recent_frames: int = 16
    state_history_budgets: tuple[int, ...] = (1024, 4096, 16384)
    answer_prompt: str = "standard"
    store_budget: str = "fixed"
    min_frame_tokens: int = 16
    stream_include_end: bool = False
    route_threshold: float = 0.0
    route_escalation: str = "SHORT"
    short_history_tokens: int = 16384
    vcast_temperature: float = 0.7

    def __post_init__(self):
        for name in ("capacity", "store_tokens", "store_bytes", "read_frames", "read_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not math.isfinite(self.history_fps) or self.history_fps <= 0:
            raise ValueError("history_fps must be finite and positive")
        if self.history_max_pixels < 1024:
            raise ValueError("history_max_pixels must be at least 1024")
        if self.read_mode not in {"fixed", "fill", "state", "action", "routed"}:
            raise ValueError("read_mode must be fixed, fill, state, action or routed")
        if not (isinstance(self.route_threshold, (int, float)) and 0.0 <= self.route_threshold <= 1.0):
            raise ValueError("route_threshold must be a probability")
        if self.route_escalation not in {"SHORT", "LONG"}:
            raise ValueError("route_escalation must be SHORT or LONG")
        if (not isinstance(self.short_history_tokens, int) or isinstance(self.short_history_tokens, bool)
                or self.short_history_tokens <= 0):
            raise ValueError("short_history_tokens must be a positive integer")
        if self.answer_prompt not in {"standard", "frame_id_witness"}:
            raise ValueError("answer_prompt must be standard or frame_id_witness")
        if self.answer_prompt == "frame_id_witness" and self.read_mode != "fill":
            raise ValueError("frame_id_witness requires read_mode=fill")
        budgets = self.state_history_budgets
        if (not budgets or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in budgets)
                or any(a >= b for a, b in zip(budgets, budgets[1:]))):
            raise ValueError("state_history_budgets must be strictly increasing positive integers")
        if self.context_tokens is not None and (
                not isinstance(self.context_tokens, int) or isinstance(self.context_tokens, bool)
                or self.context_tokens <= 0):
            raise ValueError("context_tokens must be a positive integer")
        if self.history_compression not in FEATURE_STREAM_COMPRESSIONS | {"resize"}:
            raise ValueError("Unknown history_compression mode")
        if (not isinstance(self.vcast_temperature, (int, float))
                or isinstance(self.vcast_temperature, bool)
                or not math.isfinite(self.vcast_temperature) or self.vcast_temperature <= 0):
            raise ValueError("vcast_temperature must be finite and positive")
        if not isinstance(self.pool_stride, int) or isinstance(self.pool_stride, bool) or self.pool_stride < 1:
            raise ValueError("pool_stride must be a positive integer")
        if self.history_compression in FEATURE_STREAM_COMPRESSIONS and self.history_fps != 1.0:
            raise ValueError("Feature compression uses a shared 1-fps stream for recent frames and history")
        if not math.isfinite(self.pool_age_seconds) or self.pool_age_seconds <= 0:
            raise ValueError("pool_age_seconds must be finite and positive")
        if self.history_compression in {"progressive_pool", "budget_pool"} and self.pool_stride < 2:
            raise ValueError("Progressive and budget pooling require pool_stride >= 2")
        if self.store_budget not in {"fixed", "context"}:
            raise ValueError("store_budget must be fixed or context")
        if self.store_budget == "context" and (
                self.history_compression not in {"budget_pool", "vcast"} or self.context_tokens is None):
            raise ValueError("store_budget=context requires budget_pool or vcast and an explicit context_tokens cap")
        if (not isinstance(self.min_frame_tokens, int) or isinstance(self.min_frame_tokens, bool)
                or self.min_frame_tokens < 1):
            raise ValueError("min_frame_tokens must be a positive integer")
        if not isinstance(self.stream_include_end, bool):
            raise ValueError("stream_include_end must be a boolean")
        if (not isinstance(self.recent_frames, int) or isinstance(self.recent_frames, bool)
                or self.recent_frames <= 0):
            raise ValueError("recent_frames must be a positive integer")

    @property
    def enabled(self):
        return all((self.capacity, self.store_tokens, self.store_bytes)) and (
            self.read_mode in {"fill", "state", "action", "routed"} or all((self.read_frames, self.read_tokens)))


@dataclass(frozen=True)
class VisualFrame:
    timestamp: float
    grid_thw: tuple[int, int, int]
    features: Any
    deepstack: tuple[Any, ...]
    size: tuple[int, int]
    source_size: tuple[int, int] | None = None
    pool_source_shape: tuple[int, int] | None = None
    pool_scale: int = 1
    # Original merged-grid row-major coordinates. None denotes a dense grid.
    source_indices: tuple[int, ...] | None = None

    @property
    def token_count(self):
        return int(self.features.shape[0])

    @property
    def nbytes(self):
        """Feature/coordinate payload, including DeepStack; not allocator RSS."""
        return (sum(x.numel() * x.element_size() for x in (self.features, *self.deepstack))
                + (8 * len(self.source_indices) if self.source_indices is not None else 0))


def pool_visual_frame(frame: VisualFrame, stride: int = 2, merge_size: int = 2) -> VisualFrame:
    """Pool a merged spatial grid, with every DeepStack branch and area weights.

    Qwen3-VL 4.57.6 outputs merged image tokens in row-major spatial order.
    The result is a *synthetic coarse grid*, not the native encoding of a
    smaller image. Original cell areas weight repeated pooling, so odd boundary
    cells are not overrepresented. It owns independent CPU tensors.
    """
    import torch
    import torch.nn.functional as F

    if frame.source_indices is not None:
        raise ValueError("Sparse selected tokens cannot be spatially pooled")
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in (stride, merge_size)):
        raise ValueError("Pooling stride and merge size must be positive integers")
    if len(frame.grid_thw) != 3 or frame.grid_thw[0] != 1:
        raise ValueError("Feature pooling supports image grids with t=1 only")
    _, grid_h, grid_w = frame.grid_thw
    if min(grid_h, grid_w) <= 0 or grid_h % merge_size or grid_w % merge_size:
        raise ValueError("Grid must be positive and divisible by merge size")
    height, width = grid_h // merge_size, grid_w // merge_size
    if frame.features.ndim != 2 or frame.features.shape[0] != height * width:
        raise ValueError("Feature shape does not match merged spatial grid")
    if (frame.size[0] % grid_w or frame.size[1] % grid_h
            or frame.size[0] // grid_w != frame.size[1] // grid_h
            or min(frame.size) <= 0):
        raise ValueError("Carrier size does not match a square-patch grid")
    patch_size = frame.size[0] // grid_w
    source_shape = frame.pool_source_shape or (height, width)
    if (not isinstance(frame.pool_scale, int) or isinstance(frame.pool_scale, bool)
            or frame.pool_scale < 1 or len(source_shape) != 2
            or any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in source_shape)
            or (frame.pool_source_shape is None and frame.pool_scale != 1)
            or (math.ceil(source_shape[0] / frame.pool_scale),
                math.ceil(source_shape[1] / frame.pool_scale)) != (height, width)):
        raise ValueError("Original pooling geometry does not match the current grid")
    identity = stride == 1 or height * width == 1
    target_h, target_w = (height, width) if identity else (math.ceil(height / stride), math.ceil(width / stride))
    weights = denominator = None
    if not identity:
        row_mass = torch.tensor([min(frame.pool_scale, source_shape[0] - i * frame.pool_scale)
                                 for i in range(height)], device=frame.features.device, dtype=torch.float32)
        col_mass = torch.tensor([min(frame.pool_scale, source_shape[1] - i * frame.pool_scale)
                                 for i in range(width)], device=frame.features.device, dtype=torch.float32)
        weights = (row_mass[:, None] * col_mass[None, :])[None, None]
        denominator = F.avg_pool2d(weights, kernel_size=stride, stride=stride,
                                   ceil_mode=True, count_include_pad=False)

    def pool(tensor):
        if tuple(tensor.shape) != tuple(frame.features.shape) or not tensor.is_floating_point():
            raise ValueError("All feature branches must share a floating-point spatial shape")
        if identity:
            return tensor.detach().to("cpu", copy=True)
        with torch.inference_mode():
            spatial = tensor.detach().reshape(height, width, -1).permute(2, 0, 1).unsqueeze(0)
            pooled = F.avg_pool2d(spatial.float() * weights.to(tensor.device), kernel_size=stride,
                                  stride=stride, ceil_mode=True, count_include_pad=False)
            pooled = pooled / denominator.to(tensor.device)
            flat = pooled.squeeze(0).permute(1, 2, 0).reshape(target_h * target_w, -1)
            return flat.to(device="cpu", dtype=tensor.dtype, copy=True)

    return VisualFrame(frame.timestamp, (1, target_h * merge_size, target_w * merge_size),
                       pool(frame.features), tuple(pool(x) for x in frame.deepstack),
                       (target_w * merge_size * patch_size, target_h * merge_size * patch_size),
                       source_size=frame.source_size, pool_source_shape=source_shape,
                       pool_scale=frame.pool_scale if identity else frame.pool_scale * stride)


class RecentFeatureWindow:
    """Keep full features until an incoming frame evicts the oldest observation."""

    def __init__(self, capacity=16):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("Recent capacity must be a positive integer")
        self.capacity = capacity
        self._frames = deque()
        self._last_timestamp = -math.inf

    def observe(self, frame: VisualFrame):
        if not math.isfinite(frame.timestamp) or frame.timestamp <= self._last_timestamp:
            raise ValueError("Recent timestamps must be finite and strictly increasing")
        self._last_timestamp = frame.timestamp
        self._frames.append(frame)
        return self._frames.popleft() if len(self._frames) > self.capacity else None

    def frames(self):
        return list(self._frames)


def uniform_indices(length: int, count: int) -> list[int]:
    count = min(length, count)
    if count <= 0:
        return []
    if count == 1:
        return [(length - 1) // 2]
    return [round(i * (length - 1) / (count - 1)) for i in range(count)]


class VisualMemoryBank:
    """Query-blind random-priority reservoir under frame/token/payload caps.

    Equal-size records give uniform reservoir retention. Variable-size records
    additionally undergo budget eviction, so uniform inclusion is not claimed.
    The resize writer may store recent samples; readout always excludes them.
    The feature-pooling writer admits frames only after recent-FIFO eviction.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._rng = random.Random(config.seed)
        self._items = []
        self._last_timestamp = -math.inf
        self._seen = 0
        self._oversize = 0

    def observe(self, frame: VisualFrame):
        if not math.isfinite(frame.timestamp) or frame.timestamp <= self._last_timestamp:
            raise ValueError("Observed timestamps must be finite and strictly increasing")
        if frame.token_count <= 0:
            raise ValueError("A visual frame must contain tokens")
        self._last_timestamp = frame.timestamp
        self._seen += 1
        cfg = self.config
        if frame.token_count > cfg.store_tokens or frame.nbytes > cfg.store_bytes:
            self._oversize += 1
            return
        if not cfg.capacity:
            return
        self._items.append((self._rng.random(), self._seen, frame))
        self._items.sort(key=lambda entry: (entry[0], entry[1]))
        while (len(self._items) > cfg.capacity
               or sum(x[2].token_count for x in self._items) > cfg.store_tokens
               or sum(x[2].nbytes for x in self._items) > cfg.store_bytes):
            self._items.pop(0)

    def candidates(self, before: float) -> list[VisualFrame]:
        if not math.isfinite(before):
            raise ValueError("History cutoff must be finite")
        return sorted((x[2] for x in self._items if x[2].timestamp < before),
                      key=lambda frame: frame.timestamp)

    def read(self, before: float) -> list[VisualFrame]:
        cfg = self.config
        eligible = [f for f in self.candidates(before) if f.token_count <= cfg.read_tokens]
        selected = []
        tokens = 0
        preferred = uniform_indices(len(eligible), cfg.read_frames)
        order = preferred + [i for i in range(len(eligible)) if i not in preferred]
        for i in order:
            if len(selected) >= cfg.read_frames:
                break
            frame = eligible[i]
            if tokens + frame.token_count <= cfg.read_tokens:
                selected.append(frame)
                tokens += frame.token_count
        return sorted(selected, key=lambda frame: frame.timestamp)

    def stats(self):
        return dict(seen=self._seen, stored_frames=len(self._items),
                    stored_tokens=sum(x[2].token_count for x in self._items),
                    stored_bytes=sum(x[2].nbytes for x in self._items),
                    skipped_oversize=self._oversize)


@dataclass
class _ProgressiveRecord:
    frame: VisualFrame
    archived_at: float
    stage: int = 1


class ProgressiveVisualMemoryBank:
    """Age cached frames through progressively coarser grids; drop oldest at caps.

    A small heap schedules only the next compression of each nonsaturated frame.
    Frames at one token require no further work. Dropped records never return.
    """

    def __init__(self, config: MemoryConfig, merge_size: int = 2):
        if config.pool_stride < 2:
            raise ValueError("Progressive pooling requires pool_stride >= 2")
        self.config, self.merge_size = config, merge_size
        self._records = OrderedDict()
        self._due = []
        self._clock = self._last_timestamp = -math.inf
        self._seen = self._tokens = self._bytes = 0
        self._dropped = self._oversize = self._operations = 0
        self._pool_input_tokens = self._pool_output_tokens = 0

    def _pool(self, frame):
        pooled = pool_visual_frame(frame, self.config.pool_stride, self.merge_size)
        self._operations += int(pooled.token_count < frame.token_count)
        self._pool_input_tokens += frame.token_count
        self._pool_output_tokens += pooled.token_count
        return pooled

    def _compact_heap(self):
        if len(self._due) > 2 * len(self._records) + 64:
            self._due = [(due, index) for due, index in self._due if index in self._records]
            heapq.heapify(self._due)

    def advance(self, now: float):
        if not math.isfinite(now) or now < self._clock:
            raise ValueError("Cannot move progressive memory time backwards")
        self._clock = now
        while self._due and self._due[0][0] <= now:
            due, index = heapq.heappop(self._due)
            record = self._records.get(index)
            if record is None:
                continue
            old = record.frame
            pooled = self._pool(old)
            if pooled.token_count >= old.token_count:
                raise RuntimeError("Scheduled progressive pooling must reduce tokens")
            self._tokens += pooled.token_count - old.token_count
            self._bytes += pooled.nbytes - old.nbytes
            record.frame = pooled
            record.stage += 1
            if pooled.token_count > 1:
                next_due = due + self.config.pool_age_seconds
                if not math.isfinite(next_due) or next_due <= due:
                    raise ValueError("Pooling interval cannot advance this timestamp")
                heapq.heappush(self._due, (next_due, index))
        self._compact_heap()

    def observe(self, frame: VisualFrame, now: float):
        if (not math.isfinite(frame.timestamp) or frame.timestamp <= self._last_timestamp
                or frame.timestamp > now):
            raise ValueError("Archived timestamps must increase and not exceed current time")
        # Age existing evidence first; it may free enough budget to keep it.
        self.advance(now)
        pooled = self._pool(frame)
        self._last_timestamp = frame.timestamp
        self._seen += 1
        cfg = self.config
        if pooled.token_count > cfg.store_tokens or pooled.nbytes > cfg.store_bytes:
            self._oversize += 1
            return
        if not cfg.capacity:
            return
        index = self._seen
        self._records[index] = _ProgressiveRecord(pooled, now)
        self._tokens += pooled.token_count
        self._bytes += pooled.nbytes
        if pooled.token_count > 1:
            next_due = now + cfg.pool_age_seconds
            if not math.isfinite(next_due) or next_due <= now:
                raise ValueError("Pooling interval cannot advance this timestamp")
            heapq.heappush(self._due, (next_due, index))
        while (len(self._records) > cfg.capacity or self._tokens > cfg.store_tokens
               or self._bytes > cfg.store_bytes):
            _, oldest = self._records.popitem(last=False)
            dropped = oldest.frame
            self._tokens -= dropped.token_count
            self._bytes -= dropped.nbytes
            self._dropped += 1
        self._compact_heap()

    def candidates(self, before: float):
        if not math.isfinite(before):
            raise ValueError("History cutoff must be finite")
        return [entry.frame for entry in self._records.values() if entry.frame.timestamp < before]

    # Same fixed-budget reader; the storage policy is entirely different.
    read = VisualMemoryBank.read

    def stats(self):
        return dict(seen=self._seen, stored_frames=len(self._records), stored_tokens=self._tokens,
                    stored_bytes=self._bytes, skipped_oversize=self._oversize,
                    dropped_oldest=self._dropped, pool_operations=self._operations,
                    pool_input_tokens=self._pool_input_tokens, pool_output_tokens=self._pool_output_tokens,
                    pending_pool_events=len(self._due))


class BudgetVisualMemoryBank:
    """Keep every retired observation; coarsen under budget pressure before dropping.

    A record costs its visual tokens plus a fixed per-image prompt overhead.
    While over budget, the oldest record among those with the most tokens is
    pooled one more stride, so resolution never increases with age. Records
    stop at ``min_frame_tokens`` (one further pooling may land below it). Only
    when every record is at that floor is a record dropped: the interior one
    whose removal leaves the smallest temporal gap, i.e. uniform thinning that
    keeps the earliest observation. The token budget may be set per stream.
    """

    def __init__(self, config: MemoryConfig, merge_size: int = 2):
        if config.pool_stride < 2:
            raise ValueError("Budget pooling requires pool_stride >= 2")
        self.config, self.merge_size = config, merge_size
        self.token_budget = config.store_tokens
        self.frame_overhead = 0
        self._records = []
        self._last_timestamp = -math.inf
        self._seen = self._tokens = self._bytes = 0
        self._dropped = self._oversize = self._operations = 0
        self._pool_input_tokens = self._pool_output_tokens = 0

    def set_budget(self, tokens: int, frame_overhead: int = 0):
        for name, value in (("tokens", tokens), ("frame_overhead", frame_overhead)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"History budget {name} must be a nonnegative integer")
        self.token_budget, self.frame_overhead = tokens, frame_overhead
        self._enforce()

    def _pool(self, frame):
        pooled = pool_visual_frame(frame, self.config.pool_stride, self.merge_size)
        self._operations += int(pooled.token_count < frame.token_count)
        self._pool_input_tokens += frame.token_count
        self._pool_output_tokens += pooled.token_count
        return pooled

    def _cost(self):
        return self._tokens + self.frame_overhead * len(self._records)

    def _over(self):
        cfg = self.config
        return (len(self._records) > cfg.capacity or self._cost() > self.token_budget
                or self._bytes > cfg.store_bytes)

    def _remove(self, index):
        dropped = self._records.pop(index)
        self._tokens -= dropped.token_count
        self._bytes -= dropped.nbytes
        self._dropped += 1

    def _enforce(self):
        floor = self.config.min_frame_tokens
        while self._records and self._over():
            target = None
            for index, frame in enumerate(self._records):
                # Strict comparison keeps the oldest record among equal sizes.
                if frame.token_count > max(floor, 1) and (
                        target is None or frame.token_count > self._records[target].token_count):
                    target = index
            if target is not None:
                old = self._records[target]
                pooled = self._pool(old)
                if pooled.token_count >= old.token_count:
                    raise RuntimeError("Budget pooling must reduce tokens")
                self._tokens += pooled.token_count - old.token_count
                self._bytes += pooled.nbytes - old.nbytes
                self._records[target] = pooled
                continue
            times = [frame.timestamp for frame in self._records]
            if len(times) <= 2:
                self._remove(0)
                continue
            self._remove(min(range(1, len(times) - 1), key=lambda i: (times[i + 1] - times[i - 1], i)))

    def observe(self, frame: VisualFrame, now: float | None = None):
        if not math.isfinite(frame.timestamp) or frame.timestamp <= self._last_timestamp:
            raise ValueError("Archived timestamps must be finite and strictly increasing")
        self._last_timestamp = frame.timestamp
        self._seen += 1
        cfg = self.config
        if not cfg.capacity:
            return
        pooled = self._pool(frame)
        if pooled.token_count + self.frame_overhead > self.token_budget or pooled.nbytes > cfg.store_bytes:
            self._oversize += 1
            return
        self._records.append(pooled)
        self._tokens += pooled.token_count
        self._bytes += pooled.nbytes
        self._enforce()

    def candidates(self, before: float):
        if not math.isfinite(before):
            raise ValueError("History cutoff must be finite")
        return [frame for frame in self._records if frame.timestamp < before]

    read = VisualMemoryBank.read

    def stats(self):
        tokens = [frame.token_count for frame in self._records]
        return dict(seen=self._seen, stored_frames=len(self._records), stored_tokens=self._tokens,
                    stored_bytes=self._bytes, skipped_oversize=self._oversize,
                    dropped_frames=self._dropped, pool_operations=self._operations,
                    pool_input_tokens=self._pool_input_tokens, pool_output_tokens=self._pool_output_tokens,
                    token_budget=self.token_budget, frame_overhead=self.frame_overhead,
                    budget_cost=self._cost(),
                    earliest_timestamp=self._records[0].timestamp if self._records else None,
                    frame_tokens_min=min(tokens, default=0), frame_tokens_max=max(tokens, default=0))


def context_history_budget(adapter, native_tokens, prompts, recent_frames, answer_reserve,
                           context_tokens):
    """Visual-token budget left for history after recent frames and the longest prompt.

    Uses one-token placeholder frames with late timestamps, so timestamp text is
    not underestimated. Returns (budget, per-image text overhead).
    """
    import torch

    def dummy(timestamp):
        return VisualFrame(float(timestamp), (1, 2, 2), torch.empty(1, 1), (), (28, 28))

    if not prompts:
        raise ValueError("A stream needs at least one question")
    one, two = [dummy(99990)], [dummy(99990), dummy(99991)]
    overhead = 0
    fixed = 0
    for question, kwargs in prompts:
        single = adapter.count_input_tokens([], one, question, **kwargs)
        double = adapter.count_input_tokens([], two, question, **kwargs)
        # Frame-ID history IDs can be longer than recent IDs; pad per image.
        overhead = max(overhead, double - single - 1 + 2)
        fixed = max(fixed, single - 1)
    limit = adapter.context_limit(context_tokens)
    recent = recent_frames * (native_tokens + overhead)
    budget = limit - answer_reserve - fixed - recent - HISTORY_BUDGET_MARGIN_TOKENS
    return max(0, budget), overhead


def _engine_lock(engine):
    if not hasattr(engine, "_visual_memory_lock"):
        engine._visual_memory_lock = threading.RLock()
    return engine._visual_memory_lock


@contextmanager
def cached_image_features(engine, frames: list[VisualFrame]):
    """Temporarily replace only this engine's native image-feature function.

    Dense records use the native path unchanged. Sparse records retain the
    original grid and coordinates through a scoped mRoPE adapter. No KV from
    another context is spliced into the decoder.
    """
    import torch

    core = engine.model.model
    config = engine.model.config
    merge = config.vision_config.spatial_merge_size
    layers = len(config.vision_config.deepstack_visual_indexes)
    hidden = config.text_config.hidden_size
    for frame in frames:
        if (len(frame.grid_thw) != 3 or frame.grid_thw[0] != 1
                or any(x <= 0 for x in frame.grid_thw)):
            raise ValueError("Only complete image grids (t=1) are supported")
        if frame.grid_thw[1] % merge or frame.grid_thw[2] % merge:
            raise ValueError("Image grid is not divisible by spatial_merge_size")
        expected = math.prod(frame.grid_thw) // merge**2
        if frame.source_indices is not None:
            from onestreamer.sparse_visual_inputs import validate_source_indices
            validate_source_indices(frame, merge)
            expected = len(frame.source_indices)
        if tuple(frame.features.shape) != (expected, hidden):
            raise ValueError("Cached final features do not match the image grid/hidden size")
        if len(frame.deepstack) != layers:
            raise ValueError("Cached DeepStack layer count does not match this model")
        if any(tuple(x.shape) != (expected, hidden) for x in frame.deepstack):
            raise ValueError("Cached DeepStack shape does not match the final features")
    expected_grid = torch.tensor([f.grid_thw for f in frames], dtype=torch.long)
    state = {"calls": 0}

    def get_cached(pixel_values, image_grid_thw=None):
        if image_grid_thw is None or not torch.equal(image_grid_thw.detach().cpu(), expected_grid):
            raise ValueError("Readout image grids/order differ from the cached evidence")
        state["calls"] += 1
        if state["calls"] != 1:
            raise RuntimeError("Only one image prefill per cached readout is supported")
        return (tuple(f.features.to(engine.device) for f in frames),
                [torch.cat([f.deepstack[layer].to(engine.device) for f in frames], dim=0)
                 for layer in range(layers)])

    sentinel = object()
    with _engine_lock(engine):
        previous = core.__dict__.get("get_image_features", sentinel)
        core.get_image_features = get_cached
        try:
            if any(frame.source_indices is not None for frame in frames):
                from onestreamer.sparse_visual_inputs import sparse_visual_positions
                positions = sparse_visual_positions(engine, frames)
            else:
                positions = nullcontext()
            with positions:
                yield state
        finally:
            if previous is sentinel:
                delattr(core, "get_image_features")
            else:
                core.get_image_features = previous


class NativeFeatureAdapter:
    """Thin adapter for the exact Qwen3-VL interface used by OneStreamer."""

    def __init__(self, engine):
        import transformers

        if transformers.__version__ != "4.57.6":
            raise RuntimeError("This native-feature adapter requires transformers==4.57.6")
        if engine.model.config.model_type != "qwen3_vl":
            raise ValueError("V0 supports Qwen3-VL only; other backbones need an adapter")
        self.engine = engine

    def encode(self, frames, max_pixels=None, store_on_cpu=True) -> list[VisualFrame]:
        import torch

        if not frames:
            return []
        engine = self.engine
        images = [image if max_pixels is not None else engine._pad_to_patch_grid(image)
                  for _, image in frames]
        options = dict(images=images, return_tensors="pt", do_resize=max_pixels is not None)
        if max_pixels is not None:
            options.update(min_pixels=1024, max_pixels=max_pixels)
        pixels = engine.processor.image_processor(**options)
        grid = pixels["image_grid_thw"]
        with _engine_lock(engine), torch.inference_mode():
            final, deepstack = engine.model.model.get_image_features(
                pixels["pixel_values"].to(device=engine.device, dtype=engine.model.dtype),
                image_grid_thw=grid.to(engine.device))
        patch = engine.processor.image_processor.patch_size
        def retain(tensor):
            tensor = tensor.detach()
            # History owns independent CPU storage. Recent query features can
            # stay on the accelerator, avoiding a needless device round trip.
            return tensor.to("cpu", copy=True) if store_on_cpu else tensor
        result = []
        offset = 0
        for (timestamp, source_image), feature, shape in zip(frames, final, grid.tolist()):
            length = feature.shape[0]
            result.append(VisualFrame(float(timestamp), tuple(shape),
                                     retain(feature),
                                     tuple(retain(x[offset:offset + length])
                                           for x in deepstack),
                                     (shape[2] * patch, shape[1] * patch),
                                     source_size=tuple(source_image.size)))
            offset += length
        if len(result) != len(frames) or any(x.shape[0] != offset for x in deepstack):
            raise RuntimeError("Native encoder returned inconsistent image/DeepStack lengths")
        return result

    def prepare_inputs(self, chat, frames):
        """Dense native expansion or sparse text-only expansion; no readout ViT."""
        import torch
        from PIL import Image

        engine = self.engine
        prompt = engine.processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        sparse = any(frame.source_indices is not None for frame in frames)
        if sparse:
            from onestreamer.sparse_visual_inputs import prepare_sparse_chat_inputs
            inputs = prepare_sparse_chat_inputs(engine, prompt, frames)
        else:
            images = [Image.new("RGB", frame.size) for frame in frames]
            inputs = engine.processor(text=[prompt], images=images, padding=True,
                                      do_resize=False, truncation=False, return_tensors="pt")
        expected_grid = torch.tensor([f.grid_thw for f in frames], dtype=torch.long)
        if not torch.equal(inputs["image_grid_thw"].cpu(), expected_grid):
            raise ValueError("Processor changed the cached image grid")
        visual_id = engine.model.config.image_token_id
        count = int((inputs["input_ids"] == visual_id).sum().item())
        if count != sum(f.token_count for f in frames):
            raise ValueError("Image placeholder count does not match cached visual tokens")
        if inputs["input_ids"].shape[0] != 1 or "pixel_values_videos" in inputs:
            raise ValueError("Only single-sample image-based QA is supported")
        # generate has expand_size=1. In this exact HF version the image branch
        # only checks pixel_values is not None before calling our validated hook.
        # Avoid copying the dummy processed image patches to the accelerator.
        inputs.pop("pixel_values", None)
        inputs = inputs.to(engine.device)
        inputs["pixel_values"] = torch.empty(0, device=engine.device, dtype=engine.model.dtype)
        metrics = dict(input_tokens=int(inputs["input_ids"].shape[-1]), visual_tokens=count,
                       text_tokens=int(inputs["input_ids"].shape[-1]) - count,
                       image_grid_thw=expected_grid.tolist(),
                       input_image_sizes=[list(f.size) for f in frames])
        if any(f.source_indices is not None for f in frames):
            merge = engine.model.config.vision_config.spatial_merge_size
            metrics.update(visual_position_mode="original_grid_sparse_mrope",
                           image_retained_tokens=[f.token_count for f in frames],
                           image_original_tokens=[math.prod(f.grid_thw) // merge**2 for f in frames])
        return inputs, metrics

    def build_chat(self, history, recent, question, direct=False):
        frames = list(history) + list(recent)
        if not recent or any(a.timestamp >= b.timestamp for a, b in zip(frames, frames[1:])):
            raise ValueError("History/recent evidence must be nonempty, distinct and chronological")
        system = ("Answer the question using only the supplied timestamped visual evidence. "
                  "HISTORICAL_VISUAL_MEMORY contains sparse earlier observations; "
                  "RECENT_VISUAL_CONTEXT contains the recent scene. Missing historical "
                  "frames do not prove an event never happened. Respect timestamps and "
                  "distinguish earlier states from the current state. "
                  + ("Follow the requested answer format exactly." if direct else
                     "Output only the uppercase letter of the correct option."))
        content = []
        for tag, items in (("HISTORICAL_VISUAL_MEMORY", history), ("RECENT_VISUAL_CONTEXT", recent)):
            content.append(dict(type="text", text=f"<{tag}>"))
            if not items:
                content.append(dict(type="text", text="No additional historical visual evidence."))
            for frame in items:
                content.extend([dict(type="text", text=f"<{frame.timestamp:.3f}s>"),
                                dict(type="image")])
            content.append(dict(type="text", text=f"</{tag}>"))
        content.append(dict(type="text", text=question))
        return [dict(role="system", content=[dict(type="text", text=system)]),
                dict(role="user", content=content)]

    def context_limit(self, context_tokens=None):
        model_limit = int(self.engine.model.config.text_config.max_position_embeddings)
        if model_limit <= 0:
            raise ValueError("Model must declare a positive max_position_embeddings")
        if context_tokens is not None and (
                not isinstance(context_tokens, int) or isinstance(context_tokens, bool)
                or context_tokens <= 0):
            raise ValueError("context_tokens must be a positive integer")
        return min(model_limit, context_tokens if context_tokens is not None else DEFAULT_CONTEXT_TOKENS)

    def build_answer_chat(self, history, recent, question, direct=False,
                          answer_prompt="standard", eval_mode=None, history_indices=None):
        if answer_prompt == "standard":
            return self.build_chat(history, recent, question, direct)
        if answer_prompt != "frame_id_witness":
            raise ValueError(f"Unknown answer prompt: {answer_prompt}")
        from onestreamer import frame_evidence
        specs = frame_evidence.frame_specs(history, recent, history_indices or {})
        return frame_evidence.build_chat(self, history, recent, question, direct, specs, eval_mode)

    def count_input_tokens(self, history, recent, question, direct=False,
                           answer_prompt="standard", eval_mode=None, history_indices=None):
        """Exact expanded length, without preprocessing images or running ViT."""
        return self._count_chat_tokens(self.build_answer_chat(
            history, recent, question, direct, answer_prompt, eval_mode, history_indices),
                                       list(history) + list(recent))

    def _count_chat_tokens(self, chat, frames):
        processor = self.engine.processor
        prompt = processor.apply_chat_template(
            chat,
            add_generation_prompt=True, tokenize=False)
        # Using the processor also preserves its text kwargs/defaults. Without
        # images each atomic image marker remains a single token at this stage.
        raw = processor(text=[prompt], images=None, padding=True, truncation=False,
                        return_tensors="pt")
        ids = raw["input_ids"]
        image_count = int((ids == self.engine.model.config.image_token_id).sum().item())
        if ids.shape[0] != 1 or image_count != len(frames):
            raise ValueError("Expected exactly one unexpanded image marker per cached frame")
        return int(ids.shape[-1]) + sum(f.token_count - 1 for f in frames)

    def build_state_chat(self, history, recent, question, direct=False):
        # Native states are prefixes of a QA response, not labels for a separate
        # "do not answer" classification task. Keep the active question before
        # the visual update, as in the native proactive-QA session. Only the
        # first output token is evaluated by probe_state; no answer/caption is
        # generated here or carried into the final answer/memory bank.
        chat = self.build_chat(history, recent, question, direct)
        chat[0]["content"] = [dict(type="text", text=STATE_SYSTEM_PROMPT)]
        question_turn = dict(role="user", content=[dict(type="text", text=(
            question + "\nWhen you can answer, begin with </Response> before the requested answer. "
            "Otherwise use the native </Standby> or </Silence> protocol."
        ))])
        visual_turn = dict(role="user", content=chat[1]["content"][:-1])
        return [chat[0], question_turn, visual_turn]

    def count_state_input_tokens(self, history, recent, question, direct=False):
        return self._count_chat_tokens(self.build_state_chat(history, recent, question, direct),
                                       list(history) + list(recent))

    def state_token_ids(self):
        tokenizer = self.engine.processor.tokenizer
        ids = {}
        for state in STATE_TOKENS:
            encoded = tokenizer.encode(state, add_special_tokens=False)
            token_id = tokenizer.convert_tokens_to_ids(state)
            if (len(encoded) != 1 or encoded[0] != token_id
                    or token_id == tokenizer.unk_token_id
                    or token_id not in tokenizer.all_special_ids):
                raise ValueError(f"Checkpoint lacks atomic native special state token {state}")
            ids[state] = token_id
        if len(set(ids.values())) != len(STATE_TOKENS):
            raise ValueError("Native state tokens must have distinct token IDs")
        return ids

    def probe_state(self, history, recent, question, direct=False, context_tokens=None):
        """One unrestricted next-token decision, using cached visual features.

        Record full-vocabulary probabilities: a non-state argmax is an explicit
        protocol failure, never silently coerced into a confident state.
        """
        import torch

        state_ids = self.state_token_ids()
        frames = list(history) + list(recent)
        limit = self.context_limit(context_tokens)
        expected = self.count_state_input_tokens(history, recent, question, direct)
        if expected + 1 > limit:
            raise ValueError(f"State input ({expected}) plus one token exceeds context limit {limit}")
        inputs, metrics = self.prepare_inputs(self.build_state_chat(history, recent, question, direct), frames)
        if metrics["input_tokens"] != expected:
            raise RuntimeError("Native state processor length differs from the validated context budget")
        with cached_image_features(self.engine, frames) as hook, torch.inference_mode():
            output = self.engine.model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
            logits = output.logits[0, -1].float()
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Nonfinite state logits")
            probabilities = logits.softmax(dim=-1)
            predicted_id = int(logits.argmax().item())
            state_probabilities = {state: float(probabilities[token_id].item())
                                   for state, token_id in state_ids.items()}
            predicted_probability = float(probabilities[predicted_id].item())
        if hook["calls"] != 1:
            raise RuntimeError("Native cached state features were not consumed exactly once")
        tokenizer = self.engine.processor.tokenizer
        metrics.update(context_limit=limit, state_reserve_tokens=1,
                       state_prompt_version=STATE_PROMPT_VERSION,
                       remaining_context_tokens=limit - expected - 1)
        return dict(state=next((state for state, token_id in state_ids.items()
                                if token_id == predicted_id), None),
                    predicted_token=tokenizer.decode([predicted_id], skip_special_tokens=False),
                    predicted_token_id=predicted_id, predicted_probability=predicted_probability,
                    state_probabilities=state_probabilities,
                    state_probability_mass=sum(state_probabilities.values()),
                    **metrics)

    def build_read_action_chat(self, history, recent, question, history_status, direct=False):
        """Reuse visual blocks, but ask for a reading action rather than an answer."""
        chat = self.build_chat(history, recent, question, direct)
        status = json.dumps(history_status, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return [
            dict(role="system", content=[dict(type="text", text=READ_ACTION_SYSTEM_PROMPT)]),
            dict(role="user", content=[dict(type="text", text=f"<VIDEO_QUESTION>\n{question}\n</VIDEO_QUESTION>")]),
            dict(role="user", content=chat[1]["content"][:-1] + [
                dict(type="text", text=f"<HISTORY_STATUS>\n{status}\n</HISTORY_STATUS>"),
                dict(type="text", text="Choose the reading action only: A = STOP; B = READ_MORE. Output one letter."),
            ]),
        ]

    def count_read_action_input_tokens(self, history, recent, question, history_status, direct=False):
        return self._count_chat_tokens(
            self.build_read_action_chat(history, recent, question, history_status, direct),
            list(history) + list(recent))

    def read_action_token_ids(self):
        """The action protocol uses existing ordinary atomic tokens, not new states."""
        tokenizer = self.engine.processor.tokenizer
        ids = {}
        for label in READ_ACTION_LABELS:
            encoded = tokenizer.encode(label, add_special_tokens=False)
            if (len(encoded) != 1 or encoded[0] in tokenizer.all_special_ids
                    or encoded[0] == tokenizer.unk_token_id
                    or tokenizer.decode(encoded, skip_special_tokens=False) != label):
                raise ValueError(f"Checkpoint lacks ordinary atomic read-action token {label}")
            ids[label] = encoded[0]
        if len(set(ids.values())) != len(ids):
            raise ValueError("Read-action tokens must have distinct token IDs")
        return ids

    def probe_read_action(self, history, recent, question, history_status,
                          direct=False, context_tokens=None):
        """One raw next-token decision; invalid predictions remain explicit.

        The A/B probabilities are full-vocabulary action probabilities, never
        answer correctness or a forced two-label renormalization.
        """
        import torch

        label_ids = self.read_action_token_ids()
        frames = list(history) + list(recent)
        limit = self.context_limit(context_tokens)
        expected = self.count_read_action_input_tokens(history, recent, question, history_status, direct)
        if expected + 1 > limit:
            raise ValueError(f"Read-action input ({expected}) plus one token exceeds context limit {limit}")
        chat = self.build_read_action_chat(history, recent, question, history_status, direct)
        inputs, metrics = self.prepare_inputs(chat, frames)
        if metrics["input_tokens"] != expected:
            raise RuntimeError("Read-action processor length differs from the validated context budget")
        with cached_image_features(self.engine, frames) as hook, torch.inference_mode():
            output = self.engine.model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
            logits = output.logits[0, -1].float()
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Nonfinite read-action logits")
            probabilities = logits.softmax(dim=-1)
            predicted_id = int(logits.argmax().item())
            action_probabilities = {READ_ACTION_LABELS[label]: float(probabilities[token_id].item())
                                    for label, token_id in label_ids.items()}
            predicted_probability = float(probabilities[predicted_id].item())
        if hook["calls"] != 1:
            raise RuntimeError("Cached read-action features were not consumed exactly once")
        predicted_label = next((label for label, token_id in label_ids.items() if token_id == predicted_id), None)
        metrics.update(context_limit=limit, action_reserve_tokens=1,
                       action_prompt_version=READ_ACTION_PROMPT_VERSION,
                       remaining_context_tokens=limit - expected - 1)
        return dict(action=READ_ACTION_LABELS.get(predicted_label), action_label=predicted_label,
                    predicted_token=self.engine.processor.tokenizer.decode([predicted_id], skip_special_tokens=False),
                    predicted_token_id=predicted_id, predicted_probability=predicted_probability,
                    action_probabilities=action_probabilities,
                    action_probability_mass=sum(action_probabilities.values()), **metrics)

    def select_history(self, candidates, recent, question, direct=False, max_tokens=16,
                       context_tokens=None, answer_prompt="standard", eval_mode=None,
                       history_indices=None):
        """Greedily fill the remaining context with whole stored image bundles.

        Try temporally spread candidates first, continuing past images that do
        not fit. Count timestamp/wrapper/question tokens as well as visual ones.
        No stored evidence is re-encoded or recompressed at question time.
        """
        limit = self.context_limit(context_tokens)
        if max_tokens <= 0 or max_tokens >= limit:
            raise ValueError("Answer reserve must be positive and smaller than context limit")
        prompt_kwargs = ({} if answer_prompt == "standard" else
                         dict(answer_prompt=answer_prompt, eval_mode=eval_mode,
                              history_indices=history_indices))
        base = self.count_input_tokens([], recent, question, direct, **prompt_kwargs)
        input_limit = limit - max_tokens
        if base > input_limit:
            raise ValueError(f"Recent context/question needs {base} input tokens plus {max_tokens} "
                             f"answer tokens, exceeding context limit {limit}; increase the cap")
        candidates = sorted(candidates, key=lambda frame: frame.timestamp)
        # Validates causality, uniqueness and recent order even for images that
        # will eventually be rejected for their size.
        self.build_answer_chat(candidates, recent, question, direct, **prompt_kwargs)
        order, seen = [], set()
        count = min(2, len(candidates))
        while count:
            for index in uniform_indices(len(candidates), count):
                if index not in seen:
                    order.append(index)
                    seen.add(index)
            if count == len(candidates):
                break
            count = min(count * 2, len(candidates))
        selected, used = [], base
        count_calls = 1  # Includes the recent-only baseline above.

        def measure(items):
            nonlocal count_calls
            count_calls += 1
            return self.count_input_tokens(items, recent, question, direct, **prompt_kwargs)

        # Large progressive banks can contain thousands of one-token records.
        # Avoid retokenizing a growing full prompt once per stored frame.
        tail = order
        tail_skipped = 0
        if len(order) > 128:
            all_cost = measure(candidates)
            if all_cost <= input_limit:
                selected, used, tail = candidates, all_cost, []
            else:
                recent_visual = sum(f.token_count for f in recent)
                viable = [index for index in order
                          if recent_visual + candidates[index].token_count <= input_limit
                          and measure([candidates[index]]) <= input_limit]
                low, high = 0, len(viable)
                while low < high:
                    middle = (low + high + 1) // 2
                    trial = sorted((candidates[i] for i in viable[:middle]), key=lambda f: f.timestamp)
                    cost = measure(trial)
                    if cost <= input_limit:
                        low = middle
                        selected, used = trial, cost
                    else:
                        high = middle - 1
                tail = viable[low:low + 128]
                tail_skipped = max(0, len(viable) - low - len(tail))
        for index in tail:
            trial = sorted([*selected, candidates[index]], key=lambda frame: frame.timestamp)
            cost = measure(trial)
            if cost <= input_limit:
                selected, used = trial, cost
        return selected, dict(context_limit=limit, answer_reserve_tokens=max_tokens,
                              recent_only_input_tokens=base, selected_input_tokens=used,
                              history_extra_input_tokens=used - base,
                              remaining_input_tokens=input_limit - used,
                              eligible_history_frames=len(candidates),
                              budget_count_calls=count_calls,
                              tail_candidates_skipped=tail_skipped)

    def answer(self, history, recent, question, direct=False, max_tokens=16, context_tokens=None,
               answer_prompt="standard", eval_mode=None, history_indices=None):
        import torch

        frames = list(history) + list(recent)
        limit = self.context_limit(context_tokens)
        prompt_kwargs = ({} if answer_prompt == "standard" else
                         dict(answer_prompt=answer_prompt, eval_mode=eval_mode,
                              history_indices=history_indices))
        expected = self.count_input_tokens(history, recent, question, direct, **prompt_kwargs)
        if max_tokens <= 0 or expected + max_tokens > limit:
            raise ValueError(f"Input ({expected}) plus answer reserve ({max_tokens}) "
                             f"exceeds context limit {limit}")
        chat = self.build_answer_chat(history, recent, question, direct, **prompt_kwargs)
        inputs, metrics = self.prepare_inputs(chat, frames)
        if metrics["input_tokens"] != expected or metrics["input_tokens"] + max_tokens > limit:
            raise RuntimeError("Native processor length differs from the validated context budget")
        metrics.update(context_limit=limit, answer_reserve_tokens=max_tokens,
                       remaining_context_tokens=limit - metrics["input_tokens"] - max_tokens)
        tokenizer = self.engine.processor.tokenizer
        pad = tokenizer.pad_token_id
        with cached_image_features(self.engine, frames) as state, torch.inference_mode():
            output = self.engine.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=True,
                num_beams=1, num_return_sequences=1, temperature=1.0, top_p=1.0, top_k=0,
                pad_token_id=tokenizer.eos_token_id if pad is None else pad)
        if state["calls"] != 1:
            raise RuntimeError("Native cached features were not consumed exactly once")
        text = self.engine.processor.batch_decode(
            output[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=False)[0]
        return self.engine._strip_end_tokens(text), metrics


class StreamSampler:
    """Persistent decoder with a fixed origin; never prefetches beyond a cutoff."""

    def __init__(self, path, start, fps, include_end=False):
        import cv2

        self._cv2 = cv2
        self.include_end = include_end
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError(f"Cannot open video: {path}")
        self.source_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(self.source_fps) or self.source_fps <= 0 or self.total_frames <= 0:
            self.close()
            raise ValueError(f"Invalid video metadata: {path}")
        if start < 0 or not math.isfinite(start) or fps <= 0 or not math.isfinite(fps):
            self.close()
            raise ValueError("Invalid stream start/fps")
        self.origin, self.fps = start, fps
        self.observation_time = start
        self.slot, self.last_sample = 0, -1
        self.current = math.floor(start * self.source_fps)
        if self.current >= self.total_frames:
            self.close()
            raise ValueError("Stream start lies outside the video")
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current)
        self.cutoff = start

    def advance(self, end):
        from PIL import Image

        if not math.isfinite(end) or end < self.cutoff:
            raise ValueError("Cannot move a stream cutoff backwards")
        self.cutoff = end
        duration = self.total_frames / self.source_fps
        while True:
            scheduled = self.origin + self.slot / self.fps
            # OVO chunks end exactly at the question time, so the end-inclusive
            # slot may coincide with the video duration.
            if (scheduled > min(end, duration) if self.include_end
                    else scheduled >= duration or scheduled >= end):
                break
            index = math.floor(scheduled * self.source_fps)
            if self.include_end:
                # A slot at the cutoff takes the last source frame strictly
                # before it, as the official end-aligned recent reader does.
                cap = min(math.ceil(end * self.source_fps), self.total_frames) - 1
                if index > cap and cap < self.current:
                    break  # No unread frame precedes the cutoff; keep the slot.
                index = min(index, cap)
            self.slot += 1
            if index == self.last_sample:
                continue
            frame = None
            while self.current <= index:
                ok, frame = self.capture.read()
                if not ok:
                    raise RuntimeError(f"Failed to decode source frame {self.current}")
                self.current += 1
            if frame is None:
                raise RuntimeError("Sampling cursor moved behind the decoder")
            self.last_sample = index
            # Availability is the fixed sampling time, which may be later than
            # the source-frame timestamp after floor-to-frame quantization.
            self.observation_time = scheduled
            yield index / self.source_fps, Image.fromarray(
                self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB))

    def close(self):
        self.capture.release()


def _ensure_paths():
    here = Path(__file__).resolve()
    for directory in (here.parents[1], here.parents[3]):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))


def _sync(engine):
    import torch

    device = torch.device(engine.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def infer(job, rows, config=None, engine=None):
    """Independent runner for official OVO/StreamingBench sample dictionaries."""
    _ensure_paths()
    from onestreamer.engine import OneStreamerEngine
    from onestreamer.ovobench import build_question_text, score_prediction
    from onestreamer.schema import Sample
    from onestreamer.video import read_recent_frames

    config = config or MemoryConfig()
    frame_evidence = None
    if config.answer_prompt == "frame_id_witness":
        from onestreamer import frame_evidence
    answer_tokens = (frame_evidence.ANSWER_RESERVE_TOKENS if frame_evidence is not None
                     else 16)
    engine = engine or OneStreamerEngine(
        model_path=job["model_path"], dtype=job.get("dtype", "bfloat16"),
        device=job.get("device", "cuda"),
        attn_implementation=job.get("attn_implementation", "flash_attention_2"))
    adapter = NativeFeatureAdapter(engine)
    if config.read_mode == "state":
        # Fail before expensive video ingestion for incompatible checkpoints.
        adapter.state_token_ids()
    elif config.read_mode == "action":
        adapter.read_action_token_ids()
    router = None
    if config.read_mode == "routed":
        from onestreamer import question_router
        router = question_router.QuestionRouter(engine)
    groups = {}
    for row in rows:
        sample = Sample.from_dict(row)
        path = Path(row.get("media") or (Path(job["video_root"]) / sample.video)).resolve()
        groups.setdefault((str(path), sample.start), []).append((sample, row))
    for (path, origin), items in sorted(groups.items()):
        progressive = config.history_compression == "progressive_pool"
        budget_pool = config.history_compression == "budget_pool"
        vcast = config.history_compression == "vcast"
        if vcast:
            from onestreamer.vcast_memory import VCastVisualMemoryBank
        bank = (ProgressiveVisualMemoryBank(config, engine.model.config.vision_config.spatial_merge_size)
                if progressive else
                BudgetVisualMemoryBank(config, engine.model.config.vision_config.spatial_merge_size)
                if budget_pool else
                VCastVisualMemoryBank(config, engine.model.config.vision_config.spatial_merge_size)
                if vcast else VisualMemoryBank(config))
        # Size the bank from the first native frame, before any FIFO retirement.
        budget_pending = (budget_pool or vcast) and config.store_budget == "context" and config.enabled
        pooled_stream = config.history_compression in FEATURE_STREAM_COMPRESSIONS
        window = RecentFeatureWindow(capacity=config.recent_frames) if pooled_stream else None
        # The pooling-mode recent-only control uses the identical causal stream
        # and full-resolution FIFO, even when historical storage is disabled.
        sampler = (StreamSampler(path, origin, config.history_fps, include_end=config.stream_include_end)
                   if config.enabled or pooled_stream else None)
        observation_total = 0.0
        encoded_frames = 0
        pooled_input_tokens = pooled_output_tokens = 0
        try:
            for sample, row in sorted(items, key=lambda item: (item[0].end, item[0].sample_id)):
                _sync(engine)
                start = time.perf_counter()
                if sampler is not None:
                    for frame in sampler.advance(sample.end):
                        if progressive and config.enabled:
                            bank.advance(sampler.observation_time)
                        if pooled_stream:
                            native = adapter.encode([frame], store_on_cpu=False)[0]
                            if budget_pending:
                                prompts = [(build_question_text(s), dict(
                                    direct=s.eval_mode != "mcq_letter",
                                    **({} if frame_evidence is None else dict(
                                        answer_prompt=config.answer_prompt, eval_mode=s.eval_mode,
                                        history_indices={}))))
                                           for s, _ in items]
                                bank.set_budget(*context_history_budget(
                                    adapter, native.token_count, prompts, config.recent_frames,
                                    answer_tokens, config.context_tokens))
                                budget_pending = False
                            retired = window.observe(native)
                            if retired is not None and config.enabled:
                                if vcast:
                                    # The oldest recent frame is the observed successor of the retired one.
                                    bank.observe(retired, now=sampler.observation_time, successor=window.frames()[0])
                                elif progressive or budget_pool:
                                    bank.observe(retired, now=sampler.observation_time)
                                else:
                                    compressed = pool_visual_frame(
                                        retired, stride=config.pool_stride,
                                        merge_size=engine.model.config.vision_config.spatial_merge_size)
                                    pooled_input_tokens += retired.token_count
                                    pooled_output_tokens += compressed.token_count
                                    bank.observe(compressed)
                                    del compressed
                            # Do not keep an evicted full-resolution GPU tensor
                            # alive until the next query/observation.
                            del retired, native
                        else:
                            bank.observe(adapter.encode([frame], max_pixels=config.history_max_pixels)[0])
                        encoded_frames += 1
                if progressive and config.enabled:
                    bank.advance(sample.end)
                if (progressive or budget_pool) and config.enabled:
                    progressive_stats = bank.stats()
                    pooled_input_tokens = progressive_stats["pool_input_tokens"]
                    pooled_output_tokens = progressive_stats["pool_output_tokens"]
                _sync(engine)
                observation_delta = time.perf_counter() - start
                observation_total += observation_delta
                start = time.perf_counter()
                if pooled_stream:
                    recent = window.frames()
                    recent_sizes = [list(f.source_size or f.size) for f in recent]
                else:
                    # Legacy branch retains exact official end-aligned pixels.
                    recent_pixels = read_recent_frames(path, sample.start, sample.end, config.recent_frames, 1.0)
                    recent = adapter.encode(recent_pixels, store_on_cpu=False)
                    recent_sizes = [list(image.size) for _, image in recent_pixels]
                if not recent:
                    raise ValueError("No sampled recent frame is available at this cutoff")
                question = build_question_text(sample)
                direct = sample.eval_mode != "mcq_letter"
                selection_meta = {}
                history_indices = None
                routed_answer = None
                if config.read_mode == "routed":
                    # The question is known only now; memory writing above never saw it.
                    route = router.route(sample.question)
                    final_scope = route["label"]
                    if final_scope == "NOW":
                        now_prediction, now_metrics, confidence, now_frames = question_router.official_recent_answer(
                            engine, sample, row["media"], question)
                        route["recent_confidence"] = confidence
                        if confidence >= config.route_threshold:
                            routed_answer = (now_prediction, now_metrics)
                            recent_sizes = [list(image.size) for _, image in now_frames]
                            routed_recent_timestamps = [t for t, _ in now_frames]
                        else:
                            final_scope = config.route_escalation
                    route["final_scope"] = final_scope
                    selection_meta = dict(route=route)
                    if routed_answer is not None:
                        history = []
                    else:
                        if final_scope == "SHORT" and vcast:
                            # Same frames as the budget_pool SHORT read, fitted to the same cap.
                            candidates, short_meta = bank.short_history(recent[0].timestamp,
                                                                        config.short_history_tokens)
                            route.update(short_meta)
                        else:
                            candidates = bank.candidates(before=recent[0].timestamp)
                        if final_scope == "SHORT" and not vcast:
                            candidates = question_router.newest_history(candidates, config.short_history_tokens)
                        history, fill_meta = adapter.select_history(
                            candidates, recent, question, direct=direct,
                            max_tokens=answer_tokens, context_tokens=config.context_tokens)
                        selection_meta.update(fill_meta)
                elif config.read_mode == "state":
                    from onestreamer.state_readout import run_state_readout
                    history, selection_meta = run_state_readout(
                        adapter, bank.candidates(before=recent[0].timestamp), recent, question,
                        direct=direct, history_budgets=config.state_history_budgets,
                        max_tokens=16, context_tokens=config.context_tokens)
                elif config.read_mode == "action":
                    from onestreamer.action_readout import run_action_readout
                    history, selection_meta = run_action_readout(
                        adapter, bank.candidates(before=recent[0].timestamp), recent, question,
                        query_time=sample.end, stream_start=origin, direct=direct,
                        history_budgets=config.state_history_budgets,
                        max_tokens=16, context_tokens=config.context_tokens)
                elif config.read_mode == "fill":
                    candidates = bank.candidates(before=recent[0].timestamp)
                    if frame_evidence is not None:
                        history_indices = {f.timestamp: i for i, f in enumerate(candidates)}
                        history, selection_meta = adapter.select_history(
                            candidates, recent, question, direct=direct,
                            max_tokens=answer_tokens, context_tokens=config.context_tokens,
                            answer_prompt=config.answer_prompt, eval_mode=sample.eval_mode,
                            history_indices=history_indices)
                    else:
                        history, selection_meta = adapter.select_history(
                            candidates, recent, question, direct=direct,
                            max_tokens=answer_tokens, context_tokens=config.context_tokens)
                else:
                    history = bank.read(before=recent[0].timestamp)
                if any(f.timestamp >= sample.end for f in history):
                    raise RuntimeError("Future visual evidence entered the readout")
                _sync(engine)
                preparation_seconds = time.perf_counter() - start
                qa_start = time.perf_counter()
                if frame_evidence is not None:
                    raw_prediction, metrics = adapter.answer(
                        history, recent, question, direct=direct, max_tokens=answer_tokens,
                        context_tokens=config.context_tokens,
                        answer_prompt=config.answer_prompt, eval_mode=sample.eval_mode,
                        history_indices=history_indices)
                    specs = frame_evidence.frame_specs(history, recent, history_indices)
                    witness = frame_evidence.parse_witness(
                        raw_prediction, specs, sample.eval_mode, question,
                        valid_letters="".join(sample.options))
                    prediction = frame_evidence.answer_for_scoring(witness)
                    evidence_meta = dict(prompt_version=frame_evidence.PROMPT_VERSION,
                                         frame_map=specs, **witness)
                    evidence_fields = dict(raw_prediction=raw_prediction,
                                           frame_evidence=evidence_meta)
                elif routed_answer is not None:
                    prediction, metrics = routed_answer
                    evidence_fields = {}
                else:
                    prediction, metrics = adapter.answer(
                        history, recent, question, direct=direct, max_tokens=answer_tokens,
                        context_tokens=config.context_tokens)
                    evidence_fields = {}
                _sync(engine)
                qa_seconds = time.perf_counter() - qa_start
                parsed, hit = score_prediction(sample, prediction)
                yield dict(row, model_alias=job.get("model", Path(job["model_path"]).name),
                           variant=(f"{VCAST_VERSION if vcast else VERSION}_{config.history_compression}_{config.read_mode}_r{config.recent_frames}"
                                    + (f"_recent{config.recent_frames}_control" if not config.enabled else "")
                                    + ("_frame_id_witness_v2" if frame_evidence is not None else "")),
                           prediction=prediction, parsed_output=parsed, hit=hit,
                           **evidence_fields,
                           memory_text="", memory_meta={"memory_tokens": 0}, caption_cache=None,
                           caption_counts={}, history_timestamps=[f.timestamp for f in history],
                           history_visual_tokens=sum(f.token_count for f in history),
                           history_pool_scales=[f.pool_scale for f in history],
                           history_frame_tokens=[f.token_count for f in history],
                           history_read_frames=len(history), bank=bank.stats(),
                           history_selection_meta=selection_meta,
                           state_probe_seconds=selection_meta.get("state_probe_seconds", 0.0),
                           state_rounds=selection_meta.get("state_rounds", []),
                           state_stop_reason=selection_meta.get("state_stop_reason"),
                           action_probe_seconds=selection_meta.get("action_probe_seconds", 0.0),
                           action_rounds=selection_meta.get("action_rounds", []),
                           action_stop_reason=selection_meta.get("action_stop_reason"),
                           action_invalid_count=selection_meta.get("action_invalid_count", 0),
                           recent_timestamps=(routed_recent_timestamps if routed_answer is not None
                                              else [f.timestamp for f in recent]),
                           route=selection_meta.get("route"),
                           recent_frames=config.recent_frames, recent_read_frames=len(recent),
                           recent_image_sizes=recent_sizes,
                           recent_sampling=("official_end_aligned" if routed_answer is not None or not pooled_stream
                                            else "fixed_origin_fifo"),
                           stream_encoded_frames_cumulative=encoded_frames,
                           pooled_input_tokens_cumulative=pooled_input_tokens,
                           pooled_output_tokens_cumulative=pooled_output_tokens,
                           observation_seconds_delta=observation_delta,
                           observation_seconds_cumulative=observation_total,
                           recent_preparation_seconds=preparation_seconds,
                           qa_seconds=qa_seconds,
                           runtime_seconds=preparation_seconds + qa_seconds,
                           total_runtime_seconds=observation_delta + preparation_seconds + qa_seconds,
                           **metrics)
                # A suspended generator may retain the current query's tensors;
                # release these references before advancing the next query.
                del recent, history
        finally:
            if sampler is not None:
                sampler.close()


def _load_rows(path):
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return [json.loads(line) for line in raw.splitlines() if line.strip()], {}
    if isinstance(value, list):
        return value, {}
    if isinstance(value, dict) and "samples" in value:
        return value["samples"], value
    if isinstance(value, dict) and "sample_id" in value:
        return [value], {}
    raise ValueError("Expected a sample list, sample JSONL, or manifest containing samples")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--samples", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--subtasks", nargs="+", help="Exact official subtask labels, e.g. EPM ASI")
    parser.add_argument("--recent-frames", type=int, default=16,
                        help="Full-feature recent window size; historical pooling starts only upon FIFO eviction")
    parser.add_argument("--history-capacity", type=int,
                        help="Frame cap; defaults to token cap for progressive_pool/budget_pool/vcast, otherwise 64; 0 disables history")
    parser.add_argument("--history-store-tokens", type=int, default=16384)
    parser.add_argument("--history-store-mib", type=float, default=512)
    parser.add_argument("--history-read-frames", type=int, default=4)
    parser.add_argument("--history-read-tokens", type=int, default=1024)
    parser.add_argument("--route-threshold", type=float, default=0.0,
                        help="routed: NOW questions whose Recent16 first-token probability is below this read history")
    parser.add_argument("--route-escalation", choices=("SHORT", "LONG"), default="SHORT")
    parser.add_argument("--short-history-tokens", type=int, default=16384)
    parser.add_argument("--history-read-mode", choices=("fixed", "fill", "state", "action", "routed"), default="fixed",
                        help="fill fills context; state uses native readiness; action requests history with A/B actions")
    parser.add_argument("--answer-prompt", choices=("standard", "frame_id_witness"),
                        default="standard", help="Exploratory frame-ID witness answer requires --history-read-mode fill")
    parser.add_argument("--state-history-budgets", type=int, nargs="+", default=[1024, 4096, 16384],
                        help="Increasing historical visual-token budgets after the recent-only state/action probe")
    parser.add_argument("--context-tokens", type=int, default=DEFAULT_CONTEXT_TOKENS,
                        help="Total context cap including answer reserve; default: 32768 (32K)")
    parser.add_argument("--history-fps", type=float, default=1.0)
    parser.add_argument("--history-max-pixels", type=int, default=262144)
    parser.add_argument("--history-compression", choices=("resize", "avg_pool", "progressive_pool", "budget_pool", "vcast"),
                        default="resize",
                        help="progressive_pool repeatedly compresses aged frames; avg_pool compresses once on FIFO "
                             "eviction; budget_pool coarsens records under pressure; vcast selects original tokens")
    parser.add_argument("--vcast-temperature", type=float, default=0.7,
                        help="vcast temporal-curvature allocation temperature (positive; default 0.7)")
    parser.add_argument("--history-store-budget", choices=("fixed", "context"), default="fixed",
                        help="context sizes the budget_pool/vcast bank per stream to the context left after recent "
                             "frames, the longest question prompt and the answer reserve")
    parser.add_argument("--history-min-frame-tokens", type=int, default=16,
                        help="Per-frame floor before budget_pool/vcast thins records uniformly in time")
    parser.add_argument("--stream-include-end", action="store_true",
                        help="Sample the slot at the question time using the last source frame before it, "
                             "matching the official end-aligned final recent frame")
    parser.add_argument("--history-pool-stride", type=int, default=2,
                        help="Spatial pooling stride per axis; 2 gives about 4x fewer historical tokens")
    parser.add_argument("--history-pool-age-seconds", type=float, default=60.0,
                        help="Video seconds after FIFO eviction between progressive pooling steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("flash_attention_2", "sdpa", "eager"),
                        default="flash_attention_2")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without loading the model")
    args = parser.parse_args(argv)
    _ensure_paths()
    from onestreamer.schema import Sample

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not math.isfinite(args.history_store_mib) or args.history_store_mib < 0:
        parser.error("--history-store-mib must be finite and nonnegative")
    capacity = args.history_capacity
    if capacity is None:
        capacity = (args.history_store_tokens if args.history_compression in {"progressive_pool", "budget_pool", "vcast"}
                    else 64)
    config = MemoryConfig(
        capacity=capacity, store_tokens=args.history_store_tokens,
        store_bytes=int(args.history_store_mib * 1024**2), read_frames=args.history_read_frames,
        read_tokens=args.history_read_tokens, history_fps=args.history_fps,
        history_max_pixels=args.history_max_pixels, seed=args.seed,
        read_mode=args.history_read_mode, context_tokens=args.context_tokens,
        history_compression=args.history_compression, pool_stride=args.history_pool_stride,
        pool_age_seconds=args.history_pool_age_seconds, recent_frames=args.recent_frames,
        state_history_budgets=tuple(args.state_history_budgets), answer_prompt=args.answer_prompt,
        store_budget=args.history_store_budget, min_frame_tokens=args.history_min_frame_tokens,
        stream_include_end=args.stream_include_end, route_threshold=args.route_threshold,
        route_escalation=args.route_escalation, short_history_tokens=args.short_history_tokens,
        vcast_temperature=args.vcast_temperature)
    if config.answer_prompt == "frame_id_witness":
        from onestreamer.frame_evidence import ANSWER_RESERVE_TOKENS, PROMPT_VERSION
        answer_tokens, recent_prompt = ANSWER_RESERVE_TOKENS, PROMPT_VERSION
    else:
        answer_tokens, recent_prompt = 16, "visual_memory_v0_common_control"
    rows, original = _load_rows(args.manifest or args.samples)
    model_path = args.model_path or original.get("model_path")
    if not model_path:
        parser.error("--model-path is required when it is absent from the input manifest")
    ids = [Sample.from_dict(row).sample_id for row in rows]
    if len(set(ids)) != len(ids):
        parser.error("Duplicate sample IDs")
    if args.subtasks:
        unknown = set(args.subtasks) - {row.get("subtask", "") for row in rows}
        if unknown:
            parser.error(f"Unknown subtasks: {sorted(unknown)}")
        rows = [row for row in rows if row.get("subtask") in args.subtasks]
    rows = rows[:args.limit] if args.limit is not None else rows
    if not rows:
        parser.error("No samples selected")
    for row in rows:
        sample = Sample.from_dict(row)
        # Persist schema defaults too (e.g. an omitted start means 0.0).
        row.update(asdict(sample))
        if not math.isfinite(sample.start) or not math.isfinite(sample.end) or sample.start < 0:
            parser.error(f"Invalid stream bounds: {sample.sample_id}")
        if args.video_root is not None:
            path = args.video_root / sample.video
        elif row.get("media"):
            path = Path(row["media"])
        elif original.get("video_root"):
            path = Path(original["video_root"]) / sample.video
        else:
            parser.error("--video-root is required when input samples have no media path")
        if not path.is_file():
            parser.error(f"Video does not exist: {path}")
        row["media"] = str(path.resolve())
    job = dict(model=original.get("model", Path(model_path).name), model_path=model_path,
               device=args.device, dtype=args.dtype, attn_implementation=args.attn_implementation,
               video_root=str(args.video_root or original.get("video_root", ".")))
    pooled_stream = config.history_compression in FEATURE_STREAM_COMPRESSIONS
    progressive = config.history_compression == "progressive_pool"
    budget_pool = config.history_compression == "budget_pool"
    vcast = config.history_compression == "vcast"
    manifest = dict(version=VCAST_VERSION if vcast else VERSION, config=asdict(config), job=job, samples=rows,
                    source_manifest=str((args.manifest or args.samples).resolve()),
                    protocol=dict(recent_frames=config.recent_frames, recent_fps=1.0,
                                  answer_tokens=answer_tokens,
                                  answer_prompt=config.answer_prompt,
                                  caption_generation=False, history_sampling=("fixed_origin_causal_end_inclusive"
                                                    if config.stream_include_end else "fixed_origin_causal"),
                                  history_storage=("selected_original_tokens_with_deepstack_and_coordinates" if vcast else
                                                   "pooled_native_features_with_deepstack" if pooled_stream
                                                   else "native_full_image_features_with_deepstack"),
                                  history_write=("on_recent_fifo_eviction_then_budget_select" if vcast else
                                                 "on_recent_fifo_eviction_then_age_pool" if progressive else
                                                 "on_recent_fifo_eviction_then_budget_pool" if budget_pool else
                                                 "on_recent_fifo_eviction" if pooled_stream
                                                 else "on_observation"),
                                  history_selection=("causal_curvature_budget_and_dual_anchor_tokens" if vcast else
                                                     "age_pool_then_drop_oldest_at_caps" if progressive
                                                     else "coarsen_oldest_finest_then_uniform_thin" if budget_pool
                                                     else "seeded_priority_reservoir"),
                                  read_selection=("native_state_progressive_visual_read" if config.read_mode == "state" else
                                                  "explicit_action_progressive_visual_read" if config.read_mode == "action" else
                                                  "question_routed_recent16_or_history" if config.read_mode == "routed" else
                                                  "temporal_spread_context_fill" if config.read_mode == "fill"
                                                  else "uniform_temporal"),
                                  recent_replay=("fixed_origin_fifo_cached_features" if pooled_stream
                                                 else "official_decoder"),
                                  recent_prompt=recent_prompt,
                                  runtime_seconds=("cached_recent_readout_plus_qa" if pooled_stream
                                                   else "recent_decode_encode_readout_plus_qa"),
                                  storage_bytes=("feature_coordinate_and_selector_payload" if vcast else
                                                 "tensor_payload_including_deepstack")))
    if config.answer_prompt == "frame_id_witness":
        manifest["protocol"].update(
            witness_prompt=recent_prompt,
            witness_answer_extraction="valid_ANSWER_line_independent_of_citation_format",
            witness_frame_ids="H_index_in_query_eligible_stored_history_R_index_in_recent",
            witness_citation_audit="syntax_and_supplied_frame_ids_only_not_semantic_support",
            witness_task_labels_in_prompt=False)
    if config.read_mode == "state":
        manifest["protocol"].update(state_prompt=STATE_PROMPT_VERSION,
                                   state_probe="full_vocabulary_greedy_next_token",
                                   state_invalid_fallback="answer_current_view",
                                   state_task_labels=False,
                                   state_budgets_include_initial_zero=True,
                                   state_timing="included_in_recent_preparation_and_runtime_seconds")
    if config.read_mode == "action":
        manifest["protocol"].update(action_prompt=READ_ACTION_PROMPT_VERSION,
                                   action_probe="full_vocabulary_greedy_next_token",
                                   action_labels=READ_ACTION_LABELS,
                                   action_invalid_fallback="continue_to_next_view",
                                   action_task_labels=False,
                                   action_budgets_include_initial_zero=True,
                                   action_timing="included_in_recent_preparation_and_runtime_seconds")
    source_paths = [Path(__file__)]
    if vcast:
        source_paths.extend(Path(__file__).with_name(name)
                            for name in ("vcast_memory.py", "sparse_visual_inputs.py"))
        manifest["protocol"].update(
            compression_reference="xinyouu/V-CAST@f15d2764e2381e0344fea9eb6228a4a58654f4f7",
            compression_variant="causal_pressure_only_irreversible_selection_v1",
            visual_positions="original_merged_grid_coordinates",
            temporal_allocation="observed_retained_frame_curvature_softmax_with_observed_successor",
            min_frame_tokens_policy="thin_frames_before_relaxing_last_frame_floor",
            selection_materialization="lazy_prefix_of_fixed_score_ranking_trim_at_2x",
            short_history_read="budget_pool_pooled_span_then_vcast_fit_to_cap")
    if config.read_mode in {"state", "action"}:
        source_paths.append(Path(__file__).with_name("state_readout.py"))
    if config.read_mode == "action":
        source_paths.append(Path(__file__).with_name("action_readout.py"))
    if config.answer_prompt == "frame_id_witness":
        source_paths.append(Path(__file__).with_name("frame_evidence.py"))
    if config.read_mode == "routed":
        source_paths.append(Path(__file__).with_name("question_router.py"))
    manifest["source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in source_paths}
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["fingerprint"] = fingerprint
    if args.dry_run:
        print(json.dumps(dict(samples=len(rows), streams=len({(r['media'], r['start']) for r in rows}),
                              model_path=model_path, config=asdict(config), fingerprint=fingerprint), indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty; v0 does not resume or overwrite predictions")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = time.perf_counter()
    results = []
    with (args.output / "predictions.jsonl").open("x", encoding="utf-8") as handle:
        for row in infer(job, rows, config):
            row.update(status="ok", fingerprint=fingerprint)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            results.append(row)
            print(f"finished {row['sample_id']} hit={row['hit']} history_tokens={row['history_visual_tokens']}", flush=True)
    from onestreamer.metrics import aggregate
    summary = aggregate(results)
    summary.update(wall_seconds_including_model_load=time.perf_counter() - started,
                   observation_seconds=sum(r["observation_seconds_delta"] for r in results),
                   query_seconds=sum(r["runtime_seconds"] for r in results),
                   state_probe_seconds=sum(r.get("state_probe_seconds", 0.0) for r in results),
                   action_probe_seconds=sum(r.get("action_probe_seconds", 0.0) for r in results),
                   action_probe_count=sum(len(r.get("action_rounds", [])) for r in results),
                   action_invalid_count=sum(r.get("action_invalid_count", 0) for r in results),
                   action_stop_reasons=dict(Counter(r["action_stop_reason"] for r in results
                                                   if r.get("action_stop_reason") is not None)),
                   total_measured_seconds=sum(r["total_runtime_seconds"] for r in results))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

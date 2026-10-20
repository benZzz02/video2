"""Causal, budget-driven historical token selection inspired by V-CAST.

Reference: https://github.com/xinyouu/V-CAST/tree/
f15d2764e2381e0344fea9eb6228a4a58654f4f7 (NeurIPS 2026).
The spatial score and temporal curvature follow the reference formulas. This
is a streaming adaptation, not the paper's offline fixed-retention algorithm:
selection happens only under pressure, on already observed retained frames,
with hard token/byte/capacity bounds and variable per-frame token counts.
Pruning is irreversible. Original frame means and surviving original scores
are retained; neither discarded feature tensors nor discarded records remain.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

from .visual_memory import MemoryConfig, VisualFrame, VisualMemoryBank


def temporal_weights(frame_means, temperature: float = 0.7):
    """V-CAST curvature softmax; means must describe original complete frames."""
    import torch
    import torch.nn.functional as F

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("V-CAST temperature must be finite and positive")
    if frame_means.ndim != 2 or not torch.isfinite(frame_means).all():
        raise ValueError("Frame means must be a finite [frames, channels] tensor")
    count = len(frame_means)
    if count == 0:
        return frame_means.new_empty(0, dtype=torch.float32)
    curvature = torch.ones(count, device=frame_means.device, dtype=torch.float32)
    if count > 2:
        incoming = frame_means[1:-1] - frame_means[:-2]
        outgoing = frame_means[2:] - frame_means[1:-1]
        curvature[1:-1] = 1 - F.cosine_similarity(incoming, outgoing, dim=-1, eps=1e-6)
    return torch.softmax(curvature / temperature, dim=0)


def allocate_tokens(weights, capacities, budget, floors, *, token_bytes=None, byte_budget=None):
    """Capped weighted integer allocation without increasing either budget.

    Floors must already fit. Unused shares of saturated frames are redistributed.
    A byte budget optionally constrains heterogeneous feature payloads. Ties are
    resolved chronologically, so equal-score runs are reproducible on CPU/GPU.
    """
    count = len(capacities)
    if len(weights) != count or len(floors) != count:
        raise ValueError("Allocation arrays must have the same length")
    if any(not math.isfinite(float(w)) or w < 0 for w in weights):
        raise ValueError("Allocation weights must be finite and nonnegative")
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 0
           for x in [budget, *capacities, *floors]):
        raise ValueError("Allocation counts must be nonnegative integers")
    if any(lo > hi for lo, hi in zip(floors, capacities)):
        raise ValueError("Allocation floors cannot exceed capacities")
    costs = [1] * count if token_bytes is None else list(token_bytes)
    if len(costs) != count or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in costs):
        raise ValueError("Per-token byte costs must be positive integers")
    limit = sum(c * n for c, n in zip(costs, capacities)) if byte_budget is None else byte_budget
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("Allocation byte budget must be a nonnegative integer")
    allocated = list(floors)
    remaining = budget - sum(allocated)
    bytes_left = limit - sum(c * n for c, n in zip(costs, allocated))
    if remaining < 0 or bytes_left < 0:
        raise ValueError("Allocation floors exceed a hard budget")
    while remaining:
        active = [i for i in range(count) if allocated[i] < capacities[i] and costs[i] <= bytes_left]
        if not active:
            break
        active_weights = [float(weights[i]) for i in active]
        total_weight = sum(active_weights)
        if total_weight == 0:
            active_weights = [1.0] * len(active)
            total_weight = float(len(active))
        unit_cost = sum(w * costs[i] for i, w in zip(active, active_weights)) / total_weight
        steps = min(remaining, max(1, int(bytes_left / unit_cost)))
        raw = [steps * w / total_weight for w in active_weights]
        before = remaining
        for i, share in zip(active, raw):
            take = min(int(share), capacities[i] - allocated[i], remaining, bytes_left // costs[i])
            allocated[i] += take
            remaining -= take
            bytes_left -= take * costs[i]
        # Largest remainders, with source chronology as the deterministic tie-break.
        order = sorted(range(len(active)), key=lambda j: (-(raw[j] - math.floor(raw[j])), -active_weights[j], active[j]))
        for j in order:
            i = active[j]
            if remaining and allocated[i] < capacities[i] and costs[i] <= bytes_left:
                allocated[i] += 1
                remaining -= 1
                bytes_left -= costs[i]
        if remaining == before:
            break
    return allocated


@dataclass
class _Record:
    frame: VisualFrame
    original_mean: Any
    scores: Any

    @property
    def metadata_bytes(self):
        return sum(x.numel() * x.element_size() for x in (self.original_mean, self.scores))

    def token_bytes(self, floor):
        # Reserve int64 coordinates only if present or if allocation can make
        # the frame sparse. A dense frame fixed at its full token count needs
        # no coordinates; charging them could falsely make feasible floors fail.
        frame = self.frame
        payload = sum(x.shape[1] * x.element_size() for x in (frame.features, *frame.deepstack))
        coordinates = 8 if frame.source_indices is not None or frame.token_count > floor else 0
        return payload + self.scores.element_size() + coordinates

    @property
    def mean_bytes(self):
        return self.original_mean.numel() * self.original_mean.element_size()


class VCastVisualMemoryBank:
    """Preserve original spatial evidence within the configured history budget.

    Native frames enter as independent CPU tensors. No selection is performed
    until storage pressure. Each reduction allocates tokens using curvature of
    original normalized frame means, then gathers the highest original spatial
    scores identically from the final features and all DeepStack branches.

    The floor is a preference: temporally redundant records are removed if the
    floors cannot fit. A final remaining record may shrink below the floor, down
    to one token; no minimum can override hard limits. ``stored_bytes`` includes
    feature tensors, sparse coordinates, surviving scores and original means,
    but excludes Python/container/allocator overhead, like the existing banks.
    """

    def __init__(self, config: MemoryConfig, merge_size: int = 2):
        if not isinstance(merge_size, int) or isinstance(merge_size, bool) or merge_size < 1:
            raise ValueError("Merge size must be a positive integer")
        self.config, self.merge_size = config, merge_size
        self.temperature = config.vcast_temperature
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("V-CAST temperature must be finite and positive")
        self.token_budget = config.store_tokens
        self.frame_overhead = 0
        self._records = []
        self._last_timestamp = -math.inf
        self._seen = self._dropped = self._oversize = self._operations = 0
        self._selection_input_tokens = self._selection_output_tokens = 0

    def set_budget(self, tokens: int, frame_overhead: int = 0):
        for name, value in (("tokens", tokens), ("frame_overhead", frame_overhead)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"History budget {name} must be a nonnegative integer")
        self.token_budget, self.frame_overhead = tokens, frame_overhead
        self._enforce()

    def _new_record(self, frame):
        import torch
        import torch.nn.functional as F

        if (len(frame.grid_thw) != 3 or frame.grid_thw[0] != 1
                or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in frame.grid_thw)
                or any(x % self.merge_size for x in frame.grid_thw[1:])):
            raise ValueError("V-CAST requires an original image grid divisible by merge size")
        expected = frame.grid_thw[1] * frame.grid_thw[2] // self.merge_size**2
        if frame.source_indices is not None or frame.pool_source_shape is not None or frame.pool_scale != 1:
            raise ValueError("V-CAST admission requires complete native features before pooling or selection")
        branches = (frame.features, *frame.deepstack)
        if any(x.ndim != 2 or x.shape[0] != expected or x.shape[1] < 1
               or not x.is_floating_point() or not torch.isfinite(x).all() for x in branches):
            raise ValueError("V-CAST requires finite floating-point native features and aligned DeepStack rows")
        if self._records and frame.features.shape[1] != self._records[0].original_mean.numel():
            raise ValueError("History frames must use the same feature dimension")
        with torch.inference_mode():
            full = frame.features.detach().float()
            mean = F.normalize(full.mean(dim=0), dim=0, eps=1e-6)
            cosine = F.cosine_similarity(full, mean.unsqueeze(0), dim=-1, eps=1e-6)
            norms = full.norm(dim=-1)
            scores = 1 - cosine + (norms - norms.min()) / (norms.max() - norms.min() + 1e-6)
            owned = [x.detach().to(device="cpu", copy=True) for x in branches]
            return _Record(replace(frame, features=owned[0], deepstack=tuple(owned[1:])),
                           mean.to(device="cpu", copy=True), scores.to(device="cpu", copy=True))

    def _drop_temporal_redundancy(self):
        times = [r.frame.timestamp for r in self._records]
        index = 0 if len(times) <= 2 else min(
            range(1, len(times) - 1), key=lambda i: (times[i + 1] - times[i - 1], i))
        self._records.pop(index)
        self._dropped += 1

    def _totals(self):
        tokens = sum(r.frame.token_count for r in self._records)
        features = sum(r.frame.nbytes for r in self._records)
        metadata = sum(r.metadata_bytes for r in self._records)
        return tokens, features, metadata

    def _over(self):
        tokens, features, metadata = self._totals()
        return (len(self._records) > self.config.capacity
                or tokens + self.frame_overhead * len(self._records) > self.token_budget
                or features + metadata > self.config.store_bytes)

    def _select(self, record, keep):
        import torch

        old = record.frame
        if keep == old.token_count:
            return
        if not 0 < keep < old.token_count:
            raise ValueError("Selection must retain between one and the current number of tokens")
        # Stable score sorting selects earlier source indices on exact ties.
        chosen = torch.argsort(record.scores, descending=True, stable=True)[:keep].sort().values
        source = old.source_indices if old.source_indices is not None else tuple(range(old.token_count))
        indices = tuple(source[i] for i in chosen.tolist())
        with torch.inference_mode():
            record.frame = replace(old, features=old.features.index_select(0, chosen),
                                   deepstack=tuple(x.index_select(0, chosen) for x in old.deepstack),
                                   source_indices=indices)
            record.scores = record.scores.index_select(0, chosen)
        self._operations += 1
        self._selection_input_tokens += old.token_count
        self._selection_output_tokens += keep

    def _enforce(self):
        import torch

        if not self._over():
            return
        while len(self._records) > self.config.capacity:
            self._drop_temporal_redundancy()
        while self._records:
            # Removing a record can resolve all pressure. In particular, do not
            # charge hypothetical sparse coordinates to a now-fitting dense bank.
            if not self._over():
                return
            capacities = [r.frame.token_count for r in self._records]
            floors = [min(self.config.min_frame_tokens, n) for n in capacities]
            token_limit = self.token_budget - self.frame_overhead * len(self._records)
            byte_limit = self.config.store_bytes - sum(r.mean_bytes for r in self._records)
            costs = [r.token_bytes(floor) for r, floor in zip(self._records, floors)]
            fits = lambda values: (sum(values) <= token_limit
                                   and sum(n * cost for n, cost in zip(values, costs)) <= byte_limit)
            if not fits(floors):
                if len(self._records) == 1:
                    # Lowering the last record's soft floor may now make an
                    # originally dense frame sparse, so recompute its cost.
                    costs = [self._records[0].token_bytes(1)]
                    if fits([1]):
                        floors = [1]
                    else:
                        self._drop_temporal_redundancy()
                        continue
                else:
                    self._drop_temporal_redundancy()
                    continue
            means = torch.stack([r.original_mean for r in self._records])
            weights = temporal_weights(means, self.temperature).tolist()
            allocated = allocate_tokens(weights, capacities, token_limit, floors,
                                        token_bytes=costs, byte_budget=byte_limit)
            for record, keep in zip(self._records, allocated):
                self._select(record, keep)
            break
        if self._over():
            raise RuntimeError("V-CAST selection violated a hard history budget")

    def observe(self, frame: VisualFrame, now: float | None = None):
        if (not math.isfinite(frame.timestamp) or frame.timestamp <= self._last_timestamp
                or (now is not None and (not math.isfinite(now) or frame.timestamp > now))):
            raise ValueError("Archived timestamps must increase and not exceed current time")
        if not self.config.enabled:
            self._last_timestamp = frame.timestamp
            self._seen += 1
            return
        record = self._new_record(frame)
        self._last_timestamp = frame.timestamp
        self._seen += 1
        was_empty = not self._records
        self._records.append(record)
        self._enforce()
        if was_empty and not self._records:
            self._oversize += 1

    def candidates(self, before: float):
        if not math.isfinite(before):
            raise ValueError("History cutoff must be finite")
        return [r.frame for r in self._records if r.frame.timestamp < before]

    read = VisualMemoryBank.read

    def stats(self):
        tokens, features, metadata = self._totals()
        counts = [r.frame.token_count for r in self._records]
        return dict(seen=self._seen, stored_frames=len(self._records), stored_tokens=tokens,
                    stored_bytes=features + metadata, stored_feature_bytes=features,
                    metadata_bytes=metadata, skipped_oversize=self._oversize,
                    dropped_frames=self._dropped, selection_operations=self._operations,
                    selection_input_tokens=self._selection_input_tokens,
                    selection_output_tokens=self._selection_output_tokens,
                    token_budget=self.token_budget, frame_overhead=self.frame_overhead,
                    budget_cost=tokens + self.frame_overhead * len(self._records),
                    earliest_timestamp=self._records[0].frame.timestamp if self._records else None,
                    frame_tokens_min=min(counts, default=0), frame_tokens_max=max(counts, default=0))

"""Sparse image placeholders with native Qwen3-VL spatial positions.

The image grid always describes the original image. Only selected placeholders
are passed to the decoder, and each selected token keeps its original mRoPE
coordinates. Transformers 4.57.6 discards supplied ``position_ids`` inside
``generate``; a scoped, per-engine ``get_rope_index`` override therefore supplies
the original coordinates on prefill and the corrected delta on later steps.
"""

from __future__ import annotations

from contextlib import contextmanager
import math


def validate_source_indices(frame, merge):
    """Return (original count, source indices), rejecting malformed sparse data."""
    grid = frame.grid_thw
    if (len(grid) != 3 or grid[0] != 1 or any(x <= 0 for x in grid)
            or grid[1] % merge or grid[2] % merge):
        raise ValueError("Sparse visual inputs require original complete image grids (t=1)")
    original_count = math.prod(grid) // merge**2
    source = getattr(frame, "source_indices", None)
    selected = tuple(range(original_count)) if source is None else tuple(source)
    if (not selected or any(not isinstance(i, int) or isinstance(i, bool) for i in selected)
            or selected[0] < 0 or selected[-1] >= original_count
            or any(a >= b for a, b in zip(selected, selected[1:]))):
        raise ValueError("Sparse source indices must be nonempty, ordered, unique original indices")
    if len(selected) != frame.features.shape[0]:
        raise ValueError("Sparse source indices do not match the cached feature count")
    return original_count, selected


def _layouts(engine, frames):
    """Validate original grids and return (original count, selected indices)."""
    merge = engine.model.config.vision_config.spatial_merge_size
    return [validate_source_indices(frame, merge) for frame in frames]


def _validate_inputs(engine, inputs, frames):
    import torch

    ids = inputs.get("input_ids")
    mask = inputs.get("attention_mask")
    if ids is None or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("Sparse visual inputs support one image-based prompt at a time")
    if mask is not None and (mask.ndim != 2 or mask.shape != ids.shape):
        raise ValueError("Sparse visual inputs require a two-dimensional attention mask")
    if inputs.get("video_grid_thw") is not None or inputs.get("pixel_values_videos") is not None:
        raise ValueError("Sparse visual inputs do not support native video placeholders")
    expected = torch.tensor([f.grid_thw for f in frames], dtype=torch.long)
    grid = inputs.get("image_grid_thw")
    if grid is None or not torch.equal(grid.detach().cpu(), expected):
        raise ValueError("Sparse visual input grids/order differ from cached frames")
    return ids, mask


def _blocks(engine, ids, mask, counts):
    """Find exact, separate image-placeholder runs, preserving all text."""
    image_id = engine.model.config.image_token_id
    start_id = engine.model.config.vision_start_token_id
    positions = (ids[0] == image_id).nonzero(as_tuple=True)[0].tolist()
    runs = []
    for index in positions:
        if not runs or index != runs[-1][1]:
            runs.append([index, index + 1])
        else:
            runs[-1][1] += 1
    if len(runs) != len(counts):
        raise ValueError("Sparse visual input image-block count differs from cached frames")
    for (start, end), count in zip(runs, counts):
        if end - start != count or start == 0 or int(ids[0, start - 1]) != start_id:
            raise ValueError("Sparse visual input placeholder block does not match its original grid")
        if mask is not None and not bool((mask[0, start:end] == 1).all()):
            raise ValueError("Visual placeholders cannot be masked or padded")
    return runs


def prepare_sparse_inputs(engine, inputs, frames):
    """Trim a full native processor result in place; keep original grid metadata.

    Dense frames pass through unchanged. This only removes image placeholders,
    never frame markers, timestamps, question text or recent-frame tokens.
    ``sparse_visual_positions`` must scope the corresponding model invocation.
    """
    import torch

    if not any(getattr(frame, "source_indices", None) is not None for frame in frames):
        return inputs
    layouts = _layouts(engine, frames)
    ids, mask = _validate_inputs(engine, inputs, frames)
    blocks = _blocks(engine, ids, mask, [count for count, _ in layouts])
    keep = torch.ones(ids.shape[-1], dtype=torch.bool, device=ids.device)
    for (start, end), (_, selected) in zip(blocks, layouts):
        keep[start:end] = False
        keep[torch.tensor(selected, device=ids.device) + start] = True
    indices = keep.nonzero(as_tuple=True)[0]
    for name in ("input_ids", "attention_mask", "token_type_ids"):
        if name in inputs:
            value = inputs[name]
            if value.ndim != 2 or value.shape != ids.shape:
                raise ValueError(f"Unexpected {name} shape for sparse visual input")
            inputs[name] = value.index_select(-1, indices.to(value.device))
    return inputs


def prepare_sparse_chat_inputs(engine, prompt, frames):
    """Expand atomic image markers directly, without allocating dummy pixels.

    Qwen3-VL's image marker is an atomic special token. The native processor's
    text-only branch therefore gives exactly the surrounding text tokens; each
    marker can be repeated to the retained feature length. No dummy image pixels
    are allocated; the later position-only reconstruction still uses the full
    original grid, without materializing full embeddings or a full KV cache.
    """
    import torch

    _layouts(engine, frames)
    inputs = engine.processor(text=[prompt], images=None, padding=True,
                              truncation=False, return_tensors="pt")
    ids = inputs["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("Sparse visual inputs support one image-based prompt at a time")
    counts = [1] * len(frames)
    blocks = _blocks(engine, ids, inputs.get("attention_mask"), counts)
    repeats = torch.ones(ids.shape[-1], dtype=torch.long, device=ids.device)
    for (start, _), frame in zip(blocks, frames):
        repeats[start] = frame.token_count
    for name in ("input_ids", "attention_mask", "token_type_ids"):
        if name in inputs:
            value = inputs[name]
            if value.ndim != 2 or value.shape != ids.shape:
                raise ValueError(f"Unexpected {name} shape for sparse visual input")
            inputs[name] = value.repeat_interleave(repeats.to(value.device), dim=-1)
    inputs["image_grid_thw"] = torch.tensor([f.grid_thw for f in frames], dtype=torch.long)
    return inputs


@contextmanager
def sparse_visual_positions(engine, frames):
    """Supply native source coordinates for one sparse prefill, then restore state.

    Call while holding the engine's visual-memory lock. Reconstructing the full
    placeholder layout is local to the prefill; no full-size embeddings or KV
    cache are allocated. Decode uses native HF logic with ``dense_delta +
    removed_tokens``, since the cache length counts only retained tokens.
    """
    import torch

    if not any(getattr(frame, "source_indices", None) is not None for frame in frames):
        yield
        return
    layouts = _layouts(engine, frames)
    core = engine.model.model
    native_get_rope_index = core.get_rope_index
    sentinel = object()
    previous_method = core.__dict__.get("get_rope_index", sentinel)
    previous_delta = core.rope_deltas

    def get_sparse_rope_index(input_ids=None, image_grid_thw=None,
                              video_grid_thw=None, attention_mask=None):
        ids, mask = _validate_inputs(engine, dict(
            input_ids=input_ids, image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw, attention_mask=attention_mask,
        ), frames)
        blocks = _blocks(engine, ids, mask, [len(selected) for _, selected in layouts])
        chunks, mask_chunks, keep = [], [], []
        old_offset = new_offset = 0
        for (start, end), (original_count, selected) in zip(blocks, layouts):
            text_count = start - old_offset
            chunks.append(ids[:, old_offset:start])
            keep.extend(range(new_offset, new_offset + text_count))
            new_offset += text_count
            chunks.append(ids.new_full((1, original_count), engine.model.config.image_token_id))
            keep.extend(new_offset + index for index in selected)
            if mask is not None:
                mask_chunks.extend([mask[:, old_offset:start], mask.new_ones((1, original_count))])
            new_offset += original_count
            old_offset = end
        chunks.append(ids[:, old_offset:])
        keep.extend(range(new_offset, new_offset + ids.shape[-1] - old_offset))
        if mask is not None:
            mask_chunks.append(mask[:, old_offset:])
        dense_ids = torch.cat(chunks, dim=-1)
        dense_mask = torch.cat(mask_chunks, dim=-1) if mask is not None else None
        original_positions, original_delta = native_get_rope_index(
            dense_ids, image_grid_thw, attention_mask=dense_mask,
        )
        retained = torch.tensor(keep, dtype=torch.long, device=original_positions.device)
        positions = original_positions.index_select(-1, retained)
        delta = original_delta + dense_ids.shape[-1] - ids.shape[-1]
        return positions, delta

    core.get_rope_index = get_sparse_rope_index
    try:
        yield
    finally:
        if previous_method is sentinel:
            delattr(core, "get_rope_index")
        else:
            core.get_rope_index = previous_method
        core.rope_deltas = previous_delta

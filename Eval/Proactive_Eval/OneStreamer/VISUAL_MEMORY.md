# Visual memory prototype

An independent, training-free inference entry point for the released OneStreamer
Qwen3-VL checkpoint. Production logic lives in
`src/onestreamer/visual_memory.py`; the official queue and PHCM runner are unchanged.

## What this version does

1. Observe each evaluated stream in timestamp order, starting at its `start`
   (normally zero), sampling history at 1 fps.
2. Encode sampled images with the checkpoint's native vision encoder. Preserve
   final image features **and every DeepStack branch**. Historical compression
   can use image resizing or direct feature pooling, as described below.
3. The `progressive_pool` mode ages all retained records through coarser spatial
   grids and drops the oldest if storage caps are exceeded. `resize`/`avg_pool`
   retain the seeded reservoir baseline. The default `fixed` read mode selects
   up to four older images within 1,024 visual tokens. The `fill` mode uses the
   remaining context for temporally spread historical images.
4. Combine these cached historical features with the selected recent-window protocol
   for one answer generation. PHCM and caption generation are never invoked.

CLI defaults: at most 16,384 stored visual tokens and 512 MiB of tensor payload
(including DeepStack). The frame cap defaults to 64 for `resize`/`avg_pool` and
to the configured token cap for `progressive_pool`; an explicit
`--history-capacity` overrides it. The Python `MemoryConfig` still defaults to
64 frames, so callers constructing it directly should set their own capacity.
The default `resize` mode uses at most
262,144 pixels per history image. Both pooling modes encode source-pixel frames
and uses the pooling stride instead of that historical image-size limit.
The first active cap limits storage. Variable image sizes can make budget
eviction nonuniform. The byte cap does not cover Python overhead, temporary
encoder tensors, recent frames, or total process/GPU memory.

This is a feasibility baseline: image resizing or dense spatial feature pooling,
with simple temporal selection. It does not yet implement learned compression,
query retrieval, or retention of every historical fact.
The algorithm has a separate backbone adapter; only Qwen3-VL is supported now.

## Run locally in the official inference environment

Requires the repository's inference dependencies and **transformers==4.57.6**.
Replace the paths below with existing checkpoint, sample and dataset paths.
From the repository root:

```bash
python Eval/Proactive_Eval/OneStreamer/src/onestreamer/visual_memory.py \
  --manifest /path/to/official/manifest.json \
  --model-path /path/to/OneStreamer-4B \
  --video-root /path/to/OVOBench \
  --output /path/to/new-visual-run \
  --subtasks EPM ASI --limit 20 --dry-run
```

Remove `--dry-run` to run inference. Alternatively use `--samples samples.jsonl`
with official sample dictionaries; both OVO-Bench and StreamingBench use this
schema. `--video-root` overrides existing sample `media` paths. Dry-run validates
sample fields and media file existence; it does not load the checkpoint or
decode videos. Each real run requires an empty output directory and writes
`manifest.json`, `predictions.jsonl`, and `summary.json`; there is no resume support.

For the controlled recent-only ablation, use the same command and a separate output
directory with **`--history-capacity 0`**. In `resize` mode this skips historical
encoding; in both pooling modes it skips pooling/archive writes while maintaining
the same full-feature recent stream. It shares the prototype's question prompt
with the visual-history run; the prompt is not
byte-identical to the official recent16 entry point. Use the official runner
separately when reporting the official recent16 baseline.

Keep the checkpoint, samples, recent frames, seed and answer settings fixed.
For a first feasibility test, compare this recent-only control against visual
history on the same EPM/ASI subset. This implementation alone establishes no
BT gain, no RT/FPD preservation, and no speedup over PHCM.

## Configure the recent window

`--recent-frames N` sets the number of full-feature recent observations in all
three modes. It accepts any positive integer and defaults to 16. For example:

```bash
--recent-frames 32 --history-compression progressive_pool \
--history-pool-age-seconds 60 --history-read-mode fill --context-tokens 32768
```

For a fixed configured N, the first N sampled frames remain uncompressed. The
(N+1)th arrival evicts the first frame into historical pooling. Each frame's age
clock begins at **its own actual eviction time**; increasing N postpones both
its initial compression and subsequent levels. No historical readout includes a
frame still in the recent FIFO. Before N frames have been observed, the recent
window contains only the available frames and history remains empty.

Keep `--recent-frames N` identical in the history and `--history-capacity 0`
control runs. Results record requested `recent_frames`, actual
`recent_read_frames`, and N in the variant; the manifest also records N. The
remaining-context calculation uses the actual recent features, not N times an
assumed cost. Increasing N can exceed context/GPU memory limits; the code does
not silently shrink the requested window. The 16-token answer reserve is an
independent setting and is not changed by `--recent-frames`.

## Pool features when they leave the recent window

To use direct feature compression with the remaining-context read mode, add:

```bash
--history-compression avg_pool --history-pool-stride 2 \
--history-read-mode fill --context-tokens 32768
```

This path implements:

```text
1-fps stream -> one full-resolution native encoding per sampled frame
            -> full-feature Recent-N FIFO
            -> eviction -> spatial average pooling -> CPU history bank
```

The first N sampled frames remain full fidelity. When the (N+1)th arrives, the
first frame is pooled once and archived. Questions reuse the FIFO's features;
neither recent readout nor historical compression calls the vision encoder
again. Frames are never pooled across time. Discarded historical evidence is not
reopened later.

Each final/DeepStack branch is reshaped into its merged spatial grid and receives
the same `avg_pool2d` operation. Stride 2 reduces an even-sized 16-by-16 token grid
to 8-by-8 (256 -> 64 tokens, 4x compression); stride 4 reduces it to 4-by-4
(16 tokens, 16x compression). Odd grid edges use partial windows, excluding
padding from the average. Ratios therefore depend on actual grid geometry.
Stride 1 provides an identity-feature control. Timestamp and original image
size are preserved; the readout carrier grid/size is updated to the pooled grid.

Pooled features are a **lossy approximation with a new coarse positional grid**.
They are not equivalent to running the encoder on a smaller image. Pooling
reduces stored payload and QA visual tokens; the full-resolution encoding cost
has already been incurred at stream arrival. It can also remove OCR, small-object
and fine-grained action evidence. Quality and total cost need measurement.

**Sampling control:** pooling mode uses fixed-origin 1-fps observations and the
last N of those observations. This differs from the official decoder's
question-end-aligned recent samples. Its matched recent-only control must retain
`--history-compression avg_pool` and add `--history-capacity 0`. Both modes are
explicitly identified in predictions and the manifest; do not attribute their
sampling-phase differences to memory. `--history-fps` must be 1 in pooling mode.
`--history-max-pixels` applies only to `resize` mode.

The history storage caps still apply after pooling. The full-feature FIFO resides
on the reader device and is additional to the CPU history payload cap. Predictions
report cumulative stream encodings and pre/post-pooling token totals (including
pooled frames subsequently rejected/evicted by bank budgets).

## Compress older history again as time advances

Use `progressive_pool` for repeated, age-based compression instead of one-time
pooling. For example, add these flags to the main command:

```bash
--history-compression progressive_pool --history-pool-stride 2 \
--history-pool-age-seconds 60 --history-read-mode fill --context-tokens 32768
```

The default 60-second interval is configurable video time, measured **since FIFO
eviction**, not since the old frame was captured. For an original 16-by-16 token
grid and stride 2:

| State | Visual tokens per frame |
| --- | ---: |
| Inside the configured recent window | 256 |
| Just evicted into history | 64 |
| 60 seconds after eviction | 16 |
| 120 seconds after eviction | 4 |
| 180 seconds after eviction and later | 1 |

Each retained timestamp remains a separate frame record. No frames are merged
across time. Frames are kept while budgets allow; **after processing due pooling,
the oldest records are dropped** if the frame, stored-token or CPU tensor-byte
cap is exceeded. A single newly pooled frame larger than the entire token/byte
cap is skipped without clearing existing valid history. The recent FIFO is never
dropped to make room for history. These caps are distinct from the model context
cap: readout may select a subset of the stored bank without deleting the rest.

This preserves all *sampled* historical timestamps while storage allows, not all
original source-video frames or all visual detail. Each frame stops shrinking
at one token per feature branch, so sufficiently long streams eventually require
dropping history. Increase storage caps when needed; the method has no lossless
or unbounded-history guarantee.

Repeated pooling uses original spatial-cell area weights. Without these weights,
partial edge blocks in odd grids would be overrepresented after another pooling
step. All final/DeepStack branches follow the same geometry, with float32
accumulation and conversion back to the feature dtype. No original full features
or source images are retained in the historical record. Repeated low-precision
rounding still introduces approximation.

A next-event heap schedules only frames that need another compression; saturated
one-token frames cause no further pooling. Long time jumps process all due levels,
and repeated query cutoffs do not duplicate work. Aging occurs during stream
advancement and at query cutoffs without accessing future frames. Dropped entries
are removed from the schedule lazily with bounded heap cleanup.

Use `--history-compression progressive_pool --history-capacity 0` for the matched
recent-only control. Sampling and feature reuse are identical to `avg_pool`,
including the difference from official end-aligned recent sampling. Logs report oldest
drop count, storage totals, compression operations, selected per-frame token
counts and cumulative pool scales. Pool token counters include every compression
step, not just the original and final sizes of currently retained records.

## Fill the remaining context with history

Add `--history-read-mode fill` to remove the fixed four-image/1,024-token read
limits. Its budget includes **all input tokens and the 16-token answer reserve**:

```text
effective cap = min(model max_position_embeddings, --context-tokens [default 32768])
history allowance = effective cap - actual recent features - question/prompt/formatting - answer reserve
```

The adapter counts the actual chat template, timestamp text, image boundary
tokens and expanded visual tokens. It tests whole historical image bundles in
a temporal-spread order, continuing past an oversized candidate to try smaller
ones. Before generation it validates the prepared input length again. Recent
frames are never silently dropped or resized to meet the cap: if the recent window plus
the question and answer reserve already exceeds it, the run reports an error.
Both read modes now enforce the context cap.

For example, add these flags to the command above to target a 32,768-token total
context with a larger CPU history bank:

```bash
--history-read-mode fill --context-tokens 32768 \
--history-capacity 256 --history-store-tokens 65536 --history-store-mib 2048
```

The default total context is **32K (32,768 tokens)**, including recent frames,
historical frames, prompt text and the answer reserve. This also applies to
`MemoryConfig()` and adapter calls without an explicit context cap. An explicit
`--context-tokens` can override it for controlled experiments; the model's own
limit still applies. The example bank caps are configurable storage limits,
not measured performance recommendations.

Filling is limited by **already retained evidence**, not all source-video frames.
The configured frame/token/byte storage caps may leave the bank smaller than the
available context; raise storage caps at the start of a run when testing larger contexts.
Frames already evicted are not recovered from the video after a question arrives.
Whole-image granularity may also leave unused tokens. This greedy procedure
does not claim globally optimal packing. For more than 128 candidates, a full-bank
check, individual feasibility checks and prefix search avoid repeatedly counting
every growing long prompt. At most 128 extra candidates are tried after the
prefix search, so this bounded heuristic can leave unused space; logs report
count calls and skipped tail candidates. The reader never recompresses features
to fit a question. Progressive aging is driven separately by video time.
`--history-max-pixels` controls image resizing in `resize` mode;
`--history-pool-stride` controls feature compression in both pooling modes.

Predictions include the effective cap, answer reserve and remaining context.
In fill mode, `history_selection_meta` also reports the recent-only input size,
eligible historical frames, selected input size and remaining input budget.
Use `--history-capacity 0` with the same `--recent-frames` for the recent-only control.

## Causality, caching and timing

- Samples sharing `(video, start)` are evaluated by increasing question time.
  The historical sampler keeps a fixed sampling phase between questions.
  It never decodes a sampled frame beyond the current cutoff. Clip starts follow
  the official floor-to-source-frame convention, including fractional starts.
- Historical evidence is encoded while advancing the stream, before reading the
  question. Evicted historical evidence is not reopened at question time.
  Readout excludes the entire recent window and sorts evidence chronologically.
- In `resize` mode, recent frames use the official decoder at each question to
  preserve its exact end-aligned sampling; this re-decodes/re-encodes recent frames.
  In both pooling modes, recent features come from the continuously maintained FIFO.
- Readout uses complete carrier grids, native token expansion, mRoPE and DeepStack. Only the
  instance's image-feature method is temporarily replaced, then restored. Cached
  features bypass ViT during answer prefill. Single-sample greedy generation
  with KV caching is required; beam search and video-token inputs are unsupported.
- Predictions report incremental/cumulative historical observation time,
  recent preparation/readout time, QA time, and total measured time. Here
  `runtime_seconds` measures recent preparation/readout plus QA. The `resize`
  branch includes recent decoding/encoding there, while both pooling modes have already
  paid for full-frame encoding in `observation_seconds_delta`. Compare total
  stream-plus-query cost with matching boundaries; query-only timing is not a
  speedup measurement. The summary also reports wall time including model load.

## Tests

With CPU PyTorch, OpenCV and the pinned Transformers installed:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s Eval/Proactive_Eval/OneStreamer/tests -p 'test_*.py' -v
```

Tests use synthetic video and/or a tiny randomly initialized Qwen3-VL model.
They do not download weights or establish released-checkpoint benchmark quality.

Local verification on 2026-10-09: **61 tests passed**, with CPU PyTorch 2.9.1 and
Transformers 4.57.6. Native pixels and cached features produced exactly equal
prefill logits, mRoPE state, generated tokens and per-step generation scores on
the tiny model. Tests also forbid ViT execution during cached readout, exercise
the native image processor/chat/token expansion, and check causal sampling,
budgets, official scoring and the zero-history control. Fill-mode tests verify
actual processor token counting, complete Recent16 preservation, output reserves,
variable-size historical frames, and overflow rejection before generation.
Pooling tests verify hand-computed spatial means in every branch, odd-grid
edges, independent reduced tensor storage, the 16/17-frame eviction boundary,
one encoding per sampled frame, repeated-cutoff reuse and matched recent-only
controls. Tiny-model stride-1 pooling preserves native logits/generation;
stride-2/4 pooled features pass native placeholder/grid/budget checks and are
readable without invoking ViT. This does not establish pooled-feature accuracy.
Progressive tests cover retirement-based aging, repeated timestamps, long jumps,
the one-token floor, all three caps, oldest-record drops, bounded event heaps,
weighted repeated pooling of odd grids, fractional sampling origins and large-bank
context filling. Tiny-model repeated pooling from 15 to 6 to 2 to 1 tokens remains
readable and agrees with original-area averages within floating-point tolerance.
Configurable recent-window integration tests cover N=1/4/8/16/32 across both
pooling modes, fractional stream starts, matched recent-only controls and the
legacy resize path. They verify first archival only at frame N+1, unchanged
recent features, repeated cutoffs, and aging since actual retirement. Real tiny
HF tests also read pooled history with N=1/3/17 recent frames and validate the
actual prompt budget. CLI rejects nonpositive recent counts.
A CPU/meta-device regression checks that DeepStack tensors are moved to the
reader device before concatenation (not a CUDA execution test). The processor fixture
uses a locally constructed tokenizer and bypasses initialization checks only
for the unused video component. Released-checkpoint CUDA/bfloat16 inference and
benchmark performance remain unverified.

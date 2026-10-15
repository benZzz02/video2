# Exploratory full OVO-Bench frame-ID prompt run

Started 2026-10-14 on 3,035 official OVO-Bench questions with the two-GPU
resumable local supervisor. The result directory is
`results/ovo-frame-id-witness-fill-2gpu-20261014/`; its `manifest.json`,
`status.json`, per-attempt logs, predictions, and eventual `summary.json` are
the source of truth. The launcher is `run_full_frame_id_witness_20261014.sh`.

This run tests the **frame-ID final-answer prompt**, not a learned or
automatically adaptive history reader. It uses the existing `fill` selection
over progressive pooled visual history plus the fixed-origin Recent16 window,
without caption memory or benchmark task-group labels in the model prompt.
The configuration uses a 131,072-token context cap and reserves 128 tokens for
the two-line answer/citation response. The prompt is byte-for-byte the v2
six-case pilot's system text and footer for MCQ and integer questions; the
869 short-answer questions use the same system text with a short-answer footer.
Each supplied image receives an adjacent H or R frame ID. A historical H ID
indexes the eligible stored history at that query, so a selected subset may
have gaps in its H numbers.

Benchmark scoring uses a valid `ANSWER:` line only. The raw response and a
separate citation audit are saved for every question. The audit checks syntax,
order, and whether an ID names a supplied frame; it cannot establish that the
pixels depict the queried entity or support the answer. A missing or malformed
`EVIDENCE:` line therefore does not erase a parseable answer. The earlier
six-case pilot used a stricter two-line format requirement for score extraction,
so its small answer counts are not directly the scoring protocol for this run.

The changed prompt and larger output reserve may alter which history frames
fit compared with the prior standard `fill` run. Compare the recorded
`history_timestamps`, `history_visual_tokens`, and input lengths alongside
BT/RT/FT accuracy. The baseline official Recent16 also differs in sampling
phase and prompt, so it is a reference rather than a matched causal control.

A three-question live GPU smoke covered MCQ, integer, and substring modes;
all completed with stored history and raw witness output. CPU checks passed:
23 prompt/runner tests and 48 visual-memory regressions. The full run is bound
to GPU UUIDs for indices 0 and 3. Existing training on indices 1 and 2 was
not stopped.

# Native-state readout protocol correction

This revision addresses the prompt protocol used by the experimental visual
read controller. It has not yet demonstrated improved real-checkpoint routing.

The previous `native_readiness_v2` asks the model not to answer and instead
classify readiness with a special token. In the repository's native QA
protocol, `</Response>` introduces an answer. Treating it as an isolated
classification label while prohibiting answers changes the task. The v2
full-run snapshot contains 921 Silence predictions over 241 completed
samples, and exactly the same final answers as the original visual fill run.
These observations identify a failure of the controller, but do not prove
the prompt is its sole cause.

The corrected `native_response_prefix_v3` permits an answer beginning with
`</Response>` when the evidence is sufficient. It places the active question
before the visual update, using separate user turns as in
`Eval/VLMEvalKit/vlmeval/vlm/qwen3_ovotiming/inference_fast.py`
(`_append_turn`). This is a prompt choice, not proof that ordering caused
the failure; the generic shared engine uses a different layout. It retains the complete timestamped visual
blocks, all cached visual features and DeepStack branches. Negative answers,
counts, static-scene questions and predictions are still valid answer tasks;
the conservative new-event-only timing prompt is not substituted for QA.

The program still reads exactly one next-token logit vector. It generates
no answer text or caption during the probe. It neither masks the vocabulary
to three states nor uses task labels, target answers or new model training.
Final answering and the memory writer are unchanged. Each probe now records
its prompt version. Existing v2 runs must not be resumed using v3 code.

Pinned Transformers 4.57.6 tiny-Qwen3-VL regressions found no forward-index
or stale-RoPE defect: raw logits, argmax and RoPE match one-token generation
exactly, including deliberately poisoned prior position state and changing
view lengths. Interleaving probes leaves final-answer tokens and per-step
logits identical. A synthetic generation suppression setting changes only
the processed generation choice, confirming the probe's unrestricted-logit
semantics. The restored real checkpoint's generation settings have no
suppressed or forced tokens. Its EOS IDs differ from all three native states.
All 86 core tests and 18 local-runner/diagnostic tests pass.

`tools/check_state_readout.py` compares legacy v2 and the corrected protocol
on the same selected rows with one model load and separate fresh visual
banks. It records raw states, stop reasons, final answers and timings without
overwriting prior results. It reports all-Silence or invalid-state outputs
as suspect; diverse states alone are not evidence of correct routing.

The NAS checkpoint/data path was restored on 2026-10-12. The same original
12 diagnostic rows completed on GPUs 0 and 3. Outputs are isolated under
`results/state-protocol-v3-ab-20261012/shard-{0,1}`. The old protocol produced
25 Silence probes; v3 produced 7 Response and 10 Silence probes with no
invalid states. Correct answers decreased from 7/12 to 6/12: BT 2/4 to 1/4,
RT remained 2/4 and FT remained 3/4. One historical question stopped at
recent-only evidence and lost a previously correct answer. This fixes the
observed all-Silence behavior on the diagnostic batch, but does not establish
correct evidence sufficiency or preservation of BT/FT gains. No prompt or
threshold was adjusted based on these answers. A separate full v3 run started at 2026-10-12 21:25 Asia/Shanghai under
the existing full-evaluation authorization: 3035 samples, GPUs 0/3, tmux
`onestreamer-state-v3-full`, output `results/ovo-state-memory-v3-2gpu-20261012`.
Do not change evaluation sources or this worktree HEAD while that run is
active; its manifest records the base commit and exact changed source hashes. Preserve
successful and failed diagnostics; do not tune on final test answers.

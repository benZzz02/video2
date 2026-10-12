# Training-free native-state visual readout

Implemented in `src/onestreamer/state_readout.py`, integrated through
`NativeFeatureAdapter` and `--history-read-mode state` in `visual_memory.py`.

The existing visual writer is unchanged: 1-fps recent FIFO, native feature
bundles including DeepStack, historical pooling after eviction. No caption or
textual description is generated or stored. The controller receives only the
question, visual evidence and resource limits; no task label, target answer,
subtask routing table, or trained router is supplied.

At each query it starts with the recent window. A separate cached-feature
prefill reads one unrestricted next-token prediction. `</Response>` stops;
`</Standby>` and `</Silence>` both permit the next history budget. Default
historical visual-token caps are 0 → 1024 → 4096 → 16384. Evidence is nested,
chronological and strictly older than the recent window. Temporal selection
uses existing stored bundles; it does not claim semantic frame retrieval.

Both the state prompt and unchanged final answer prompt must fit the actual
model context limit. The controller bounds rounds, skips unchanged views,
and stops on evidence/budget exhaustion. A non-state argmax is logged as
`invalid_state` and answers using the current view; it is never coerced into
a state. Full-vocabulary state probabilities are diagnostics, not calibrated
confidence estimates. State-probe runtime is included in total query time.

`native_readiness_v2` quotes the question to be assessed and puts the state
output instruction after it, so embedded option/Yes-No/count output
instructions do not occupy the final instruction position. Final QA still
uses the original prompt and answer reserve. The state probe does not pass
generated text into final QA.

Validation: 82 core unit/integration tests and 13 local runner tests passed.
The five state-adapter tests include real tiny Qwen3-VL CPU prefills, every
native state plus a non-state argmax, cache reuse without another ViT pass,
full-vocabulary probabilities, context limits and missing special tokens.
After v2 changed prompt construction, all five adapter tests passed again.

Real checkpoint smoke uses the `vllm` Python environment and GPU UUIDs for
physical GPU 0 and GPU 3, under tmux session `onestreamer-state-smoke`.
The same 12 original samples cover four BT, four RT and four FT examples;
they were selected by coverage/duration/input size, not correctness.

- First run: `results/state-readout-smoke-20261012/`
- Revised prompt run: `results/state-readout-smoke-v2-20261012/`
- Inputs: `results/state-readout-smoke-20261012-input.json`

Both real-checkpoint runs completed with exit code 0, and all 12 samples in
each passed the recorded nesting, causality, token-budget, no-caption and
runtime-accounting audit (`readout_audit.json` in each output directory).

The first run completed all 12 with valid cache/budget/causality invariants,
but six samples ended with an invalid state, and no Response state occurred.
In particular, direct-answer questions produced Yes/No/numbers. This prompted
the v2 formatting revision; the first results and source fingerprints remain
intact. This development smoke is not an accuracy/no-regression evaluation.
Do not claim that native readiness is calibrated for retrieval, or that RT is
protected, until the revised controller is evaluated on separate data and
compared with a matched recent-only control.

The v2 run fixed output-format compliance but not the decision semantics:
all 25 probes across the 12 samples emitted `</Silence>`. None emitted
`</Response>` or `</Standby>`. Four samples had no historical candidates;
the other eight exhausted their available historical evidence. Thus this
smoke exhibits a degenerate read policy, not successful selective reading.
State probes added 17.99 measured seconds within 27.59 total query seconds.
The implementation is available as an experimental opt-in; the previous
default `fill` mode remains the default. Following the user's request, the
3035-sample full evaluation was subsequently launched in tmux session
`onestreamer-state-full`, using the same v2 prompt, vllm Python environment
and physical GPU 0/3. Output is in
`results/ovo-state-memory-v2-2gpu-20261012/`. It was still running when this
code was published; the full-set behavior and accuracy remain unestablished.

The next diagnostic should test the checkpoint's native response-prefix
protocol (an answer begins with `</Response>`, but the probe still reads only
one token), and its original question-before-frames ordering. This is a
hypothesis for investigation, not a demonstrated fix. Merely constraining
decoding to the three states would hide compliance problems without proving
that the chosen state measures the value of historical evidence.

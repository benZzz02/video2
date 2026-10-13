# State v3 memory audit (2026-10-13)

This is an offline, read-only audit of 3,035 matched predictions. No inference was run. All numerical counts are sample counts, not independent-video counts.

## Verified facts

- Every v3 bank snapshot and recent timestamp list equals oldfill's. Oldfill reads all stored frames in every sample, so its historical timestamps reveal the full bank available to v3 independently of the gate.
- All logged historical/recent timestamps are strictly before the answer cutoff. v3 selected histories are subsets of the available bank. This checks logged causality, not a separate video re-decoding audit.
- 1,216 samples have eviction; 581 evict before any second age-based pooling. Of those 581: 154 retain 23 frames at 690 tokens/frame; 231 retain 32 at 510; 181 retain 41 at 391; 15 retain 13 at 1,200.
- The bank budget is 16,384 tokens, with a 60-second interval before re-pooling. At 1 fps, these four sizes hit capacity before their first second-stage pool. The progressive bank therefore behaves like a short FIFO in those samples.
- All 698 REC examples have zero eviction. Thus eviction cannot explain their errors; 427 are wrong in both runs, including 108 where v3 reads the complete bank (possibly empty). Temporal coverage also does not guarantee observing every short action at 1 fps.

| Subset | n | Eviction | Median available prefix span | Prefix span < 50% | No history read | All bank read |
|---|---:|---:|---:|---:|---:|---:|
| BT | 631 | 437 | 41.38% | 359 | 243 | 345 |
| BT_both_wrong | 146 | 108 | 40.17% | 87 | 61 | 73 |
| REC | 698 | 0 | 100.00% | 0 | 595 | 220 |
| REC_both_wrong | 427 | 0 | 100.00% | 0 | 347 | 108 |

Prefix span is `(end - earliest_available_timestamp) / (end - start)` using the entire bank plus Recent16, regardless of what v3 selects. It is not a ground-truth answer-evidence coverage metric.

## Selection and final answering

- `state_readout.py:39` builds an endpoint-first, progressively uniform temporal order. `:118` and `:134` select whole bundles by token budget. The question affects probing and token counts, but does not semantically rank candidate frames.
- `state_readout.py:181` stops on Response; both Standby and Silence request another budget. There are no Standby or invalid final states in this run.
- 1,043 examples end with Silence: 886 exhaust the bank and 157 have no bank. `visual_memory.py:983` still unconditionally calls the ordinary answer function, which uses the normal forced-answer prompt, not the state prompt. These are bounded best-effort answers, not a native Silence response.
- The 1,043 final-Silence examples have exactly the same final history as oldfill and 734/1,043 are correct in both runs. Therefore Silence is not a calibrated indicator that answering will be wrong, and these outcomes do not demonstrate a new v3 retrieval regression.

## Concrete cases

| ID | What the logs show | Limit on interpretation |
|---|---|---|
| ovo-0000 (EPM) | At cutoff 215 s, bank is only 176–198 s plus Recent16 at 199–214 s; 176 earlier frames evicted. All four probes return Silence; both runs answer D, gold C. | The target chopping/interaction time is not logged; cannot prove the gold evidence was among discarded frames. |
| ovo-0264 (EPM) | Asks about painting in the first bedroom. At 163 s, only 115–162 s remains; 115 frames evicted. Both runs wrong; all probes Silence. | Refers to a first room, but its actual visual occurrence was not manually verified. |
| ovo-2344 (REC) | Entire 0–13 s prefix present with no historical bank. Both runs predict 2; gold 1. | Demonstrates an error that memory selection cannot explain. Recognition/temporal sampling/counting remain hypotheses. |
| ovo-2346 (REC) | All 0–4 s observations present; no bank; probe Silence; both still output 1 while gold is 0. | Exposes the native Silence to forced answer transition, but not proof that abstention should score as zero. |
| ovo-2358 (REC) | 17 historical frames available, zero eviction. v3 says Response and selects none; both runs predict 1, gold 2. | Early stop omitted evidence but oldfill also failed with it, so this is not sufficient evidence that more readout alone fixes it. |
| ovo-2616 (REC) | 80 historical frames plus Recent16 cover 0–95 s. v3 reads the complete bank; both predict 2 javelin throws, gold 3. | Full 1-fps coverage still leaves action sampling, spatial pooling and temporal reasoning as possible causes. |
| ovo-3021 (REC) | 82 history frames plus Recent16 cover 0–97 s; no eviction; v3 reads all. Both predict 2 spikes, gold 9. | Same limitation: no proof that every spike was observable in retained 1-fps frames. |

## At most three training-free mechanisms to test

1. **Compress under capacity pressure before deleting old records.** Keep temporal coverage by further pooling older visual records when the token budget would overflow; preserve selected event neighborhoods or temporal anchors. This targets the verified eviction-before-aging cliff. It may erase detail, so compare equal storage-token budgets and inspect the earliest retained timestamp and recognition quality. No task label or caption is required.
2. **Retrieve question-relevant visual segments with explicit temporal coverage.** Use the frozen model to request evidence entities/events/time spans and select stored frame IDs or short adjacent segments, with an empty result allowed. Retain a coverage path for multi-event evidence instead of treating top similar isolated frames as complete. This targets query-blind temporal ordering. Its benefit is a hypothesis until tested, and nothing can retrieve already discarded observations.
3. **Separate evidence coverage from native willingness to answer.** Have the frozen model judge which observations the question requires and whether the selected view covers them, instead of using Response alone as sufficient evidence. Keep recent-only generation independent when comparing candidate corrections; do not feed a memory candidate answer into a verifier as a premise. Final Silence should remain visible in diagnostics even when the benchmark requires best-effort answering. This adds inference cost and cannot guarantee per-question nonregression.

## Code references

- [visual_memory.py:358](/home/zhangnuohua/medhorizon/video2-state-fix-20261012/Eval/Proactive_Eval/OneStreamer/src/onestreamer/visual_memory.py:358): age-based pooling.
- [visual_memory.py:401](/home/zhangnuohua/medhorizon/video2-state-fix-20261012/Eval/Proactive_Eval/OneStreamer/src/onestreamer/visual_memory.py:401): first re-pooling scheduled after 60 s.
- [visual_memory.py:406](/home/zhangnuohua/medhorizon/video2-state-fix-20261012/Eval/Proactive_Eval/OneStreamer/src/onestreamer/visual_memory.py:406): delete oldest immediately under capacity pressure.
- [visual_memory.py:625](/home/zhangnuohua/medhorizon/video2-state-fix-20261012/Eval/Proactive_Eval/OneStreamer/src/onestreamer/visual_memory.py:625): native state prompt differs from final answer prompt.
- [visual_memory.py:659](/home/zhangnuohua/medhorizon/video2-state-fix-20261012/Eval/Proactive_Eval/OneStreamer/src/onestreamer/visual_memory.py:659): full-vocabulary state probe.
- [visual_memory.py:983](/home/zhangnuohua/medhorizon/video2-state-fix-20261012/Eval/Proactive_Eval/OneStreamer/src/onestreamer/visual_memory.py:983): unconditional final answer.
- [state_readout.py:39](/home/zhangnuohua/medhorizon/video2-state-fix-20261012/Eval/Proactive_Eval/OneStreamer/src/onestreamer/state_readout.py:39): query-blind temporal ordering.
- [state_readout.py:129](/home/zhangnuohua/medhorizon/video2-state-fix-20261012/Eval/Proactive_Eval/OneStreamer/src/onestreamer/state_readout.py:129): nested budget selection.
- [state_readout.py:181](/home/zhangnuohua/medhorizon/video2-state-fix-20261012/Eval/Proactive_Eval/OneStreamer/src/onestreamer/state_readout.py:181): state stopping and history exhaustion.

Both reviewed code files match the run manifest SHA-256. The JSON stores per-sample derived fields and exact case metadata; rerun `memory_audit.py` to regenerate this audit.

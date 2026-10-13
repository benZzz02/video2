# State v3 error-analysis snapshot

This directory contains four CPU-only analysis scripts and reports for the
completed 3035-sample native-state v3 evaluation. It does not implement the
proposed question-conditioned history-budget router. See `REPORT.md` for the
current scope: change how much history is read, keeping the memory writer fixed.
The separate memory-retention audit is background evidence, not the next change.

The scripts were executed against locally saved v3, visual-fill and official
Recent16 predictions. They validate paired inputs and record source hashes in
the generated JSON. They do not run inference, train a model, or create captions.

Run on the evaluation server from this directory:

```bash
/home/zhangnuohua/miniconda3/envs/vllm/bin/python paired_analysis.py
/home/zhangnuohua/miniconda3/envs/vllm/bin/python state_signal.py
/home/zhangnuohua/miniconda3/envs/vllm/bin/python memory_audit.py
/home/zhangnuohua/miniconda3/envs/vllm/bin/python case_analysis.py
```

The required predictions are local artifacts, not included in this commit:

- This repository's `results/ovo-state-memory-v3-2gpu-20261012`.
- The sibling original `video2/results/ovo-visual-memory-2gpu-20261011` checkout.
- For `case_analysis.py`, the official Recent16 predictions under the NAS
  `OneStreamer-OVO-eval/outputs` tree; its exact path is defined in the script.

`paired_analysis.py --help` lists explicit path overrides. The other scripts
use the recorded server layout via constants near the top of each file; adjust
those paths when reproducing elsewhere. Their detailed JSON/CSV outputs and
logs remain local. The Markdown reports preserve findings and caveats.

# Experiment: exp-001-20260827

## Goal

Start a real Qwen3.5-9B masked SFT run using teacher-B corrected data as the supervised target lane.

## Scope of this first run

This first run is a bounded stage-0 validation run, not a converged full training claim.

- Base model: `/media/home/johnson/llm/models/Qwen3.5-9B`
- Training script: `/media/home/johnson/llm/scripts/qwen35-9b/fine-tune-9b.py`
- Data source lane: `teacher-B`
- Input corpus coverage: first 2500 rows of `research/ai-infra-expert/corpus/train.jsonl`, independently reviewed in `experiments/2026-08-17-teacher-b-corpus-review/`
- Native SFT train split: `2400` examples
- Native SFT heldout split: `100` examples
- Split rule: deterministic `sha256(source_id)` ranking, smallest 100 to heldout
- Train SHA-256: `d6a6727a63921aa78f8efc1f0fd1a1cd4ea1d8eb37cdd44f91afcca2d96f02ca`
- Heldout SHA-256: `6c0d75eb9545e725cd44416076b2e9fa38f8cabf19cb30cff6959c74e701f813`

## Status caveats

- teacher-B currently covers `2500 / 5399` train records and `0 / 601` validation records from the original corpus.
- Therefore this run is a teacher-B-prefix training run, not a full-corpus teacher-B training run.
- Success of this run proves pipeline viability only: data conversion, tokenizer rendering, FSDP load, forward/backward/optimizer step, checkpoint save, and bounded heldout loss.
- It does not by itself prove tool-call or agent capability improvement.

## Planned stage-0 command

See `run_stage0_validation.sh`.

## Actual stage-0 result (2026-08-27)

- Training process exited with code `0`
- Log markers observed:
  - `MASKED_SFT_START world_size=8 examples=2400 max_steps=1 seq_len=768 grad_accum=4 lr=5e-06`
  - `step=1 masked_loss=2.484217 target_tokens_per_rank=2276 elapsed=29.1s`
  - `CHECKPOINT_SAVED step=1 path=/media/home/johnson/llm/checkpoints/exp-001-20260827/stage0-validation/step-1`
  - `MASKED_EVAL_PASS examples=16 loss=2.367023`
  - `MASKED_SFT_DONE`
- Post-run GPU check: all 8 GPUs released back to `0 MiB`
- Checkpoint files observed:
  - `step-1/.metadata`
  - `step-1/__0_0.distcp` ... `step-1/__7_0.distcp`
- Structural reload audit:
  - initial `check-fsdp-load.py` had a nested state-dict namespace bug and failed
  - script was patched in-place to remove the extra nested `model` wrapper
  - rerun succeeded with exit code `0`

## Working tree note

This experiment created a new uncommitted directory in the repo:
- `experiments/exp-001-20260827/`

## Stage-1 training launch plan

- Objective: first non-trivial teacher-B training run under the same experiment id `exp-001-20260827`
- Train dataset: `data/teacher_b_native_train.jsonl` (`2400` examples)
- Heldout dataset: `data/teacher_b_native_heldout.jsonl` (`100` examples)
- Base model: `/media/home/johnson/llm/models/Qwen3.5-9B`
- Launcher: `env -u CONDA_EXE -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u CONDA_PYTHON_EXE -u CONDA_PROMPT_MODIFIER run_stage1_train.sh`
- Checkpoint output: `/media/home/johnson/llm/checkpoints/exp-001-20260827/stage1-train`
- Settings: `max_steps=75`, `save_steps=25`, `seq_len=768`, `grad_accum=4`, `lr=5e-6`, `eval_examples=100`
- Interpretation: this is approximately one epoch over the 2400-example teacher-B native train split, not a converged final model claim.

## Actual stage-1 result (2026-08-27)

- Stage-1 training completed with process exit code `0`.
- Final log markers observed: `MASKED_SFT_START`, `step=1`, `step=10`, `step=20`, `step=30`, `step=40`, `step=50`, `step=60`, `step=70`, `step=75`, `CHECKPOINT_SAVED step=25`, `CHECKPOINT_SAVED step=50`, `CHECKPOINT_SAVED step=75`, `MASKED_EVAL_PASS`, `MASKED_SFT_DONE`.
- Final heldout result observed: `MASKED_EVAL_PASS examples=100 loss=1.841137`.
- Final checkpoint evidence observed on disk: `stage1-train/step-25/`, `stage1-train/step-50/`, and `stage1-train/step-75/` each contain 8 shard files; `step-75/.metadata` exists.
- Post-run GPU check: all 8 GPUs returned to `0 MiB` with no remaining compute processes.
- Checkpoint load validation: `check-fsdp-load.py --checkpoint .../stage1-train/step-75` completed with exit code `0` and emitted `AFTER [...]`.
- Runtime note: the training log contains `4` allocator OOM warnings on some ranks, but the run continued and finished normally; this is a memory-pressure signal, not a failed run.
- Log hygiene note: `train_stage1.log` does not contain the earlier Conda `ERROR REPORT` header.

## Independent generative evaluation on step-75 (2026-08-27)

- Purpose: obtain a fresh, independent post-training read on `step-75` without relying on training loss alone.
- Evaluated model artifact: `/media/home/johnson/llm/models/exports/exp-001-20260827/step75-hf`
- Benchmark source: `research/ai-infra-expert/benchmark.jsonl`
- This phase used a deterministic stratified subset of `100` cases (`10` categories × `10` cases each), not the full `500`-case benchmark.
- Sampling rule: within each category block, select positions `[1, 6, 11, 16, 21, 26, 31, 36, 41, 46]`.
- Baseline comparator: `experiments/2026-08-13-domain-base-baseline/results/generations.jsonl` filtered to the same `100` ids.
- Generation protocol for the step-75 model: local vLLM service on `127.0.0.1:8002`, `temperature=0`, `max_tokens=768`, OpenAI-compatible `/v1/chat/completions`.
- Generation execution result: `100 / 100` cases completed successfully and were written to `eval-step75/results/generations_step75_subset_100.jsonl`.
- Diagnostic evaluator outputs:
  - step-75: `eval-step75/results/diagnostic_step75_subset_100.json`
  - base subset: `eval-step75/results/diagnostic_base_subset_100.json`
  - comparison summary: `eval-step75/results/comparison_summary_subset_100.{json,md}`
- Aggregate comparison on this 100-case subset:
  - mean key-point coverage: `0.3160` (step-75) vs `0.3339` (base), delta `-0.0179`
  - numeric mean match fraction: `0.8000` (step-75) vs `0.8125` (base), delta `-0.0125`
  - numeric all-expected-matched: `1` vs `1`
  - mean response chars: `3008.3` vs `3049.9`
  - mean latency ms: `5243.4` vs `5343.3`
  - code-fence responses: `0` vs `13`
- Interpretation:
  - On this stratified subset, `step-75` does not outperform the existing base baseline on the evaluator's heuristic key-point / numeric aggregates.
  - `step-75` is cleaner in formatting on this slice (no fenced-code responses) and slightly faster, but that is not sufficient evidence of improved domain capability.
  - Current evidence therefore supports pipeline viability and a completed short teacher-B-prefix training run, not benchmark-proven capability gain.
- Methodological limits:
  - this was a `100`-case stratified subset, not the full `500`-case benchmark
  - key-point coverage is lexical/outline-based and not a blinded human rubric
  - numeric matching is heuristic and not a validated pass/fail measure
  - generated code in benchmark answers was not executed in per-case sandboxes

## No-thinking rerun on the same subset (2026-08-27)

- Trigger: the first step-75 subset run showed many answers starting with `Thinking Process:` while the base comparator had explicit `chat_template_kwargs.enable_thinking=false`, so the original comparison mixed two different generation modes.
- Runner fix: `eval-step75/scripts/generate_domain_answers.py` was patched to send `chat_template_kwargs: {"enable_thinking": false}` for both readiness probes and benchmark generations.
- Targeted ad-hoc verification: a fresh `/tmp/hermes-verify-*.py` stub-server script verified that the runner now emits `enable_thinking=false` on both requests, then the temp script was removed.
- Rerun artifact files:
  - `eval-step75/results/generations_step75_subset_100_nothinking.jsonl`
  - `eval-step75/results/diagnostic_step75_subset_100_nothinking.json`
  - `eval-step75/results/comparison_summary_subset_100_nothinking.{json,md}`
  - `eval-step75/results/top10_regressions_detail_nothinking.md`
- Rerun execution result: `100 / 100` cases generated successfully.
- Rerun sanity check: `Thinking Process` / `Here's a thinking process` prefixed responses dropped from `100 / 100` in the prior step-75 run's top-regression slice to `0 / 100` across the full rerun file.
- Aggregate comparison on the same 100-case subset after forcing no-thinking:
  - mean key-point coverage: `0.3357` (step-75 no-thinking) vs `0.3339` (base), delta `+0.0018`
  - numeric mean match fraction: `0.8000` (step-75 no-thinking) vs `0.8125` (base), delta `-0.0125`
  - numeric all-expected-matched: `1` vs `1`
  - mean response chars: `3053.7` vs `3049.9`
  - mean latency ms: `5234.3` vs `5343.3`
  - code-fence responses: `14` vs `13`
- Interpretation update:
  - the earlier apparent regression on heuristic key-point coverage was largely a generation-mode confounder from mismatched thinking settings
  - once the request mode is aligned, step-75 no-thinking is approximately at parity with the base subset and slightly above base on mean key-point coverage
  - this still does not constitute benchmark-proven domain improvement, because the evaluation remains a 100-case heuristic diagnostic subset rather than a full audited benchmark

## Aligned 4096-token base versus step-75 comparison (2026-08-28)

The prior base artifact used a 768-token response cap and could not be used as a
like-for-like comparison against the step-75 4096-token rerun. The base model was
therefore regenerated on the identical fixed 100-case subset.

- Shared generation configuration: `max_model_len=8192`, `max_tokens=4096`,
  `temperature=0`, `chat_template_kwargs.enable_thinking=false`, the same fixed
  system prompt, bfloat16, vLLM `0.26.1rc1.dev659+g23f25aac1`, and 8-way tensor
  parallelism on the same 8-GPU host.
- Both arms completed `100/100` generations with non-empty answers and
  `finish_reason=stop` for every case. The v0.2 generation-validity gate passed
  for both arms.
- Artifacts: base raw generations are in
  `eval-step75/results/generations_base_subset_100_max4096.jsonl`; base layered
  diagnostics are in
  `eval-step75/results/diagnostic_base_subset_100_max4096_layered_v02.json`.
  The paired step-75 artifact remains
  `generations_step75_subset_100_max4096.jsonl` with diagnostics in
  `diagnostic_step75_subset_100_max4096_layered_v02.json`.

Automated diagnostic comparison only:

- Across 81 cases with the lexical key-point diagnostic, step-75 had 34 higher,
  24 equal, and 23 lower scores than base; mean paired delta was `+0.0183` and
  median delta was `0.0`.
- The strongest positive category means were Performance Analysis (`+0.0547`),
  Concept Understanding (`+0.0492`), and Troubleshooting (`+0.0356`). System
  Design (`-0.0317`) and Long-form Technical Analysis (`-0.0076`) were negative;
  Architecture Comparison was near neutral (`-0.0028`).
- On the 10 numeric cases, heuristic all-expected-number matches changed from
  `0/10` for base to `2/10` for step-75; mean heuristic match fraction changed
  from `0.8000` to `0.8250`.

Interpretation boundary: these results are a positive but non-uniform automatic
signal under a fully aligned generation configuration. Lexical key-point overlap
and extracted-number matching remain diagnostics, not capability scores. The
next evidence gate is an LLM-as-a-judge paired assessment with randomized answer
order, followed by executable fixtures for the 10 code cases.

## Regression-repair probe (2026-08-28, contaminated diagnostic only)

A repair-mix probe trained from the base model for 75 steps at `lr=2e-6` on the
original 2,400 Teacher-B examples plus 80 interleaved authored repair examples.
The repair records are explicitly labelled `authored_regression_repair_v0.1` and
`needs_domain_expert_review`; 77 of 80 fall in the 75-step consumption window.
The probe checkpoint was saved at steps 25, 50, and 75, reload-checked, and
exported as `regression-repair-v0.1-step75-hf`.

Critical validity restriction: the repair examples were authored after examining
six poor cases from `benchmark_subset_100.jsonl` and deliberately target their
topic families and reference control points. The same 100-case benchmark is
therefore contaminated for this probe. The following outcomes are diagnostic
only and must not be used as generalization or capability evidence.

- Training completed successfully; heldout loss on the unchanged Teacher-B split
  was `2.379644`.
- The identical 100-case generation protocol completed `100/100` non-empty
  responses with `finish_reason=stop`, but its result is contaminated as above.
- The six targeted cases each improved against the original step-75 lexical
  diagnostic, by `+0.0667` to `+0.1875`; five returned to at least the base
  diagnostic value and `aiinfra-0196` exceeded it by `+0.0625`.
- On the whole contaminated subset, repair versus original step-75 was negative
  on the lexical diagnostic (`25` higher, `19` equal, `37` lower; mean delta
  `-0.0115`). Numeric heuristic matching declined from `2/10` to `1/10`.

Required next step: construct a newly authored, hash-pinned private holdout with
unseen topics and task forms before evaluating this repair probe or any successor.
Do not tune further on the current benchmark subset.

# Experiment: exp-002-20260828

## Goal

Run an isolated post-training experiment that adds the regression-repair corpus
to the Teacher-B native SFT lane.

## Data and provenance

- Base lane: the 2,400-example Teacher-B native train split frozen in
  `experiments/exp-001-20260827/data/teacher_b_native_train.jsonl`.
- Repair lane: 80 records from
  `experiments/exp-001-20260827/data/regression_repair_v0.1.jsonl`.
- Repair provenance: `authored_regression_repair_v0.1`.
- Repair review state: `needs_domain_expert_review`.
- Mixed post-training corpus: 2,480 records, with one repair record inserted
  after every 30 base records. This makes 77 repair records fall within the
  first 2,400 examples consumed by the 75-step training schedule.
- Heldout monitoring set: the unchanged 100-example Teacher-B heldout split
  from exp-001. It is not a capability evaluation set.

## Contamination boundary

The repair records were authored after inspecting benchmark regression topics.
They must not be evaluated on `benchmark_subset_100.jsonl` or the public
benchmark that informed their design. A newly authored, hash-pinned private
holdout is required before making any capability or generalization claim.

## Training protocol

- Base model: `/media/home/johnson/llm/models/Qwen3.5-9B`
- Training: native masked SFT, FSDP over 8 GPUs
- Steps: 75; checkpoints: 25, 50, and 75
- Sequence length: 768; gradient accumulation: 4; learning rate: `2e-6`
- Output: `/media/home/johnson/llm/checkpoints/exp-002-20260828/repair-mix-v0.1`

## Evidence gates

1. The corpus builder must verify counts, IDs, provenance labels, and the
   75-step consumption window before training.
2. Training must complete, save all scheduled checkpoints, and report heldout
   masked loss.
3. Step-75 must pass an FSDP reload audit before any export or evaluation.
4. A clean private holdout is required for all capability conclusions.

## Actual stage-1 result (2026-08-28)

- Training completed with process exit code `0`.
- Final log markers observed: `MASKED_SFT_START`, `step=1`, `step=10`, `step=20`, `step=30`, `step=40`, `step=50`, `step=60`, `step=70`, `step=75`, `CHECKPOINT_SAVED step=25`, `CHECKPOINT_SAVED step=50`, `CHECKPOINT_SAVED step=75`, `MASKED_EVAL_PASS`, `MASKED_SFT_DONE`.
- Final heldout monitoring result observed: `MASKED_EVAL_PASS examples=100 loss=2.379644`.
- Final checkpoint evidence observed on disk: `step-25/`, `step-50/`, and `step-75/` each contain 8 shard files; `step-75/.metadata` exists.
- Post-run GPU check: all 8 GPUs returned to `0 MiB` with no remaining compute processes.
- Checkpoint reload audit: `torchrun --standalone --nnodes=1 --nproc_per_node=8 /media/home/johnson/llm/scripts/qwen35-9b/check-fsdp-load.py --model /media/home/johnson/llm/models/Qwen3.5-9B --checkpoint /media/home/johnson/llm/checkpoints/exp-002-20260828/repair-mix-v0.1/step-75` completed with exit code `0` and emitted `AFTER [...]`.
- Runtime note: the training log contains `5` allocator OOM warnings on some ranks, but no `ERROR REPORT` marker; the run continued and finished normally.
- Scope note: because the added repair records were authored after inspecting benchmark regression topics and remain `needs_domain_expert_review`, this run is evidence for pipeline execution and targeted-data integration only, not a clean capability-generalization claim.
- Submission boundary: this experiment is recorded under `experiments/exp-002-20260828/` and should be committed separately from `exp-001-20260827`.
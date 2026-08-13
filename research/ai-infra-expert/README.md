# AI/LLM Infrastructure Domain-Expert Post-Training Assets

This directory contains the first usable research package for adapting Qwen3.5-9B toward AI/LLM Infrastructure expertise.

## Assets

- `corpus/train.jsonl`: 5,399 authored seed SFT records.
- `corpus/validation.jsonl`: 601 authored seed validation records.
- `corpus/manifest.json`: counts, provenance and split policy.
- `benchmark.jsonl`: 500-record domain capability benchmark from the previous phase; it is held out from training.
- `eval_model_domain.json`: first evaluation standard, intrinsic model/domain capability.
- `eval_runtime_system.json`: second evaluation standard, deployment/runtime capability.
- `build_corpus.py`: deterministic corpus generator.
- `validate_assets.py`: schema, split and contamination-boundary checks.
- `MANIFEST.sha256`: artifact hashes.
- `TRAINING-GOAL.md`: Qwen3.5-9B training objective, methods, task scope, capability levels and evidence gates.

## Important status

The corpus is an authored synthetic seed corpus (`authored_synthetic_seed_v0.3`). It is not a bulk scrape of external documents and has not yet passed expert review. Every record is marked `needs_domain_expert_review`. It is suitable for pipeline validation and controlled pilot experiments, not yet for the final claim that the model is an expert. See `TRAINING-GOAL.md` for the complete objective and training-method plan.

Before a serious CPT/SFT run:

1. Expert-review the factual and formula answers.
2. Attach source evidence to factual records.
3. Add private holdout questions and adversarial variants.
4. Run contamination checks against all training sources.
5. Freeze the benchmark hash and exclude it from all training material.
6. Decide whether to retain the synthetic records, replace them with sourced records, or use them only for pipeline smoke tests.

## Corpus design

The seed corpus contains 6,000 records:

- 500 Knowledge/Concept explanations;
- 500 Calculation records with exact formulas and unit-aware answers;
- 3,000 System Design, Performance Analysis and Troubleshooting records;
- 1,000 Code/Tool-use records with unit-test contracts;
- 1,000 advanced cluster and distributed-inference records covering RDMA, RoCE, GDR, GDS, Mooncake, NVIDIA Dynamo, multi-node NCCL, scheduling and recovery.

The advanced cluster records carry auditable metadata for `domain_subtopic`,
`cluster_scope`, `parallelism_scope`, `network_scope`, `failure_mode`, and
`technology`. They are intended to test cluster design, performance analysis,
troubleshooting, and tool-use reasoning rather than to merge runtime metrics
into model-domain capability labels.

The split is deterministic by SHA-256 of record ID:

- 90% train;
- 10% validation.

The domain benchmark is not part of either split. Do not concatenate it into training data.

## The two evaluation standards

### 1. Model Domain Capability

Measures whether the model learned domain knowledge and problem-solving ability:

- knowledge and concept accuracy;
- calculation pass rate;
- code unit-test pass rate;
- system-design and troubleshooting rubric score;
- unseen-parameter generalization;
- hallucination/error rate;
- answer and reasoning tokens.

It does not use throughput, GPU memory or latency as evidence of domain learning.

### 2. Runtime/System Capability

Measures whether a fixed model artifact can be deployed reliably and efficiently:

- TTFT, TPOT and end-to-end latency;
- throughput and concurrency capacity;
- GPU memory peak/headroom;
- OOM and error rate;
- tool-call validity in the serving path;
- recovery time and SLO success;
- GPU-hours per output token.

It must be reported separately and cannot be used to claim domain expertise.

## Reproducibility

Run:

```bash
python build_corpus.py
python validate_assets.py
sha256sum -c MANIFEST.sha256
```

The initial commit intentionally excludes model weights, checkpoints, caches, credentials and large historical logs.

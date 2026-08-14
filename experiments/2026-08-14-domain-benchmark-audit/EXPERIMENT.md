# Benchmark audit scaffold: v0.2 preparation

## Status

`AUDIT_SCAFFOLD_ONLY`. This is not an expert-audited gold benchmark and does not produce an authoritative model score.

## Objective

Prepare versioned, machine-readable assets for domain-expert review before authoritative scoring or post-training. No benchmark answer was silently promoted to gold and no expert judgment was invented.

## Frozen inputs

- Benchmark: `research/ai-infra-expert/benchmark.jsonl`
- Benchmark SHA-256: `30eb88766f916dd487dc2921b9479e238c1074a96d20611f099068af917cc869`
- Base generations: `experiments/2026-08-13-domain-base-baseline/results/generations.jsonl`
- Generations SHA-256: `508735d8453ef17f30840948e7c3dbe952d1b9754ef4954290c70c735bd69d42`
- Preparation script SHA-256 before execution: `0850b4d1c5223b03e1c04eae080783854e9c5b1e4f230cd6bfe42cb9768ca037`

## Command

```bash
/media/home/johnson/llm/qwen35-env/bin/python -m py_compile \
  experiments/2026-08-14-domain-benchmark-audit/prepare_audit_assets.py

/media/home/johnson/llm/qwen35-env/bin/python \
  experiments/2026-08-14-domain-benchmark-audit/prepare_audit_assets.py \
  --benchmark research/ai-infra-expert/benchmark.jsonl \
  --generations experiments/2026-08-13-domain-base-baseline/results/generations.jsonl \
  --output-dir experiments/2026-08-14-domain-benchmark-audit/results
```

## Generated assets

- `results/audit-queue.jsonl`: 500 records, all `audit_status=unreviewed`, evidence and contamination review pending.
- `results/calibration-sample.jsonl`: 20 records, exactly two per category, paired with frozen Base responses, reviewer fields empty.
- `results/numeric-metadata-draft.jsonl`: 50 calculation records with draft numeric candidates and 1% tolerance; all require expert verification of units and assumptions.
- `results/code-fixture-plan.jsonl`: 50 code records with draft contracts and sandbox limits; no fixture is implemented or executed.
- `results/rubric-schema.json`: draft five-dimension 1–4 rubric and two-reviewer calibration protocol.
- `results/audit-summary.json`: deterministic count and status summary.

## Actual counts

```text
records: 500
calibration cases: 20
numeric metadata drafts: 50
code fixture plans: 50
categories: 10 x 50
all_records_require_expert_review: true
authoritative_score_available: false
```

Current source quality remains constrained:

```text
provenance_status: curated_template_v0.1 for all 500 records
difficulty: easy=40, medium=220, hard=240
```

## Expert review gates

Before freezing a benchmark v0.2 scoreable release:

1. Bind factual claims to authoritative documentation or architecture papers.
2. Verify formulas, units, assumptions, and acceptable numerical variants.
3. Author network-disabled, resource-limited code fixtures and tests.
4. Review template similarity and rewrite overly templated scenarios.
5. Score the 20-case calibration sample independently with two domain reviewers.
6. Measure reviewer agreement and adjudicate disagreements.
7. Keep a private holdout separate from training, validation, and calibration data.
8. Freeze benchmark/evaluator revisions and hash all final assets.

No CPT, SFT, preference optimization, or RL was started in this phase.

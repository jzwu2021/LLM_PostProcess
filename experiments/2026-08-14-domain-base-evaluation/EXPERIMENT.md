# Domain Base evaluation: first-stage diagnostics

## Status

`COMPLETE` as a diagnostic pipeline; authoritative domain capability score is intentionally `None`.

## Objective

Run a deterministic, auditable first-stage evaluator over the frozen Base raw generations. This stage checks generation-record integrity and produces clearly labeled diagnostics. It does not convert the synthetic benchmark's reference outlines into a domain-accuracy claim.

## Inputs

- Benchmark: `research/ai-infra-expert/benchmark.jsonl`
- Benchmark SHA-256: `30eb88766f916dd487dc2921b9479e238c1074a96d20611f099068af917cc869`
- Generations: `../2026-08-13-domain-base-baseline/results/generations.jsonl`
- Generations SHA-256: `508735d8453ef17f30840948e7c3dbe952d1b9754ef4954290c70c735bd69d42`
- Model: untouched Qwen3.5-9B Base
- Frozen generation protocol: temperature 0, thinking disabled, max_tokens 768

## Evaluator

`evaluate_domain_baseline.py` SHA-256: `4ac3e5827c06c79ef8368da21edcd98f6d1b868cb2974305a60f4a2a67c4e566`

The evaluator performs:

- benchmark/generation ID coverage checks;
- generation success, non-empty response and finish-reason diagnostics;
- code-fence presence diagnostic;
- explicitly labeled heuristic numeric extraction and 1% matching;
- explicitly labeled lexical reference-keyword coverage;
- routing of open-ended cases to blinded human rubric review;
- routing of code cases to a future versioned sandbox fixture.

## Command

```bash
/media/home/johnson/llm/qwen35-env/bin/python -m py_compile \
  experiments/2026-08-14-domain-base-evaluation/evaluate_domain_baseline.py

/media/home/johnson/llm/qwen35-env/bin/python \
  experiments/2026-08-14-domain-base-evaluation/evaluate_domain_baseline.py \
  --benchmark research/ai-infra-expert/benchmark.jsonl \
  --generations experiments/2026-08-13-domain-base-baseline/results/generations.jsonl \
  --output experiments/2026-08-14-domain-base-evaluation/results/diagnostic-report.json
```

## Results

```text
benchmark cases: 500
matched generation cases: 500
missing generation cases: 0
generation_ok: 500
non-empty responses: 500
finish_reason=length: 476
finish_reason=stop: 24
authoritative_domain_score: None
```

Required follow-up review routing:

```text
blind rubric cases: 310
sandbox fixture cases: 50
numeric diagnostic cases: 50
key-point diagnostic cases: 405
```

Non-authoritative diagnostics only:

```text
numeric heuristic all-expected-matched: 3/50
mean lexical reference-keyword coverage: 0.3314
```

## Interpretation boundary

The numeric and lexical values above are diagnostics, not accuracy or capability scores. The benchmark is `curated_template_v0.1`, its factual source evidence is still pending, and reference answers are not audited exhaustive key-point annotations. Generated code was not executed because no per-case sandbox tests are versioned. Open-ended design, performance, troubleshooting, architecture, reasoning and long-form cases require blinded 1–4 rubric scoring by at least two reviewers on a calibration subset before any domain claim.

Runtime metrics such as latency, GPU memory, and wall time remain separate Runtime/System Capability measurements.

## Next gated phase

Before CPT/SFT:

1. Create audited key-point annotations and numeric answer metadata.
2. Add versioned, network-disabled code fixtures and resource limits.
3. Build blinded rubric packets for open-ended cases.
4. Score a stratified calibration subset with two domain reviewers and calculate agreement.
5. Freeze evaluator revision and private holdout policy.

No training was started in this phase.

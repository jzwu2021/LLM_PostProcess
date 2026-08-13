# Experiment 2026-08-13: Qwen3.5-9B Base Domain Benchmark Baseline

## Status

`COMPLETE` (raw-generation baseline; rubric scoring pending).

Attempt 1 (512 tokens, thinking enabled) ended at the cap for all 500 cases. Attempt 2 (1024 tokens, thinking enabled) emitted reasoning content without final answer content and ended at the cap for all 500 cases. Attempt 3 disabled thinking and produced non-empty direct content for all 500 cases, but 499/500 ended at the 512-token cap. Attempt 4 disabled thinking and used 768 tokens: all 500 requests succeeded and produced non-empty direct content, with 24 `stop` and 476 `length` finish reasons. Attempt 4 is the authoritative raw-generation artifact, but the 476 capped answers remain a limitation for later rubric/code/numeric scoring.

## Objective and hypothesis

Objective: establish a reproducible pre-training baseline for Model Domain Capability before CPT/SFT/Preference/RLVR experiments.

Hypothesis H0: the Base model's performance on AI/LLM Infrastructure knowledge, calculation, code, design, performance analysis, troubleshooting, architecture comparison, reasoning, and long-form technical analysis is the reference point for every later model variant.

No training, checkpoint modification, or benchmark-data ingestion occurs in this experiment.

## Frozen inputs

- Model: `/media/home/johnson/llm/models/Qwen3.5-9B`
- Benchmark: `research/ai-infra-expert/benchmark.jsonl`
- Benchmark records: 500
- Benchmark categories: 10 categories, 50 records each
- Benchmark revision: `benchmark_v0.1`
- Benchmark SHA-256: `30eb88766f916dd487dc2921b9479e238c1074a96d20611f099068af917cc869`
- Model revision: base local checkpoint; no post-training checkpoint
- Evaluator source: `run_domain_baseline.py`
- Prior repository commit: `132986d688e4f30e8bb7a681b4063ed937c78a1b`

## Protocol

Fixed system prompt:

```text
You are an AI/LLM Infrastructure engineering assistant. Answer the user's technical question directly and rigorously. State assumptions, units, formulas, trade-offs, uncertainty, and validation steps when relevant. Do not invent measurements or undocumented system facts.
```

Fixed decoding:

```text
temperature=0.0
max_tokens=768
chat_template_kwargs={"enable_thinking": false}
```

Each request contains only the fixed system prompt and the benchmark question. Reference answers are not sent to the model. The runner records the raw request, raw response, finish reason, usage, per-request latency, retry count, and errors.

## Commands

### Validate runner before execution

```bash
/media/home/johnson/llm/qwen35-env/bin/python -m py_compile \
  experiments/2026-08-13-domain-base-baseline/run_domain_baseline.py
```

### Start vLLM

Use the same successfully validated 8-GPU vLLM command as the previous Base baseline, with the model name `qwen35-9b-base` and localhost port `8001`. The exact environment-variable workaround is documented in the previous experiment and will be copied into the execution log.

### Run benchmark

```bash
export PATH=/media/home/johnson/llm/qwen35-env/bin:$PATH
export LD_LIBRARY_PATH=/home/johnson/miniforge3/lib:/media/home/johnson/llm/qwen35-env/lib/python3.12/site-packages/nvidia/cu13/lib:/media/home/johnson/llm/qwen35-env/lib/python3.12/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
/media/home/johnson/llm/qwen35-env/bin/python \
  experiments/2026-08-13-domain-base-baseline/run_domain_baseline.py \
  --benchmark research/ai-infra-expert/benchmark.jsonl \
  --base http://127.0.0.1:8001/v1 \
  --model qwen35-9b-base \
  --max-tokens 768 \
  --output experiments/2026-08-13-domain-base-baseline/results/generations.jsonl \
  --summary experiments/2026-08-13-domain-base-baseline/results/summary.json \
  2>&1 | tee experiments/2026-08-13-domain-base-baseline/logs/run-attempt4.log
```

## Scoring status

The first execution artifact is raw generation only. The following are deliberately not inferred from loss, throughput, latency, or response length:

- knowledge accuracy;
- numeric exact/tolerance pass rate;
- code unit-test pass rate;
- system-design rubric score;
- troubleshooting rubric score;
- hallucination rate;
- expert-level capability.

The runner emits only transparent automatic checks such as non-empty response and presence of a code fence. These are diagnostics, not domain scores. A subsequent evaluator must implement numeric tolerance, sandboxed code tests, key-point checks, and blinded 1–4 rubric scoring with reviewer agreement.

## Result

Attempt 4 authoritative raw generation completed with 500/500 requests, 500 unique IDs, 500 non-empty direct answers, 0 request errors, 24 `finish_reason=stop`, and 476 `finish_reason=length`. The raw generations are complete as a benchmark run but capped for most cases; no domain capability score is claimed. Attempt 1–3 artifacts and logs are retained for protocol audit. Hashes are recorded in `MANIFEST.sha256`.


## Required interpretation boundary

This is Model Domain Capability measurement infrastructure. GPU memory, latency, throughput, and wall time are Runtime/System Capability or training-execution metrics and must be reported separately. The benchmark is an authored synthetic scaffold (`needs_domain_expert_review`), so even a high score cannot justify a claim of production-expert capability.

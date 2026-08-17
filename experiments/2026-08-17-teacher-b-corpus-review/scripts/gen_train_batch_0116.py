import json

CORPUS = 'research/ai-infra-expert/corpus/train.jsonl'
OUT = 'experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0116.jsonl'
START, N = 1150, 10

rows = [json.loads(l) for l in open(CORPUS)][START:START + N]


def answer(variant, category):
    lens_map = {
        'Performance Analysis': (
            "Analysis lens: treat the mixed short-prompt/long-generation workload as two coupled queues "
            "(prefill-bound and decode-bound) sharing one KV-cache budget; the metric of record is where the "
            "P99 tail is manufactured, not the mean."),
        'System Design': (
            "Design lens: the evaluation harness itself is the deliverable — a reproducible load generator, "
            "a fixed workload manifest, and a scheduler-configuration matrix that can be replayed after any "
            "serving-stack change."),
        'Troubleshooting': (
            "Diagnostic lens: assume the service is already violating a P99 SLO and the evaluation must "
            "localize the violation to prefill admission, decode batch residency, or queueing before any "
            "config change is attempted."),
    }
    lens = lens_map.get(category, lens_map['Performance Analysis'])

    return f"""Scenario variant {variant} — evaluation plan for a mixed short-prompt / long-generation LLM serving endpoint.

{lens}

1. Assumptions (state them, they bound every number below)
- Single model replica class, continuous batching (vLLM-style) with paged KV cache; no speculative decoding unless declared as a separate arm.
- Closed-loop clients with a fixed concurrency ladder, not open-loop Poisson, unless the production traffic is genuinely open — the two produce different queueing tails and are not comparable.
- Tokenizer is fixed; all token counts are measured with the serving tokenizer, never estimated from characters.
- Hardware, driver, CUDA, and serving-engine versions are pinned and recorded per run; a version delta invalidates cross-run comparison.

2. Metric definitions (ambiguity here is the most common source of bogus results)
- TTFT = wall clock from request admission at the HTTP/gRPC boundary to first streamed token. Record client-side and server-side separately; the gap is queueing + network.
- TPOT = (end-to-end latency - TTFT) / (generated_tokens - 1), per request, then aggregated. Never derive it from a global token/s counter.
- Throughput reported as three separate numbers: prefill tokens/s, decode tokens/s, and completed requests/s. A single "tokens/s" figure hides the prefill-decode tradeoff.
- Queueing delay = time in scheduler waiting queue before first forward pass, exported by the engine, not inferred.
- P99 latency reported per workload class (short-prompt-short-gen, short-prompt-long-gen, long-prompt-long-gen) — pooled P99 across classes is meaningless because the mixture ratio dominates it.

3. Falsifiable hypothesis
H1: "Under the production mixture, P99 TTFT degradation above 60% concurrency is caused by prefill chunking limits (long generations holding KV blocks and shrinking the admissible prefill batch), not by raw GPU compute saturation."
Prediction if H1 is true: increasing max_num_batched_tokens (chunked prefill budget) at constant KV-cache size moves P99 TTFT down by a measurable margin while decode TPOT rises, and GPU SM occupancy during the TTFT spike is below saturation.
Prediction if H1 is false: P99 TTFT is flat under that change and SM occupancy is pegged near saturation during spikes — the bottleneck is compute, and the fix is replica scaling or quantization, not scheduler tuning.
H1 is refuted by either the flat-TTFT observation or the saturated-occupancy observation; both are recorded automatically.

4. Controlled experiment
- Factor A: chunked-prefill token budget (3 levels). Factor B: max concurrent sequences / KV-cache utilization cap (3 levels). Factor C: mixture ratio of long generations (2 levels: production ratio, and 2x long-generation share as a stress arm).
- Full factorial where affordable, otherwise a screening design with the production point replicated; randomize run order to de-confound thermal drift.
- 5 minutes warmup discarded, then >=3 independent repeats per cell on separate process starts. Report median and the spread across repeats; a cell whose repeat spread exceeds the effect size is reported as inconclusive, not as a result.
- Fixed request corpus with a frozen seed so prompt lengths and generation lengths are identical across arms; max_tokens forced rather than left to EOS, otherwise the arms differ in work, not in scheduling.

5. Confounders to control explicitly
- Prefix caching / KV reuse silently reducing prefill work on repeated corpora — either disable it or report cache hit rate per arm.
- Client-side bottlenecks (single-threaded tokenizer, HTTP connection limits) masquerading as server latency; validate with a no-op echo endpoint at the same concurrency.
- GPU clock throttling and ECC/thermal drift over long sweeps; log clocks and temperature per run.
- Cross-tenant interference if the GPU is shared (MPS/MIG or other processes); require exclusive placement for the record runs.
- Autoscaler or load-balancer behavior changing replica count mid-run.

6. Instrumentation and evidence to collect
- Per-request trace: arrival, admission, first token, completion, prompt_tokens, generated_tokens, scheduler queue depth at admission.
- Engine counters: running/waiting sequence counts, KV-cache utilization, preemption/recompute events, chunked-prefill batch composition.
- GPU: memory in use vs reserved, SM occupancy/achieved FLOPs sample, NVLink/PCIe traffic if tensor-parallel.
- Nsight Systems or engine-native profile on one representative cell only — profiling perturbs timing and must not be part of the record runs.

7. Rollback / decision gates
- Ship a scheduler change only if P99 TTFT improves by more than the measured repeat-to-repeat spread AND P99 end-to-end latency for the long-generation class does not regress beyond the SLO budget.
- Canary at <=5% of traffic for at least one full diurnal cycle; automatic rollback on P99 TTFT above the pre-change baseline, on any preemption-rate increase, or on OOM/KV-eviction events appearing where the baseline had none.
- Keep the previous engine configuration as a single-flag revert; no partial migrations.

8. Reporting
Publish the workload manifest, engine version, config matrix, raw per-request traces, and the aggregation script. A result without the raw traces is not reproducible and should be treated as provisional."""


out = []
for r in rows:
    m = r['messages']
    u = [x for x in m if x['role'] == 'user'][0]['content']
    a = [x for x in m if x['role'] == 'assistant'][0]['content']
    variant = u.split('Scenario variant ')[1].split(':')[0].strip()
    out.append({
        "source_id": r['id'],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": u,
        "source_assistant": a,
        "corrected_answer": answer(variant, r.get('category', '')),
        "quality_dimensions": {
            "technical_correctness": 3,
            "instruction_coverage": 2,
            "operational_safety": 3
        },
        "risks": [
            "source_assistant is a grading rubric, not an answer; training on it teaches meta-commentary instead of engineering reasoning",
            "no metric definitions, so TTFT/TPOT can be computed inconsistently and results become non-comparable",
            "prefix caching and forced-vs-EOS generation length are unhandled confounders that can fabricate throughput gains",
            "pooled P99 across workload classes hides class-specific SLO violations",
            "profiling overhead can be mistaken for a real latency regression if profiles are taken during record runs"
        ],
        "evidence_required": [
            "per-request traces with arrival/admission/first-token/completion timestamps and prompt/generated token counts",
            "engine counters: queue depth, KV-cache utilization, preemption and recompute events, chunked-prefill batch composition",
            "GPU telemetry: memory used vs reserved, occupancy sample, clock/thermal log across the sweep",
            "pinned versions of engine, CUDA, driver, and tokenizer plus the frozen workload manifest and seed",
            "repeat-to-repeat spread per cell to bound the minimum detectable effect"
        ],
        "confidence": 0.72
    })

with open(OUT, 'w') as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print("wrote", OUT, len(out))

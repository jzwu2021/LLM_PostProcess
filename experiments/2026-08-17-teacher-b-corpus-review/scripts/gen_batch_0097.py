import json

SRC='research/ai-infra-expert/corpus/train.jsonl'
OUT='experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0097.jsonl'
START=960; N=10

src=[json.loads(l) for l in open(SRC)]

COMMON = """Assumptions (state and check before trusting any number)
- Single-node inference server (vLLM/TGI/SGLang class) with continuous batching and paged KV cache; one model replica per measurement unless the sweep explicitly varies TP.
- Hardware fixed for the whole sweep: same GPU SKU, same driver/CUDA/runtime versions, same power cap and clock policy (record `nvidia-smi -q -d PERFORMANCE,POWER,CLOCK`). A silent clock or power-cap change invalidates cross-run comparison.
- Workload is a mix of short prompts and long generations, so aggregate "throughput" is meaningless unless prefill and decode are separated.

Why prefill and decode must be measured separately (mechanism)
- Prefill processes the whole prompt in one pass: work is O(prompt_tokens) of dense GEMMs, compute-bound, and it sets TTFT.
- Decode emits one token per step per sequence: per-step work is small GEMV-like matmuls plus a full read of model weights and of the KV cache, so it is memory-bandwidth-bound and it sets TPOT.
- Therefore a change that helps one usually hurts the other. Larger batches raise decode token throughput (better weight-read amortization) but lengthen prefill queueing and inflate TTFT. Chunked prefill / prefill-decode interleaving trades TTFT tail for TPOT smoothness. Any plan reporting a single "tokens/s" number hides this trade and cannot support a capacity decision.

Metric definitions (fix these before collecting; ambiguity here is the most common source of bogus comparisons)
- TTFT = arrival timestamp at the server front door -> first streamed token. Include queue wait; report queue wait as its own series so a TTFT regression can be attributed to admission vs prefill compute.
- TPOT = (end_to_end_latency - TTFT) / (output_tokens - 1). Per-request, then aggregate.
- Output throughput = decode tokens/s across the server; report separately from prefill tokens/s. Also report goodput: requests/s meeting the SLO, since throughput at a violated SLO is not capacity.
- Queueing: time in scheduler queue, running batch size, number of preemptions/recomputations, KV-cache utilization percent.
- P99: computed per arm over >= 2000 completed requests; below that the P99 estimator is too noisy to distinguish arms.

Experiment design
- Open-loop load generation at fixed request rate (Poisson arrivals), not closed-loop fixed-concurrency. Closed-loop hides queueing collapse because the client throttles itself.
- Fix the workload distribution: sample prompt and output lengths from one frozen trace file (seeded), replayed identically in every arm. Do not resample per arm.
- Warmup: discard the first 60 s and require steady-state before recording (CUDA graph capture, weight paging, allocator growth, and JIT/autotune all land in that window). Report the warmup rule explicitly.
- Repeat each arm >= 3 times in randomized order, on the same node, reporting median and inter-run spread. If run-to-run spread exceeds the effect size, the result is not usable.
- Sweep request rate upward until SLO violation to find the knee; report the capacity number as "requests/s at which P99 TTFT crosses the SLO", not peak throughput.

Falsifiable hypothesis (a concrete example to instantiate for this scenario)
- H1: Enabling chunked prefill with a chunk size of 512 tokens reduces P99 TTFT by >= 25% at the pre-knee load, while degrading median TPOT by <= 10%.
- Prediction if true: TTFT tail shrinks and the running-batch-size series shows fewer long prefill stalls; decode step time rises slightly and uniformly.
- Falsifier: if P99 TTFT improves < 25%, or TPOT degrades > 10%, or output throughput drops > 5%, H1 is rejected for this workload. Stating the reject thresholds up front prevents post-hoc rationalization.

Expected confounders (and the control for each)
- Tokenizer/length mismatch: count tokens with the served model's tokenizer, not characters or client-side estimates.
- Client-side bottleneck: the load generator itself can cap rate; verify client CPU < 70% and that measured arrival rate matches the target.
- Thermal/power drift: long sweeps let GPUs clock down; log SM clock and power per run and discard runs with > 5% clock drift.
- Noisy neighbors / other processes on the GPU: assert exclusive use via `nvidia-smi` process list.
- Cache effects: prefix caching can make repeated prompts artificially cheap; either disable it or make prefix reuse an explicit, reported factor.
- Cold vs warm KV pressure: report KV-cache utilization; a run that never approached preemption is not comparable to one that did.

Instrumentation
- Request-level traces (arrival, schedule, first token, completion, prompt/output token counts) written to disk per run, so percentiles can be recomputed later without re-running.
- GPU memory: peak allocated and reserved, plus KV-cache blocks free over time.
- Server scheduler counters: batch size, preemption count, waiting-queue depth, sampled at >= 1 Hz.

Reporting and rollback gate
- Report a table per arm: rate, median/P99 TTFT, median/P99 TPOT, decode throughput, goodput, preemptions, KV utilization, plus run-to-run spread. Label every number measured or estimated.
- Rollback criterion for any config change promoted to production: revert if P99 TTFT regresses > 10%, or goodput drops > 5%, or preemption rate rises above the pre-change baseline, measured on the same frozen trace during a canary at production load.
- Uncertain/platform-specific claims (exact chunk sizes, scheduler flag names, achievable numbers) must be confirmed against the deployed engine version rather than assumed; treat any number here as a hypothesis to measure, not a fact."""

FOCUS = {
 'Troubleshooting': "\n\nDiagnosis-first ordering for this variant: before tuning, confirm where the time goes. Compare the TTFT and TPOT series against the queue-wait series to split admission delay from compute; if queue wait dominates, the fix is admission/scaling, not kernel-level tuning, and a config sweep will produce a misleading win.",
 'Performance Analysis': "\n\nAnalysis emphasis for this variant: model the expected decode step time as (bytes read per step) / (achieved HBM bandwidth), where bytes read ~ model weights + active KV. If measured TPOT is far above that bound, the gap is scheduling/overhead, not bandwidth, and bandwidth-oriented tuning will not help. State the roofline estimate explicitly and mark it as an estimate.",
 'System Design': "\n\nDesign emphasis for this variant: decide up front whether prefill and decode share the same replica or are disaggregated. Disaggregation isolates TTFT from decode batching but adds a KV transfer on the interconnect; the transfer cost must be measured (bytes moved per request and achieved link bandwidth) before claiming benefit. Keep the same evaluation harness for both topologies so the comparison is valid.",
}

recs=[]
for i in range(START, START+N):
    r=src[i]
    m={x['role']:x['content'] for x in r['messages']}
    cat=r.get('category','System Design')
    ans=COMMON + FOCUS.get(cat, FOCUS['System Design'])
    recs.append({
      "source_id": r['id'],
      "teacher_lane": "teacher-B",
      "teacher_model": "claude-opus-5-current",
      "calibration_status": "provisional",
      "decision": "rewrite",
      "source_user": m['user'],
      "source_assistant": m['assistant'],
      "corrected_answer": ans,
      "quality_dimensions": {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 3},
      "risks": [
        "source_assistant is a grading rubric describing what an answer should contain, not an answer; training on it teaches meta-commentary instead of engineering reasoning",
        "no metric definitions, so TTFT/TPOT could be measured inconsistently across arms",
        "no load-generation model specified; closed-loop clients would hide queueing collapse",
        "capacity conclusions drawn from single runs without warmup or repetition can be off by more than the effect being measured"
      ],
      "evidence_required": [
        "frozen seeded request trace (prompt/output token length distribution) used identically in all arms",
        "per-request traces: arrival, schedule, first-token, completion, token counts",
        "scheduler counters: batch size, queue depth, preemptions, KV-cache utilization over time",
        "nvidia-smi clock/power/process logs per run to rule out throttling and noisy neighbors",
        "engine and driver versions plus exact server flags for each arm",
        ">= 3 repeats per arm with reported inter-run spread and >= 2000 completed requests per arm for P99"
      ],
      "confidence": 0.72
    })

with open(OUT,'w') as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False)+"\n")
print("wrote", OUT, len(recs), recs[0]['source_id'], recs[-1]['source_id'])

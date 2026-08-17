import json

START=1140; N=10; OUT='experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0115.jsonl'
rows=[json.loads(l) for l in open('research/ai-infra-expert/corpus/train.jsonl').read().splitlines()[START:START+N]]

def answer(cat, vid):
    focus = {
      'System Design': "Design emphasis: the harness must be able to change one factor at a time (batch policy, max_num_seqs, chunked-prefill on/off) without changing the workload trace.",
      'Troubleshooting': "Diagnosis emphasis: when P99 regresses, first separate queue-wait from on-GPU service time; a rise in queue-wait with flat TPOT means admission/capacity, not kernel-level slowness.",
      'Performance Analysis': "Analysis emphasis: report TTFT and TPOT as distributions (p50/p95/p99) per input-length bucket, never a single blended mean, because a short/long prompt mix makes the blended mean a mixture artifact.",
    }[cat]
    return f"""Assumptions (state explicitly, do not assume vendor-specific behavior): single model replica on a known GPU type and count; continuous batching serving engine (e.g. vLLM-class) with paged KV cache; client and server clock-synced; network RTT measured separately and subtracted from TTFT if the goal is server-side latency. Scenario variant {vid}.

Definitions used, with units:
- TTFT (ms) = first output token timestamp minus request-arrival timestamp at the server. It contains queue-wait + prefill compute.
- TPOT (ms/token) = (end-to-end latency - TTFT) / (output_tokens - 1). Decode-side only.
- Throughput = output tokens/s aggregated, and separately requests/s completed; report both, they diverge under a long-generation mix.
- Queueing (ms) = time between admission-eligible and scheduler dispatch; must be instrumented in the engine, not inferred.
- P99 latency (ms) computed over a fixed, pre-registered request population, not over whatever completed in the window (completion bias truncates long generations).

Falsifiable hypothesis (pre-register before running): H1 = "At fixed arrival rate lambda, increasing max batch size from B0 to 2*B0 increases output-token throughput by >=15% while degrading TTFT p99 by <=20%." Null: throughput gain <15% or TTFT p99 degradation >20%. This is falsifiable because both quantities are directly measured with a fixed workload trace and fixed decoding params.

Controlled experiment:
1. Freeze the workload: replay a recorded trace with fixed prompt-length and output-length distributions (log them: e.g. prompt p50/p95, output p50/p95) and a fixed arrival process (Poisson at lambda, or closed-loop with fixed concurrency - say which, they are not interchangeable).
2. Fix decoding: temperature, max_tokens, ignore_eos on/off. ignore_eos=true makes output length deterministic and is the cleaner throughput measurement; ignore_eos=false is the realistic latency measurement. Run both, do not mix.
3. Warmup: discard the first N requests (or first 60 s) until TTFT p50 is stationary; report the discarded count and the stationarity check.
4. Repeat: >=3 independent trials per configuration, randomized order, report mean and CI or min/median/max. A single run cannot distinguish a 10% effect from run-to-run noise.
5. Sweep lambda upward until the system is saturated (queue-wait grows without bound); report the knee, not just one operating point.

Instrumentation to capture per request: arrival, admission, first-token, last-token timestamps; prompt and output token counts; batch size and number of running sequences at admission; KV-cache utilization; preemption/recompute events. Plus GPU-level: memory used vs reserved, SM/tensor-core utilization, power and clocks (thermal or power capping silently changes TPOT).

{focus}

Expected confounders: (a) prefill/decode interference - a large prefill inflates TPOT of unrelated in-flight sequences unless chunked prefill bounds it; (b) KV-cache pressure causing preemption/recompute, which shows as bimodal TTFT; (c) client-side bottleneck (single-threaded client, TLS, tokenizer on the hot path) masquerading as server latency - validate by running the client on a second host; (d) power/thermal drift over long runs; (e) other tenants on the same node or NIC; (f) completion bias in the percentile window.

Rollback / stop criteria (pre-agreed, not negotiated after the fact): revert the configuration if TTFT p99 regresses >20% versus baseline, or if preemption rate >1% of requests, or if any request errors/timeouts appear above the baseline rate, or if the throughput gain is inside the trial-to-trial CI. Roll back by config flag only - keep the previous engine build deployed so rollback is a restart, not a rebuild.

Evidence needed before claiming a win: the raw per-request trace files, the config diff, the GPU counters for the same window, and the >=3-trial variance. Absent those, any single-number improvement is an estimate, not a measured fact."""

with open(OUT,'w') as f:
    for r in rows:
        m={x['role']:x['content'] for x in r['messages']}
        vid=r['id']
        rec={
          "source_id": r['id'],
          "teacher_lane": "teacher-B",
          "teacher_model": "claude-opus-5-current",
          "calibration_status": "provisional",
          "decision": "rewrite",
          "source_user": m['user'],
          "source_assistant": m['assistant'],
          "corrected_answer": answer(r['category'], vid),
          "quality_dimensions": {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 3},
          "risks": [
            "source_assistant is a grading rubric, not an answer; training on it teaches meta-commentary instead of engineering reasoning",
            "no units or definitions for TTFT/TPOT, so the metric is ambiguous between queue-inclusive and compute-only",
            "no explicit falsifiable hypothesis despite the prompt demanding one",
            "no rollback threshold, so a capacity change could ship on a single noisy run"
          ],
          "evidence_required": [
            "per-request trace with arrival/admission/first-token/last-token timestamps",
            "workload distribution log (prompt and output length percentiles) and arrival process spec",
            ">=3 repeated trials per config with variance reported",
            "GPU memory, utilization, power/clock counters for the same window",
            "preemption/recompute event counts from the serving engine"
          ],
          "confidence": 0.62
        }
        f.write(json.dumps(rec, ensure_ascii=False)+"\n")
print("wrote", OUT, len(rows))

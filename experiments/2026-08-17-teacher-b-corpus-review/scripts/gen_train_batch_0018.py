#!/usr/bin/env python3
"""Generate teacher-B provisional BLIND review batch train-batch-0018 (corpus lines 171-180)."""
import json, os, pathlib

ROOT = pathlib.Path("/home/johnson/workspace/LLM_PostProcess")
CORPUS = ROOT / "research/ai-infra-expert/corpus/train.jsonl"
OUT = ROOT / "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0018.jsonl"
START, END = 171, 180  # 1-indexed inclusive

EXP_COMMON = (
    "Assumptions: single node, 8x A30 24GB, one vLLM-class server per arm, fixed model/weights/dtype/TP degree, "
    "fixed random seed for the request sampler, warm cache and warmed CUDA graphs before measurement.\n"
    "Mechanism under test: continuous (iteration-level) batching admits and retires requests at each decode "
    "iteration boundary, so a finished sequence releases its KV slot immediately instead of holding a static "
    "batch slot until the slowest sequence in the batch finishes.\n"
)

EXP_BOUNDARY = (
    "Boundary condition: the benefit collapses when output lengths are near-uniform (no straggler to displace) or "
    "when KV cache is the binding constraint - once KV occupancy is saturated, the scheduler cannot admit new "
    "requests at iteration boundaries and continuous batching degenerates to static batching plus queueing.\n"
)

EXP_ROLLBACK = (
    "Rollback gate: revert to the previous scheduler configuration if p99 TTFT regresses >20% at equal throughput, "
    "if preemption/recompute rate exceeds 1% of decode steps, or if any OOM or KV-eviction-induced request failure "
    "is observed. Falsifiable: if throughput gain <5% with CI excluding zero at matched p99 latency, the hypothesis "
    "that continuous batching helps this workload is rejected.\n"
)

EXP_EVIDENCE = (
    "Evidence to collect: per-request arrival/first-token/completion timestamps, tokens in/out histograms, "
    "scheduler queue depth and running-batch size sampled per iteration, KV cache utilization (%), preemption count, "
    "GPU SM occupancy and DRAM bandwidth via nvidia-smi dmon / DCGM, and server-side throughput (output tok/s). "
    "Report medians and p95/p99 over >=5 independent runs with confidence intervals; label everything measured vs estimated."
)

def exp_answer(variant):
    designs = {
        1: ("Design (A/B, single factor): Arm A = static/fixed-batch scheduling (max_num_seqs fixed, no iteration-level "
            "admission); Arm B = continuous batching, identical max_num_seqs and KV budget. Workload: Poisson arrivals at "
            "lambda swept over {0.5, 1.0, 1.5, 2.0} x measured saturation rate, input length fixed at 512 tokens, output "
            "length drawn from a heavy-tailed distribution (e.g. lognormal, median 128, p99 2048 tokens) to create stragglers. "
            "Primary metric: output tokens/s at fixed p99 end-to-end latency. Secondary: p50/p99 TTFT and TPOT.\n"),
        2: ("Design (paired replay): capture one production request trace (arrival times, prompt lengths, sampled output "
            "lengths) and replay the identical trace against both schedulers, so the only difference is the admission policy. "
            "Paired replay removes arrival-process variance and lets you use a paired test (Wilcoxon signed-rank) on per-request "
            "latency deltas rather than comparing two noisy aggregate means. Primary metric: per-request latency delta "
            "distribution; secondary: goodput at an SLO of p99 TTFT <= 500 ms.\n"),
        3: ("Design (factorial 2x3): factor 1 = scheduling {static, continuous}; factor 2 = output-length dispersion "
            "{CV=0.1, CV=0.8, CV=2.0} at matched mean output length. This isolates the causal claim: the advantage of "
            "continuous batching should scale with output-length dispersion and vanish at CV~0. Fit throughput ~ scheduler * CV "
            "and test the interaction term; a non-significant interaction falsifies the straggler-displacement explanation.\n"),
        4: ("Design (saturation sweep / load-latency curve): for each scheduler, sweep offered load from 10% to 130% of "
            "measured peak in 10% steps, 5 minutes per point after a 60 s warmup, and plot throughput vs p99 latency. "
            "Compare goodput at a fixed latency SLO rather than peak throughput, because peak throughput alone hides the "
            "queueing regime where continuous batching's advantage actually materializes. Include a KV-pressure arm where "
            "gpu_memory_utilization is lowered to force eviction.\n"),
        5: ("Design (ablation on admission granularity): three arms at identical KV budget - (a) static batch, "
            "(b) continuous batching with chunked prefill disabled, (c) continuous batching with chunked prefill enabled. "
            "This separates the retire-early effect from the prefill-interference effect, which are commonly conflated. "
            "Randomize arm order across repetitions to guard against thermal drift; log GPU clocks and temperature per run "
            "and discard runs where SM clock drops >5% from baseline.\n"),
    }
    return designs[variant] + EXP_COMMON + EXP_BOUNDARY + EXP_ROLLBACK + EXP_EVIDENCE


def runbook_answer(variant):
    head = ("RUNBOOK: investigating continuous batching behaviour on an LLM inference server.\n"
            "Scope assumption: single-node vLLM-class server, TP degree fixed, no autoscaling during the investigation.\n")
    mech = ("Mechanism to keep in mind: the scheduler makes an admit/retire decision at every decode iteration boundary; "
            "a completed sequence frees its KV blocks immediately, and a queued request can be admitted mid-generation. "
            "Symptoms therefore show up as running-batch-size and KV-utilization time series, not as a single throughput number.\n")
    steps = {
        2: ("Steps:\n"
            "1. Confirm the scheduler is actually running continuous batching (server config dump: max_num_seqs, "
            "max_num_batched_tokens, enable_chunked_prefill, preemption_mode). Record the exact values; do not assume defaults.\n"
            "2. Sample /metrics at 1 s resolution for >=10 min: running requests, waiting requests, KV cache usage %, "
            "preemption count, TTFT and TPOT histograms.\n"
            "3. Classify the regime: waiting>0 and KV<80% => scheduler-limited (raise max_num_seqs / max_num_batched_tokens); "
            "KV>95% with rising preemptions => memory-limited (reduce concurrency, shorten max_model_len, or add KV offload).\n"
            "4. Only after classification, change one parameter and re-measure.\n"),
        4: ("Steps:\n"
            "1. Reproduce with a fixed replayed trace before touching production knobs; an unreproducible symptom is not "
            "yet an incident, it is a measurement problem.\n"
            "2. Check for prefill interference: correlate TPOT spikes with prefill admissions. If TPOT spikes align with "
            "long-prompt arrivals, enable/tune chunked prefill instead of lowering batch size.\n"
            "3. Check KV fragmentation and preemption/recompute counters; recompute-based preemption converts a memory "
            "problem into a latency problem and is easy to misread as a compute regression.\n"
            "4. Verify no cross-tenant noise: confirm GPU exclusivity (nvidia-smi processes), no MIG/MPS sharing, stable SM clocks.\n"),
        5: ("Steps:\n"
            "1. Snapshot baseline: server config, model, dtype, TP/PP degree, KV block size, driver and NCCL versions. "
            "Attach this snapshot to the ticket; version skew is a frequent root cause and cannot be reconstructed later.\n"
            "2. Collect a 10-minute metrics window plus a request-level log with arrival/first-token/completion timestamps.\n"
            "3. Compute admission latency (arrival -> first scheduled iteration) separately from prefill and decode time. "
            "Blaming 'batching' without this split is the most common misdiagnosis.\n"
            "4. Escalate with the three time series (queue depth, running batch, KV %) rather than a single averaged latency number.\n"),
    }
    boundary = ("Boundary condition: these steps assume KV cache is not already saturated. Once KV utilization is pinned "
                "near 100%, iteration-level admission has no free slots to grant and every tuning knob on the scheduler side "
                "will look inert - the correct lever is then memory (max_model_len, quantized KV, offload, or more GPUs).\n")
    eviction = ("Rollback gate: any config change is reverted if p99 TTFT regresses >20%, preemption rate exceeds 1% of "
                "decode steps, or error rate rises above baseline; keep the previous config file and restart command ready. "
                "Change one knob at a time and hold each change for >=1 full traffic cycle before judging.\n")
    ev = ("Evidence required to close the ticket: before/after metrics windows of equal length under comparable load, "
          "the exact diff of server flags, and a stated falsifiable claim (e.g. 'raising max_num_seqs from 128 to 256 "
          "increases output tok/s by >=10% at unchanged p99 TTFT'). Mark measured vs estimated numbers explicitly.")
    return head + mech + steps[variant] + boundary + eviction + ev


def tp_answer(variant):
    base = ("Definition: tensor parallelism (TP) shards the parameters and the arithmetic of individual layers across N GPUs - "
            "for a transformer, attention heads and the MLP weight matrices are split, so each rank holds 1/N of the weights and "
            "computes a partial result.\n"
            "Concrete mechanism: in the standard Megatron-style layout the first MLP GEMM is column-parallel (no communication "
            "after it because the GeLU is elementwise) and the second is row-parallel, requiring one all-reduce of the "
            "hidden-state activations per MLP block; attention is similarly sharded head-wise with one all-reduce per block. "
            "That is on the order of 2 all-reduces per transformer layer per forward pass, each moving roughly "
            "batch x seq x hidden x dtype_bytes of data.\n")
    why = {
        1: ("Why it matters: TP is what makes a model whose weights plus KV cache exceed one GPU's HBM servable at all, and it "
            "also cuts per-GPU weight-read bytes in the memory-bandwidth-bound decode phase, which is the phase that sets "
            "inter-token latency. On 8x A30 24GB, a bf16 9B model (~18 GB of weights) leaves almost no KV headroom at TP=1; "
            "TP=2 drops weights to ~9 GB/GPU and buys usable KV capacity.\n"),
        2: ("Why it matters: TP reduces per-GPU weight footprint and per-GPU weight traffic in decode, so it is the primary "
            "lever for latency-sensitive serving - unlike pipeline parallelism, it does not add pipeline bubbles or require "
            "microbatching to stay efficient. The cost is that it puts a blocking collective on the critical path of every "
            "layer, so it is only cheap over a high-bandwidth intra-node fabric.\n"),
    }
    boundary = ("Boundary condition: TP scales only as far as the interconnect allows. Over NVLink/NVSwitch the per-layer "
                "all-reduce is usually amortized; over PCIe (typical for A30 nodes without NVLink bridges) or across nodes over "
                "RoCE/InfiniBand, the all-reduce latency floor (single-digit to tens of microseconds per collective, times "
                "2 x num_layers per token) can dominate decode time and make TP=8 slower than TP=2. TP degree must also divide "
                "the attention head count evenly.\n")
    falsifiable = ("Falsifiable claim to test before adopting a degree: 'increasing TP from 2 to 4 improves p50 inter-token "
                   "latency by >=15% at fixed batch size'. If measured gain is below that, the collective cost is dominating and "
                   "the higher degree should be rejected.\n")
    ev = ("Evidence required: NCCL all-reduce bandwidth/latency for the actual message sizes (nccl-tests all_reduce_perf), "
          "the real topology (nvidia-smi topo -m) to confirm NVLink vs PCIe vs QPI paths, per-token latency breakdown at "
          "TP in {1,2,4,8}, and HBM occupancy for weights vs KV at each degree. Numbers above for the 9B/A30 case are estimates "
          "from parameter counts, not measurements; they must be confirmed on the target node.\n"
          "Rollback gate: revert to the previous TP degree if p99 inter-token latency regresses, if NCCL timeouts/hangs appear "
          "in logs, or if any rank shows >10% straggler skew in collective wait time.")
    return base + why[variant] + boundary + falsifiable + ev


ANSWERS = {
    "corpus-00191": exp_answer(1),
    "corpus-00192": exp_answer(2),
    "corpus-00193": exp_answer(3),
    "corpus-00194": exp_answer(4),
    "corpus-00195": exp_answer(5),
    "corpus-00197": runbook_answer(2),
    "corpus-00199": runbook_answer(4),
    "corpus-00200": runbook_answer(5),
    "corpus-00201": tp_answer(1),
    "corpus-00202": tp_answer(2),
}

RISKS_EXP = [
    "Source answer states only a single true sentence and does not design any experiment, so it fails the explicit instruction.",
    "No control variables, no metric definition, and no sample-size/repetition guidance; a reader could run an uncontrolled A/B and draw a false causal conclusion.",
    "Omits the KV-cache-saturation boundary where continuous batching provides no benefit, which can lead to tuning the wrong subsystem.",
]
RISKS_RUNBOOK = [
    "Source answer is a definition, not a runbook entry; it gives an on-call engineer no ordered diagnostic steps.",
    "No rollback gate or change-control guidance, so an engineer could change scheduler knobs in production without a revert criterion.",
    "Does not separate admission latency from prefill and decode time, the most common misattribution in continuous-batching incidents.",
]
RISKS_TP = [
    "Source answer is directionally correct but abstract: no concrete sharding mechanism, no communication volume, no numeric boundary.",
    "Does not state the interconnect dependence (NVLink vs PCIe vs cross-node RoCE), which is the dominant factor in choosing TP degree.",
    "Omits the constraint that TP degree must divide the attention head count, a common deployment-time failure.",
]

EV_EXP = [
    "Measured load-latency curves (throughput vs p50/p99 TTFT and TPOT) for each scheduling arm on the target hardware.",
    "Per-iteration scheduler telemetry: running batch size, waiting queue depth, KV cache utilization, preemption count.",
    "Output-length distribution of the replayed workload, to justify the dispersion assumption.",
    ">=5 independent repetitions with confidence intervals, plus GPU clock/temperature logs to rule out thermal drift.",
]
EV_RUNBOOK = [
    "Server configuration dump (max_num_seqs, max_num_batched_tokens, chunked prefill, preemption mode) and version snapshot.",
    "A >=10-minute /metrics window at 1 s resolution under comparable load, before and after any change.",
    "Request-level timestamps allowing admission / prefill / decode latency decomposition.",
    "Confirmation of GPU exclusivity (no MIG/MPS/co-tenant) and stable SM clocks.",
]
EV_TP = [
    "nvidia-smi topo -m output for the actual node to establish NVLink vs PCIe paths.",
    "nccl-tests all_reduce_perf results at the message sizes implied by batch x seq x hidden x dtype.",
    "Measured per-token latency and throughput at TP in {1,2,4,8} on the target model and hardware.",
    "HBM accounting for weights vs KV cache at each TP degree; model head count to validate divisibility.",
]

def main():
    lines = CORPUS.read_text().splitlines()
    sel = lines[START - 1:END]
    assert len(sel) == 10, len(sel)
    out = []
    for raw in sel:
        d = json.loads(raw)
        sid = d["id"]
        msgs = d["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        assistant = next(m["content"] for m in msgs if m["role"] == "assistant")
        if sid in ("corpus-00201", "corpus-00202"):
            risks, ev, qd, conf = RISKS_TP, EV_TP, {"technical_correctness": 4, "instruction_coverage": 2, "operational_safety": 3}, 0.72
        elif sid in ("corpus-00197", "corpus-00199", "corpus-00200"):
            risks, ev, qd, conf = RISKS_RUNBOOK, EV_RUNBOOK, {"technical_correctness": 4, "instruction_coverage": 1, "operational_safety": 2}, 0.74
        else:
            risks, ev, qd, conf = RISKS_EXP, EV_EXP, {"technical_correctness": 4, "instruction_coverage": 1, "operational_safety": 3}, 0.73
        rec = {
            "source_id": sid,
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": user,
            "source_assistant": assistant,
            "corrected_answer": ANSWERS[sid],
            "quality_dimensions": qd,
            "risks": risks,
            "evidence_required": ev,
            "confidence": conf,
        }
        out.append(json.dumps(rec, ensure_ascii=False))
    OUT.write_text("\n".join(out) + "\n")
    print("WROTE", OUT, len(out))

if __name__ == "__main__":
    main()

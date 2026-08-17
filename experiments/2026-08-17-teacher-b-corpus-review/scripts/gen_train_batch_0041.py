import json

SRC = "research/ai-infra-expert/corpus/train.jsonl"
OUT = "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0041.jsonl"
START, END = 400, 410

rows = [json.loads(l) for l in open(SRC)][START:END]

NCCL_RUNBOOK = """Runbook: NCCL collective hang or throughput regression.

Assumptions: PyTorch + NCCL >= 2.18, one process per GPU, homogeneous fabric. All numbers below are thresholds to measure, not vendor-published facts.

Step 1 - classify the symptom. Hang (no progress, watchdog fires) vs. slow (progress, low busbw). These have disjoint root-cause sets; do not mix them.

Step 2 - concrete mechanism to lean on. NCCL builds a ring/tree over discovered transports at communicator init and then runs collectives as fixed step sequences. Ring allreduce transfers 2*(N-1)/N * S bytes per rank in 2*(N-1) steps (S = buffer bytes, N = ranks). Because every step is a synchronous dependency, one slow or dead rank stalls all ranks; the reported failing rank is usually the victim, not the cause. So always collect the step/iteration counter from every rank, not just from the one that raised.

Step 3 - bisect by layer, in this order: (a) topology - nvidia-smi topo -m, confirm expected NVLINK/PIX/SYS tiers; (b) transport - NCCL_DEBUG=INFO, read the "via NET/IB" or "via P2P/NVLink" selection lines, confirm NCCL_IB_HCA / NCCL_SOCKET_IFNAME resolve to the intended NICs; (c) process group - world size, rank map, and that every rank calls the same collectives in the same order; (d) workload - variable-length batches or data-dependent branches causing rank divergence.

Step 4 - boundary condition. The default watchdog (TORCH_NCCL_BLOCKING_WAIT / desync timeout, commonly 1800 s) is only meaningful if the slowest legitimate collective is far below it. If a single allreduce over S >= 10 GB on a degraded link can legitimately exceed the timeout, the timeout itself manufactures "hangs". Compute expected time = 2*(N-1)/N * S / B_eff before concluding deadlock.

Step 5 - falsifiable check. Run all_reduce_perf -b 8 -e 1G -f 2 -g 1 across the same rank set. Prediction: if the problem is fabric, nccl-tests reproduces the low busbw with no framework involved; if nccl-tests is clean, the cause is above NCCL (data loader, rank divergence, CPU contention).

Evidence required to close: NCCL_DEBUG=INFO logs from all ranks, nccl-tests algbw/busbw before and after, nvidia-smi topo -m, ib link counters (ibqueryerrors / ethtool -S) showing no growing symbol or retransmit errors, and per-rank iteration counters at the moment of the stall.

Safety and rollback: apply one environment change at a time; never leave NCCL_P2P_DISABLE=1 or NCCL_IB_DISABLE=1 in production as a fix - they are diagnostics that trade throughput for reachability. Rollback threshold: if a tuning change does not improve end-to-end step time by >= 3% over 3 runs, revert to defaults and re-open the topology hypothesis."""

SPEC_DEF = """Definition. Speculative decoding is a lossless latency optimization for autoregressive generation: a cheap draft proposes k candidate tokens, and the expensive target model verifies all k in a single batched forward pass, accepting the longest prefix consistent with the target distribution.

Concrete mechanism. Because verification is one forward pass over k+1 positions, and decode on a GPU is memory-bandwidth-bound (per step you must read the full weight set plus KV cache, while the FLOPs for one token are trivial), the target's cost for k+1 positions is nearly the same wall-clock as for 1 position. That is exactly the arithmetic-intensity headroom speculative decoding converts into throughput. Modified rejection sampling (accept token i with probability min(1, p_target/p_draft), else resample from the normalized residual) makes the output distribution provably identical to plain sampling from the target, so this is a systems optimization, not a quality trade.

Why it matters in LLM infrastructure. Interactive serving is dominated by TPOT (time per output token), which is latency-bound rather than compute-bound at low-to-moderate batch size. Speculative decoding is one of the few techniques that reduces TPOT without changing the model's outputs.

Boundary condition. Expected speedup is roughly (1 + acceptance_rate_sum) / (1 + k * c), where c is draft cost relative to target cost. It collapses in two regimes: (1) low acceptance - a domain-shifted or too-small draft yields acceptance below roughly 0.5-0.6 per token and the wasted draft plus verify work makes it a net loss; (2) high batch size - once the server is already compute-saturated (large continuous batches), the spare FLOPs the technique exploits no longer exist and speculative decoding can reduce aggregate throughput even while improving single-request latency.

Falsifiable prediction and evidence required. Benchmark at fixed input/output lengths and sweep concurrency 1, 8, 32, 128 with and without speculation. Prediction: TPOT improves at concurrency 1-8 and the advantage shrinks or inverts by concurrency >= 64. Evidence needed: measured per-token acceptance rate, draft/target cost ratio, TTFT and TPOT percentiles (p50/p95), aggregate output tokens/s, and an output-equality or distribution check confirming losslessness.

Rollback threshold: disable speculation if p95 TPOT does not improve by >= 10% at the production concurrency, or if aggregate tokens/s regresses at all."""

SPEC_CONTRAST = """Contrast: speculative decoding vs. naive autoregressive decoding.

Naive baseline. One target forward pass per output token. Cost per token = reading all weights plus the KV cache from HBM. At low batch size this is memory-bandwidth-bound with arithmetic intensity near 1 FLOP/byte, so the GPU's tensor cores sit largely idle; TPOT is essentially (bytes moved per step) / (achieved HBM bandwidth), and it is nearly independent of how much compute the accelerator has.

Speculative variant. A cheap draft (small model, EAGLE/Medusa head, or n-gram/prompt-lookup) proposes k tokens; the target verifies positions 1..k+1 in one batched pass. Mechanism that makes this work: verifying k+1 positions moves the same weight bytes as verifying 1, so the marginal cost of extra positions is FLOPs the naive path was already wasting. Modified rejection sampling keeps the output distribution identical to the target's, so unlike quantization or a smaller model this is not a quality/latency trade.

Where they diverge quantitatively. Naive TPOT is flat in k. Speculative expected TPOT is target_step_time * (1 + k*c) / (1 + E[accepted]), with c the draft-to-target cost ratio. Break-even requires E[accepted] > k*c. With a draft costing c = 0.05 and k = 4, you need to accept about 0.2 tokens on average - easy; but the same draft at c = 0.3 needs 1.2 accepted tokens, which a domain-shifted draft often misses.

Boundary condition where naive wins. At high concurrency the server is already compute-bound: continuous batching has filled the FLOP headroom, so the extra verify positions and rejected draft tokens are real work, not free work. Past that crossover, speculative decoding lowers aggregate tokens/s while barely helping single-stream latency. It also loses on very short outputs, where draft warm-up and the extra KV bookkeeping are not amortized.

Falsifiable test. Fix prompt/output lengths, sweep concurrency 1, 8, 32, 128, run both configurations, and record per-token acceptance rate, TTFT, p50/p95 TPOT, and aggregate tokens/s. Prediction: speculation wins on TPOT at concurrency <= 8 and the aggregate-throughput curves cross somewhere in 16-64 concurrency on a bandwidth-limited accelerator.

Evidence required: measured acceptance rate (not assumed), draft cost fraction, both latency percentiles, aggregate throughput, and a losslessness check (greedy output equality, or a distribution/KL check under sampling). Rollback threshold: revert to naive decoding if aggregate tokens/s regresses at production concurrency or p95 TPOT gain is < 10%."""

RISK_STUB = "source_assistant is a one-line generic stub: it states a taxonomy/definition but does not supply the requested mechanism, boundary condition, quantities, or evidence, so it under-specifies the instruction"
RISK_UNSAFE = "no operational guardrails: no rollback threshold, no one-change-at-a-time discipline, and no warning that diagnostic env vars (NCCL_P2P_DISABLE / NCCL_IB_DISABLE) must not be left enabled in production"
RISK_UNFALS = "no falsifiable prediction or measurement plan, so a reader cannot distinguish a correct diagnosis from a plausible-sounding one"
RISK_REGIME = "omits the regime dependence (batch size / message size), which is where the naive intuition actually breaks and where a wrong decision costs throughput"

EV_NCCL = [
    "NCCL_DEBUG=INFO logs from all ranks, including transport and algorithm selection lines",
    "nccl-tests all_reduce_perf algbw/busbw sweep over the same rank set",
    "nvidia-smi topo -m and NIC/HCA mapping (NCCL_IB_HCA, NCCL_SOCKET_IFNAME)",
    "per-rank iteration counters at the moment of the stall, plus link error counters (ibqueryerrors / ethtool -S)",
]
EV_SPEC = [
    "measured per-token acceptance rate for the specific draft/target pair and workload",
    "draft-to-target cost ratio measured on the serving hardware",
    "TTFT and p50/p95 TPOT plus aggregate tokens/s at concurrency 1, 8, 32, 128",
    "losslessness check: greedy output equality or a distribution/KL comparison against non-speculative decoding",
]

def answer_for(user):
    if "NCCL" in user:
        return NCCL_RUNBOOK, EV_NCCL, [RISK_STUB, RISK_UNSAFE, RISK_UNFALS]
    if "Contrast speculative decoding" in user:
        return SPEC_CONTRAST, EV_SPEC, [RISK_STUB, RISK_REGIME, RISK_UNFALS]
    return SPEC_DEF, EV_SPEC, [RISK_STUB, RISK_REGIME, RISK_UNFALS]

out = []
for r in rows:
    msgs = {m["role"]: m["content"] for m in r["messages"]}
    user = msgs["user"]
    assistant = msgs["assistant"]
    ans, ev, risks = answer_for(user)
    is_nccl = "NCCL" in user
    out.append({
        "source_id": r["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": user,
        "source_assistant": assistant,
        "corrected_answer": ans,
        "quality_dimensions": {
            "technical_correctness": 3,
            "instruction_coverage": 1,
            "operational_safety": 2 if is_nccl else 3,
        },
        "risks": risks,
        "evidence_required": ev,
        "confidence": 0.74 if is_nccl else 0.78,
    })

with open(OUT, "w") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print("wrote", len(out), OUT)

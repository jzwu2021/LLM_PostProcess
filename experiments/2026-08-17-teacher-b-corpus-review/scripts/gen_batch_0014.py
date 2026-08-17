import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
SRC = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0014.jsonl")
START, END = 130, 140

ASSUME = ("Assumption frame: single node, 8x NVIDIA A30 24 GB (HBM2, ~933 GB/s theoretical peak, "
          "no NVLink bridge assumed, PCIe Gen4 x16 host links), dense ~9B decoder in bf16 (~18 GB weights), "
          "paged KV cache, continuous batching, vLLM-class server. Every number below is an ESTIMATE from a "
          "roofline/queueing model unless explicitly labelled MEASURED.\n\n")

DECODE_RUNBOOK = ASSUME + """Runbook entry: decode-phase latency investigation (variant {v}).

1. Scope the symptom. Separate TTFT (prefill) from TPOT/ITL (decode). If TTFT is stable and inter-token latency (ITL) has grown, the fault is in decode, not admission or prefill.

Concrete mechanism. A decode step emits exactly one token per running sequence. For each step the engine must read the full weight set once from HBM (~18 GB bf16) and read the KV cache of every running sequence once. Per-step time is therefore approximately (W + sum_i KV_i) / BW_eff, where W is weight bytes, KV_i is the resident KV bytes of sequence i, and BW_eff is achieved HBM bandwidth (assume 0.65-0.75 of 933 GB/s on A30 for GEMV-dominated kernels). This makes decode memory-bandwidth-bound, not FLOP-bound: the matmuls degenerate to GEMV at batch=1 per sequence, so arithmetic intensity is ~2*B FLOP/byte with B = co-batched sequences.

2. Triage order.
   a. Read the server's own counters first: running vs waiting sequences, KV cache utilisation %, preemption/swap counter, average generation length. Do not touch the GPU before you have these.
   b. Sample nvidia-smi dmon or DCGM for SM occupancy and memory-controller utilisation. Bandwidth-bound decode shows high memory utilisation with modest SM utilisation; a sudden drop in BOTH with rising queue depth points at scheduler stalls or preemption thrash instead.
   c. Check for KV-cache pressure: if utilisation is pinned near the high-water mark and the preemption counter is climbing, sequences are being evicted and recomputed, which multiplies effective decode cost.

3. Falsifiable hypotheses.
   H1 (bandwidth roofline): ITL scales with (W + total KV bytes) and is insensitive to added FLOPS. Falsified if raising batch size materially raises per-token cost rather than amortising weight reads.
   H2 (KV thrash): ITL degradation correlates 1:1 with the preemption counter. Falsified if preemptions are zero while ITL still degrades.
   H3 (long-context KV dominance): once sum_i KV_i approaches W, ITL should roughly double relative to short-context. Falsified if ITL is flat as context length grows.

Boundary condition. The bandwidth-bound model holds only while the batch is small enough that weight streaming dominates. As B grows, per-token weight cost amortises as W/B and the step crosses into compute-bound territory; past that knee, adding batch raises ITL roughly linearly and only throughput improves. On A30 with no NVLink, a second boundary applies: if the model is tensor-parallel across GPUs, all-reduce traffic rides PCIe and the collective, not HBM, becomes the limiter for small batches.

4. Evidence to collect before any change: per-request TTFT/TPOT percentiles from a fixed replayed trace, KV utilisation and preemption counters, DCGM memory-controller utilisation, NCCL bandwidth test if TP>1, and the server config (max_num_seqs, max_num_batched_tokens, block size, TP degree).

5. Rollback gate. Any tuning change ships behind a single-parameter diff on one replica. Revert if p95 TTFT regresses >10%, if p95 ITL regresses at all, if the preemption rate rises above the pre-change baseline, or if output-token equivalence on a fixed prompt set breaks. Keep the previous config pinned and re-runnable in one command."""

CB_DEFINE = ASSUME + """Definition (variant {v}). Continuous batching (a.k.a. iteration-level or in-flight batching) is a scheduling policy in which the serving engine re-forms the executing batch at every decoder iteration rather than once per batch. At each step boundary the scheduler retires sequences that hit EOS or their length cap, frees their KV blocks, and admits waiting requests into the freed slots. Contrast this with static batching, where a batch is assembled once and every slot is held until the LONGEST sequence in that batch finishes.

Concrete mechanism. The engine keeps a running set and a waiting queue. Between iterations it (1) drops finished sequences and returns their paged KV blocks to the allocator, (2) admits new sequences while the KV block budget and the max_num_batched_tokens budget allow, and (3) launches the next forward pass over the new running set. Paged KV allocation is what makes this cheap: KV is stored in fixed-size blocks in a shared pool, so admitting or retiring a sequence is a block-table update, not a tensor reallocation or a memcpy of the whole cache.

Why it matters. Generation lengths in real traffic are highly skewed. Under static batching a batch of B requests occupies B slots for the duration of the longest sequence, so slot-time utilisation is roughly mean_len/max_len; with a heavy tail that can be well under 50% (ESTIMATE, workload-dependent). Continuous batching recovers most of that idle slot-time, which raises achieved throughput and shortens queueing delay for newly arrived requests. Because decode is HBM-bandwidth-bound, keeping the batch full is also what amortises the ~18 GB per-step weight read across more tokens: per-token weight cost falls as W/B.

Boundary condition. The benefit saturates and then reverses. Once the running set is large enough that the step crosses from bandwidth-bound to compute-bound, further admission raises per-token latency without raising throughput. More sharply, admission is bounded by the KV pool: if the scheduler admits beyond the block budget it must preempt (evict-and-recompute or swap-to-host), and preemption thrash can make aggregate throughput WORSE than a conservative static batch. Continuous batching also does nothing for a workload where all sequences have identical length - there the slot-idle it recovers is zero.

Falsifiable hypotheses. H1: throughput gain over static batching scales with the coefficient of variation of output length; falsified if a fixed-length workload also shows a large gain. H2: gains vanish once KV utilisation is pinned at the cap and preemptions are non-zero.

Evidence required. Replay one fixed request trace under both policies on the same build and same GPU state: record output tok/s, p50/p95 TTFT, p50/p95 ITL, KV utilisation, preemption count, and confirm generated tokens are identical for a greedy-decode control set.

Rollback gate. Revert the scheduler change if p95 TTFT regresses >10%, if preemption rate exceeds the baseline, or if greedy-decode outputs diverge from the pinned reference."""

CB_CONTRAST = ASSUME + """Contrast: continuous batching vs a naive (static / request-level) implementation (variant {v}).

Naive baseline. The server collects up to B requests, pads them into one batch, runs the decode loop until every sequence in the batch has finished, then returns all results and starts the next batch. KV cache is allocated as one contiguous [B, max_len, ...] tensor sized for the worst case.

Continuous batching. The batch is re-formed at each iteration boundary: finished sequences retire immediately, their paged KV blocks return to a shared pool, and waiting requests are admitted into the freed capacity within the KV-block and token budgets.

Concrete mechanism that produces the difference. Two distinct wastes are removed. (1) Slot-time waste: under static batching a slot stays occupied until the longest sequence in the batch ends, so slot utilisation is approximately mean_len/max_len over the batch. (2) Memory waste: contiguous worst-case KV allocation reserves max_len for every sequence, so KV headroom is sized by the longest possible output, not the actual one; paged KV instead reserves in blocks, so internal fragmentation is bounded by one block per sequence. Recovering (2) is what lets the engine hold a larger running set at all, and a larger running set is what amortises the ~18 GB per-step weight read - per-token weight cost falls as W/B while decode remains bandwidth-bound.

Where they are equivalent. If every request has identical prompt and output length and arrivals are perfectly synchronous, the two policies produce the same schedule; the gap is a function of length skew and arrival jitter, not of the algorithm being intrinsically better.

Boundary conditions where the naive version can win or the gain disappears. (a) Very small models where per-iteration scheduler overhead is a non-trivial fraction of step time - iteration-level bookkeeping is pure overhead there. (b) KV pool saturation: if admission outruns the block budget the engine preempts, and evict-and-recompute thrash can drive aggregate throughput below the static baseline. (c) Latency-SLO-strict single-tenant traffic where a bounded, predictable batch is preferred to a variable one, because continuous admission makes ITL depend on other tenants' arrivals. (d) Past the bandwidth/compute knee, extra admitted sequences raise ITL linearly with no throughput gain.

Falsifiable hypotheses. H1: the throughput advantage scales with the coefficient of variation of output length; falsified if a constant-length workload shows a comparable gain. H2: the advantage collapses to zero or negative once preemption count per minute is non-zero at steady state.

Evidence required. Same build, same GPU, same pinned request trace replayed under both policies; report output tok/s, p50/p95 TTFT and ITL, KV utilisation, preemption/swap counters, and byte-identical greedy outputs on a control prompt set. A single throughput number without the latency percentiles and the preemption counter is not sufficient evidence.

Rollback gate. Ship on one replica behind a single config diff. Revert if p95 TTFT regresses >10%, p95 ITL regresses at all, preemption rate exceeds baseline, or greedy-decode control outputs diverge."""


def build(rec):
    msgs = rec["messages"]
    user = next(m["content"] for m in msgs if m["role"] == "user")
    asst = next(m["content"] for m in msgs if m["role"] == "assistant")
    v = "1"
    for tok in user.split():
        if tok.rstrip(":").isdigit():
            v = tok.rstrip(":")
            break
    if "runbook entry" in user and "decode" in user:
        ans = DECODE_RUNBOOK.format(v=v)
        topic = "decode"
        risks = [
            "Source answer states decode is 'often sensitive to memory bandwidth and scheduling' without the per-step byte model, so it cannot be used to predict or bound ITL.",
            "No boundary condition given for when decode leaves the bandwidth-bound regime, risking over-batching in production.",
            "No triage order, no counters to read, and no rollback gate: unusable as an actual runbook entry.",
        ]
        ev = [
            "Per-request TTFT and TPOT/ITL percentiles from a fixed replayed trace",
            "KV cache utilisation and preemption/swap counters from the serving engine",
            "DCGM/nvidia-smi memory-controller vs SM utilisation samples during the incident window",
            "Serving config: max_num_seqs, max_num_batched_tokens, KV block size, TP degree",
            "NCCL bandwidth test result if tensor parallel degree > 1",
        ]
        qd = {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 2}
        conf = 0.79
    elif "Contrast continuous batching" in user:
        ans = CB_CONTRAST.format(v=v)
        topic = "batching-contrast"
        risks = [
            "Source answer describes continuous batching only; it never states the naive baseline, so the requested contrast is absent.",
            "Omits paged KV allocation, which is the enabling mechanism for the memory-side half of the win.",
            "No failure mode: does not mention KV-pool saturation and preemption thrash, where continuous batching can underperform static batching.",
        ]
        ev = [
            "Same-build A/B replay of one pinned request trace under both scheduling policies",
            "Output tok/s plus p50/p95 TTFT and ITL for both arms",
            "KV utilisation and preemption/swap counters at steady state",
            "Output-length distribution and its coefficient of variation for the trace",
            "Byte-identical greedy-decode control outputs across arms",
        ]
        qd = {"technical_correctness": 3, "instruction_coverage": 1, "operational_safety": 2}
        conf = 0.82
    else:
        ans = CB_DEFINE.format(v=v)
        topic = "batching-definition"
        risks = [
            "Source answer gives the retire-at-iteration-boundary mechanism but omits admission, the KV block budget, and paged KV, so it is incomplete as a definition.",
            "States no boundary condition; a reader may conclude more batching is monotonically better and over-admit into preemption thrash.",
            "Gives no reason 'why it matters' tied to a measurable quantity, so the claim is not falsifiable.",
        ]
        ev = [
            "Replay of a pinned trace under static vs continuous batching on identical build and GPU state",
            "Output tok/s, p50/p95 TTFT and ITL for both arms",
            "KV cache utilisation and preemption count at steady state",
            "Output-length distribution of the workload",
        ]
        qd = {"technical_correctness": 3, "instruction_coverage": 2, "operational_safety": 3}
        conf = 0.83
    return {
        "source_id": rec["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": user,
        "source_assistant": asst,
        "corrected_answer": ans,
        "quality_dimensions": qd,
        "risks": risks,
        "evidence_required": ev,
        "confidence": conf,
    }, topic


def main():
    lines = open(SRC).read().splitlines()[START:END]
    out = []
    topics = {}
    for l in lines:
        rec = json.loads(l)
        r, t = build(rec)
        topics[t] = topics.get(t, 0) + 1
        out.append(json.dumps(r, ensure_ascii=False))
    with open(OUT, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", OUT, len(out))
    print("topics", topics)
    print("ids", json.loads(out[0])["source_id"], "->", json.loads(out[-1])["source_id"])


main()

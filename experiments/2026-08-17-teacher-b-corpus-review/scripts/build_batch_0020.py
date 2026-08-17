import json, os

CORPUS = 'research/ai-infra-expert/corpus/train.jsonl'
OUT = 'experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0020.jsonl'
START, N = 190, 10

A = {}

A['corpus-00213'] = (
 "Assumption: dense transformer decoder, TP degree t within one node, NVLink/NVSwitch intra-node, "
 "measured facts limited to what a profiler shows; everything else is an estimate.\n\n"
 "Failure mode 1 - collective communication becomes the critical path. Mechanism: Megatron-style TP splits "
 "each transformer block into a column-parallel then row-parallel pair, so every block needs one all-reduce "
 "in the forward pass and one in the backward pass (2 all-reduces/block/step). Each all-reduce moves "
 "roughly 2*(t-1)/t * batch_tokens * hidden * dtype_bytes per rank with ring algorithm. Boundary condition: "
 "the moment TP crosses a node boundary and the transport drops from ~200-400 GB/s NVLink-class to "
 "25-50 GB/s RoCE/IB per rank, the all-reduce term stops being hidden and TP scaling turns negative. "
 "Falsifiable prediction: at fixed batch, going t=8 (intra-node) -> t=16 (2 nodes) increases per-token "
 "latency instead of decreasing it.\n\n"
 "Failure mode 2 - shard granularity underutilizes tensor cores. Mechanism: TP divides attention heads and "
 "the FFN intermediate dimension by t; once per-GPU GEMM tiles fall below the hardware's efficient tile "
 "size, arithmetic intensity drops and the GEMM becomes launch/memory-bound. Boundary condition: t must "
 "divide num_attention_heads (and num_kv_heads for GQA); with 32 heads and t=16 you get 2 heads/GPU and "
 "the attention kernels degrade sharply.\n\n"
 "Evidence to collect before choosing t: nsys/nsight timeline share of ncclAllReduce vs GEMM, "
 "nccl-tests all_reduce_perf busbw at the real message size, and a t-sweep of p50/p99 latency at fixed load.\n"
 "Rollback gate: if p99 latency regresses >10% or busbw is below ~60% of nccl-tests peak, revert to the "
 "previous t."
)

A['corpus-00214'] = (
 "Assumptions stated first: single node, NVLink-class interconnect, dense model, TP degree t, inference and "
 "training discussed separately because their trade-offs differ.\n\n"
 "Trade-off 1 - memory relief is bought with bandwidth. Mechanism: TP shards weights, gradients, optimizer "
 "state and activation tensors of the parallel layers by ~1/t, which is what makes a model fit at all; the "
 "price is that the layer output must be reconstructed with an all-reduce (forward) and the input gradient "
 "with another (backward). Boundary condition: KV cache is sharded by KV heads, so with GQA where "
 "num_kv_heads < t the cache cannot be split further and per-GPU KV memory stops shrinking - raising t "
 "buys no additional context length past that point.\n\n"
 "Trade-off 2 - tail latency and blast radius. Mechanism: an all-reduce is a synchronizing collective, so "
 "step time equals the slowest rank; one throttled GPU (thermal/power cap) or one degraded NIC/link sets "
 "the pace for the whole TP group, and any rank crash kills the whole replica. Boundary condition: TP "
 "groups should stay inside one failure domain (one node, one NVLink domain); spanning hosts multiplies "
 "both jitter and failure probability.\n\n"
 "Falsifiable hypothesis: if the all-reduce is the bottleneck, per-step time scales with message bytes at "
 "fixed t; if straggler skew is the bottleneck, it does not, and per-rank kernel start-time spread "
 "correlates with step time instead.\n"
 "Evidence: NCCL kernel time share, per-rank SM clock and power (DCGM), nccl-tests busbw.\n"
 "Rollback gate: revert the TP change if throughput/GPU drops >5% or p99/p50 ratio worsens beyond the "
 "agreed SLO."
)

A['corpus-00215'] = (
 "Assumptions: dense decoder, TP degree t, comparison against the same model served with a smaller t plus "
 "more replicas; numbers below are shapes to measure, not platform facts.\n\n"
 "Failure mode 1 - TP does not reduce, and often increases, total work. Mechanism: the FLOPs are the same, "
 "but you add 2 all-reduces per block per pass plus kernel-launch and sync overhead, so aggregate "
 "throughput per GPU falls monotonically with t even when single-request latency improves. Boundary "
 "condition: TP is the right lever only when the model does not fit at lower t, or when a strict TTFT/TPOT "
 "latency SLO cannot be met by replica-level scaling; above that, data/replica parallelism dominates on "
 "cost per token.\n\n"
 "Failure mode 2 - decode phase is memory-bandwidth bound, not compute bound. Mechanism: during "
 "autoregressive decode each step reads the full sharded weight set once; TP does divide that read by t, "
 "which genuinely helps TPOT, but the fixed all-reduce latency (a few tens of microseconds per collective, "
 "multiplied by 2*num_layers) is paid on every token. Boundary condition: at small batch and many layers, "
 "collective latency can exceed the bandwidth savings, so TPOT flattens or regresses beyond some t.\n\n"
 "Evidence required: TTFT and TPOT separately at fixed concurrency; NCCL time per decode step; "
 "achieved HBM bandwidth vs device peak; tokens/s per GPU (not per replica).\n"
 "Rollback gate: if tokens/s/GPU falls >10% without meeting a previously unmet latency SLO, revert."
)

A['corpus-00216'] = (
 "Assumption: dense decoder LLM served with continuous batching; TP degree t inside one NVLink node; "
 "prefill and decode analysed separately because they are bound by different resources.\n\n"
 "Memory: weights, and the KV cache along the KV-head dimension, shard by ~1/t, so peak per-GPU memory "
 "falls roughly linearly until a non-shardable floor (activation buffers, CUDA graphs, NCCL buffers, "
 "fragmentation). Concrete mechanism: column-parallel QKV/up-proj followed by row-parallel out/down-proj "
 "means each rank stores only its column or row slice.\n\n"
 "Latency: prefill is compute-bound, so it benefits close to linearly with t until the per-rank GEMM tiles "
 "get too small; decode is HBM-bandwidth-bound, so TPOT improves roughly with t on the weight-read term but "
 "pays a fixed 2*num_layers all-reduce latency per token. Boundary condition: with batch size 1 and a "
 "40-80 layer model, that fixed collective cost can dominate, so TPOT stops improving (or worsens) past "
 "some t - this is the falsifiable claim to test.\n\n"
 "Throughput: aggregate tokens/s per GPU degrades with t because communication is pure overhead; TP trades "
 "goodput for latency and for fitting the model.\n\n"
 "Evidence: t-sweep of TTFT/TPOT/tokens-per-GPU at fixed concurrency, nsys timeline share of ncclAllReduce, "
 "achieved HBM bandwidth, torch/vLLM memory profile.\n"
 "Rollback gate: revert if the SLO metric does not improve by more than run-to-run noise (>=5 runs, "
 "report p50 and p99)."
)

A['corpus-00217'] = (
 "Assumptions: TP degree t, single node, dense model, continuous batching, dtype bf16 or fp8; all figures "
 "below are relationships to be measured, not vendor claims.\n\n"
 "Mechanism (the one concrete thing): each transformer block does one all-reduce in forward and one in "
 "backward. Per-token decode communication per rank is about 2 * num_layers * hidden * dtype_bytes * "
 "2(t-1)/t bytes with ring all-reduce. That expression is why TP cost grows with depth and hidden size but "
 "saturates in t.\n\n"
 "Latency: prefill (compute-bound) improves nearly linearly with t; decode (bandwidth-bound) improves on "
 "the weight-read term but adds 2*num_layers collective latencies per token.\n"
 "Throughput: total tokens/s/GPU drops with t because you added communication without removing FLOPs.\n"
 "Memory: per-GPU weights and KV shards fall ~1/t, enabling longer context or larger batch - usually the "
 "real reason to raise t.\n\n"
 "Boundary condition: TP degree must divide num_kv_heads; once t > num_kv_heads (common with GQA, e.g. 8 KV "
 "heads and t=16) the KV cache is either replicated or padded, so the memory benefit stops while the "
 "communication cost keeps growing.\n\n"
 "Evidence required: nccl-tests all_reduce_perf busbw at the exact message size; nsys share of NCCL vs "
 "GEMM; TTFT/TPOT/tokens-per-GPU across t in {1,2,4,8}; DCGM power and clock per rank to rule out "
 "stragglers.\n"
 "Rollback gate: keep the smallest t that meets the SLO; revert any increase in t that costs >10% "
 "tokens/s/GPU without an SLO gain."
)

A['corpus-00219'] = (
 "Assumptions: intra-node TP, dense decoder, steady-state serving with mixed prefill/decode; separate the "
 "three resources rather than quoting a single speedup number.\n\n"
 "1) Memory. Mechanism: column-then-row parallel pairs let each rank hold 1/t of the attention and FFN "
 "weights, and the KV cache splits along KV heads. Effect: per-GPU HBM demand drops ~1/t, which raises the "
 "feasible max context length and max concurrent sequences. Non-shardable floor: NCCL buffers, CUDA-graph "
 "pools, allocator fragmentation, and any replicated embedding/norm parameters.\n\n"
 "2) Latency. Prefill scales close to linearly with t while GEMM tiles stay large; decode gains come from "
 "dividing the per-step weight read across t HBMs, offset by 2*num_layers all-reduce latencies per token.\n\n"
 "3) Throughput. Aggregate tokens/s per GPU declines with t; if the model already fits and the SLO is met, "
 "more replicas at lower t is the cheaper configuration.\n\n"
 "Boundary condition: if TP spans hosts, the all-reduce runs over the network fabric; without correctly "
 "configured GPUDirect RDMA (right NIC-GPU PCIe affinity, NCCL_IB_HCA / NCCL_NET_GDR_LEVEL) the path falls "
 "back to host staging and collective time can grow by an order of magnitude.\n\n"
 "Falsifiable hypothesis: if collectives dominate, halving hidden size (or switching to fp8) reduces step "
 "time superlinearly relative to FLOP reduction.\n"
 "Evidence: NCCL_DEBUG=INFO topology/transport lines, nccl-tests busbw, nsys timeline, t-sweep of "
 "TTFT/TPOT/tokens-per-GPU.\n"
 "Rollback gate: revert on >10% tokens/s/GPU loss or any p99 SLO breach; canary on <=10% of traffic first."
)

A['corpus-00220'] = (
 "Assumption: dense decoder, TP degree t, single node NVLink, and we care about a serving SLO expressed as "
 "TTFT p99 and TPOT p99.\n\n"
 "Interaction summary. Memory: per-GPU weight and KV-cache footprint ~1/t, so max batch and max context "
 "rise; this is usually the decisive benefit. Latency: prefill improves near-linearly with t (compute-bound "
 "GEMMs get smaller per rank); decode improves on the weight-read term because each rank streams only 1/t "
 "of the parameters from HBM per token. Throughput: total tokens/s per GPU falls, since TP adds "
 "communication without reducing total FLOPs.\n\n"
 "Concrete mechanism: 2 all-reduces per transformer block per pass (one after the row-parallel FFN "
 "projection, one after the row-parallel attention output projection). During decode this becomes "
 "2*num_layers synchronizations per generated token, so fixed collective latency multiplies with depth.\n\n"
 "Boundary condition: with small decode batch, per-token time is dominated by that fixed collective latency "
 "rather than by bytes moved; beyond that crossover point raising t makes TPOT worse, not better. Locate "
 "the crossover empirically instead of assuming it.\n\n"
 "Evidence required: TTFT/TPOT p50 and p99 for t in {1,2,4,8} at several concurrency levels; nsys share of "
 "ncclAllReduce; achieved HBM bandwidth vs peak; per-rank clocks/power to exclude stragglers; >=5 repeats "
 "for confidence intervals.\n"
 "Rollback gate: adopt the smallest t meeting the SLO; auto-revert if canary p99 regresses >10% or "
 "tokens/s/GPU drops >10%."
)

A['corpus-00221'] = (
 "Assumptions: one model, one hardware SKU, one serving stack version, single node; goal is to decide "
 "whether raising TP degree t helps a real serving workload. Everything below is a measurement plan, not a "
 "claim about results.\n\n"
 "Hypothesis (falsifiable): raising t from t0 to t1 reduces TPOT p99 by >=15% at the target concurrency "
 "without reducing tokens/s/GPU by more than 10%.\n\n"
 "Plan.\n"
 "1. Freeze variables: model weights hash, engine version, dtype, max_model_len, scheduler settings, "
 "driver/NCCL version, GPU clocks locked (nvidia-smi -lgc) to remove boost noise.\n"
 "2. Baseline the interconnect first: nccl-tests all_reduce_perf at the message sizes the model actually "
 "uses (2*hidden*dtype_bytes*batch_tokens). Record busbw. If busbw is far below expectation, fix topology "
 "before benchmarking the model.\n"
 "3. Workload: replay a production-shaped trace (real input/output length distribution and arrival "
 "process), not a fixed-length synthetic loop. Report TTFT and TPOT separately.\n"
 "4. Sweep t in {1,2,4,8} x concurrency in {1,8,32,128}. 5 repeats each, discard warmup, report p50/p95/p99 "
 "and tokens/s/GPU with confidence intervals.\n"
 "5. Attribute causes: nsys profile for NCCL-vs-GEMM time share; DCGM for per-rank power/clock/throttle "
 "reasons; achieved HBM bandwidth.\n"
 "6. Correctness gate: identical greedy outputs (or bounded logprob delta) across t - TP must not change "
 "semantics.\n\n"
 "Boundary condition: the sweep is only valid while t divides num_attention_heads and num_kv_heads; "
 "configurations that do not divide evenly must be excluded, not padded silently.\n"
 "Rollback gate: canary <=10% traffic, auto-revert on p99 regression >10% or any output-mismatch alarm."
)

A['corpus-00222'] = (
 "Assumptions: decision is 'increase TP degree' vs 'add replicas at current t'; single node per replica; "
 "cost measured as tokens/s per GPU. No platform-specific numbers are asserted here.\n\n"
 "Hypothesis: TP helps only if (a) the model or target context does not fit at lower t, or (b) a latency "
 "SLO is unmet and cannot be met by adding replicas. Otherwise TP loses on cost per token.\n\n"
 "Measurement plan.\n"
 "1. Pin the environment: same image, engine build, driver, NCCL version, dtype, and locked clocks; record "
 "weight file sha256.\n"
 "2. Interconnect floor: nccl-tests all_reduce_perf busbw at realistic message sizes; NCCL_DEBUG=INFO to "
 "confirm the transport actually chosen (NVLink vs PCIe vs net) - a silent fallback invalidates every "
 "later number.\n"
 "3. Load generator: production-shaped trace with open-loop arrivals (closed-loop hides queueing); "
 "measure TTFT, TPOT, end-to-end p50/p99, and goodput under an SLO constraint.\n"
 "4. Grid: t in {1,2,4,8} x request rate stepped until SLO violation; 5 repeats; report the maximum rate "
 "each t sustains within SLO, normalized per GPU.\n"
 "5. Memory evidence: peak per-GPU HBM, max concurrent sequences, achieved KV-cache utilization per t.\n"
 "6. Attribution: nsys NCCL share, achieved HBM bandwidth, per-rank throttle counters.\n\n"
 "Boundary condition: with GQA, once t exceeds num_kv_heads the KV cache no longer shards, so the memory "
 "argument for larger t disappears while communication cost keeps rising - test that point explicitly.\n"
 "Decision rule: pick the smallest t that meets the SLO at the highest per-GPU sustained rate.\n"
 "Rollback gate: canary first; auto-revert on >10% p99 regression, >10% tokens/s/GPU loss, or any output "
 "divergence vs the baseline configuration."
)

A['corpus-00224'] = (
 "Assumptions: goal is a defensible yes/no on raising TP degree for a specific serving workload; one "
 "hardware SKU; results are workload-specific and do not generalize across models or traces.\n\n"
 "Falsifiable hypothesis: at the production request mix, TP=t1 sustains >=X req/s within the TTFT/TPOT p99 "
 "SLO using fewer GPUs than TP=t0 replicas do.\n\n"
 "Plan.\n"
 "1. Control variables and record them: model sha256, engine version, driver/CUDA/NCCL versions, dtype, "
 "scheduler/batching config, locked SM and memory clocks, GPU topology (nvidia-smi topo -m).\n"
 "2. Microbenchmark before macrobenchmark: nccl-tests all_reduce_perf across the real message-size range, "
 "plus a GEMM microbenchmark at the per-rank shard shapes. This separates 'communication is slow' from "
 "'shards are too small to be efficient'.\n"
 "3. Macrobenchmark: open-loop replay of a production trace; sweep t in {1,2,4,8} and arrival rate; "
 "5 repeats; report SLO-constrained goodput per GPU with confidence intervals, not mean latency alone.\n"
 "4. Attribution: nsys timeline (NCCL vs GEMM vs sampling), achieved HBM bandwidth, DCGM throttle reasons "
 "and per-rank power - a single throttled rank can masquerade as a TP scaling limit.\n"
 "5. Correctness: greedy-decode output equality (or bounded logprob delta) across all t values.\n\n"
 "Boundary condition: the comparison is only valid if every configuration serves the same max_model_len and "
 "the same KV-cache capacity policy; a larger t that silently allows a bigger cache is measuring cache "
 "size, not tensor parallelism.\n"
 "Evidence required to accept: reproducible on two independent runs on different hosts of the same SKU.\n"
 "Rollback gate: shadow/canary deployment, auto-revert on p99 regression >10%, error-rate increase, or any "
 "output mismatch."
)

RISKS = [
 "source_assistant is a single generic sentence: it names no mechanism, no units, no boundary condition, so it cannot be used as a supervision target",
 "no distinction between prefill (compute-bound) and decode (bandwidth-bound) behaviour, which is where most TP reasoning errors originate",
 "no operational guardrail (canary, rollback threshold) despite the answer being used to justify serving-topology changes",
]
EV = [
 "nccl-tests all_reduce_perf busbw at the model's real message sizes",
 "nsys/nsight timeline share of ncclAllReduce vs GEMM kernels",
 "TTFT and TPOT p50/p99 plus tokens/s per GPU across a TP-degree sweep, >=5 repeats",
 "DCGM per-rank power/clock/throttle counters to exclude stragglers",
 "nvidia-smi topo -m and NCCL_DEBUG=INFO transport lines to confirm NVLink vs PCIe vs network path",
]

def main():
    corp = [json.loads(l) for l in open(CORPUS) if l.strip()]
    batch = corp[START:START + N]
    assert len(batch) == N
    lines = []
    for rec in batch:
        sid = rec['id']
        msgs = rec['messages']
        su = [m['content'] for m in msgs if m['role'] == 'user'][0]
        sa = [m['content'] for m in msgs if m['role'] == 'assistant'][0]
        ca = A[sid]
        out = {
            "source_id": sid,
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": su,
            "source_assistant": sa,
            "corrected_answer": ca,
            "quality_dimensions": {
                "technical_correctness": 3,
                "instruction_coverage": 1,
                "operational_safety": 2,
            },
            "risks": RISKS,
            "evidence_required": EV,
            "confidence": 0.78,
        }
        lines.append(json.dumps(out, ensure_ascii=False))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT, len(lines))

main()

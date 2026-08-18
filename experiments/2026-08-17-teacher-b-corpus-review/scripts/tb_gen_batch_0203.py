import json

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = f"{ROOT}/research/ai-infra-expert/corpus/train.jsonl"
OUT = f"{ROOT}/experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0203.jsonl"
START, N = 2020, 10  # 0-indexed positional slice -> lines 2021..2030

COMMON = (
"Common frame (applies to every stance below).\n"
"Assumptions (must be restated by the answering engineer, not inherited silently):\n"
"A1. Single node, 8 GPUs, NVLink/NVSwitch intra-node; inter-node only if TP is stretched across nodes (which we forbid).\n"
"A2. Decode-dominant, latency-sensitive serving: SLO expressed as TTFT p95 and TPOT (inter-token latency) p95, not mean.\n"
"A3. Model weights fit in aggregate HBM with KV-cache headroom >= 20% at target concurrency.\n"
"A4. No speculative decoding, no quantization change, no batching-policy change during the comparison; exactly one variable moves.\n"
"Mechanism, stated plainly:\n"
"- Tensor parallelism (TP) splits every layer's GEMMs across GPUs. Each transformer layer needs 2 collectives per block "
"(all-reduce after attention out-proj and after MLP down-proj). Per-token decode latency therefore carries "
"L * 2 * allreduce_latency of added synchronous cost, where L = number of layers. TP is latency-additive but "
"capacity-multiplying: it cuts per-GPU weight bytes and per-GPU KV bytes by TP degree, and it raises the effective "
"HBM bandwidth applied to a single token.\n"
"- Pipeline parallelism (PP) splits layers into stages. Per-token cost adds only (PP-1) point-to-point sends of one "
"hidden-state vector, which is tiny, but decode serializes through stages: a single request's token must traverse all "
"stages, and with micro-batching the pipeline bubble is (PP-1)/(micro_batches + PP-1). At low concurrency there are not "
"enough in-flight micro-batches to fill the pipeline, so PP bubbles dominate and PP is the worse latency choice.\n"
"Boundary conditions that flip the answer:\n"
"- B1. If interconnect is NVLink-class (hundreds of GB/s, single-digit microsecond small-message all-reduce), TP up to 8 "
"is normally latency-viable. If interconnect is PCIe-only or crosses nodes over RoCE/IB, TP all-reduce latency grows by "
"roughly an order of magnitude and TP stops paying for itself past TP=2 (ESTIMATE; derivation: decode all-reduce messages "
"are small, hidden_size * dtype_bytes per token per layer, so they are latency-bound not bandwidth-bound, and the latency "
"floor of a fabric hop is what matters).\n"
"- B2. If the model does not fit on one GPU, some sharding is mandatory; then the question is only which axis, not whether.\n"
"- B3. If concurrency is high and steady, PP's bubble amortizes and PP becomes competitive on throughput per GPU while "
"still losing on single-request latency.\n"
"Default recommendation: TP within the node up to the point where the collective cost stops being amortized by the "
"reduced per-GPU memory traffic; use PP only to cross node boundaries or to fit a model that TP alone cannot fit. "
"Do not use PP as a latency optimization.\n"
"Measurement and evidence policy: every number below that is not produced by a run on this hardware is labelled ESTIMATE "
"and carries its derivation; only numbers read out of the stated benchmark artifacts may be labelled MEASURED. This "
"review states no MEASURED values because no benchmark was executed for it.\n"
)

STANCES = [
 ("Stance 30 - Arithmetic intensity of decode is the root cause, parallelism only moves it.",
  "Frame the whole question through arithmetic intensity. Decode with batch size B does B rows of GEMV-like work against "
  "the full weight matrix, so FLOPs per byte loaded is O(B). Below the hardware's ridge point the kernel is purely "
  "HBM-bandwidth-bound and adding GPUs via TP helps only because it adds aggregate bandwidth, not because it adds FLOPs. "
  "PP adds neither bandwidth per token nor FLOPs per token for a single request; it only adds capacity. That is the "
  "structural reason TP is the latency axis and PP is the capacity axis. Falsifiable hypothesis H30: at concurrency 1 the "
  "achieved HBM bandwidth utilization during decode exceeds 60% while achieved FLOP utilization stays below 10% "
  "(ESTIMATE; derivation: arithmetic intensity of a batch-1 decode step is roughly 2 FLOPs per weight byte, far below the "
  "ridge point of any modern accelerator). Controlled experiment: profile a single decode step at batch 1, 8, 64 and read "
  "out DRAM throughput and tensor-core utilization counters. Evidence required: hardware counters for memory throughput "
  "and SM/tensor utilization, per kernel, not aggregate nvidia-smi utilization which is a duty-cycle metric and will "
  "mislead. Rollback gate: reject any parallelism change justified by 'GPU utilization went up' without a bandwidth "
  "counter backing it."),
 ("Stance 31 - Treat the choice as a queueing problem, not a kernel problem.",
  "Model the service as a queue: TPOT sets service time per token, concurrency sets arrival pressure, and the SLO is a "
  "percentile of sojourn time. TP lowers service time and therefore lowers queueing delay superlinearly near saturation; "
  "PP raises single-request service time but raises the number of requests that can be resident, which lowers admission "
  "rejection. The correct comparison is at equal offered load, on the p95 of the sojourn distribution, because near "
  "saturation small service-time differences produce large tail differences. Falsifiable hypothesis H31: at 80% of "
  "saturation throughput, a 20% reduction in per-token service time reduces p95 end-to-end latency by more than 20% "
  "(ESTIMATE; derivation: queueing delay grows with rho/(1-rho), so improvements compound near saturation). Controlled "
  "experiment: measure the saturation throughput of each topology first, then re-run both at 50%, 80% and 95% of their "
  "own saturation and compare p95 sojourn time. Evidence required: offered-load versus latency curves with the saturation "
  "point identified for each arm; queue-depth and admission-rejection counters from the serving engine. Rollback gate: no "
  "topology is adopted on a single-load-point comparison."),
 ("Stance 32 - Interrogate the SLO before the topology.",
  "Refuse the framing until the SLO is decomposed. 'Latency-sensitive' is not actionable: TTFT and TPOT respond to "
  "opposite pressures. A chat UI cares about TTFT and about tokens arriving faster than reading speed; an agentic "
  "tool-calling workload with short outputs is TTFT-dominated; a long-form generation workload is TPOT-dominated. TP helps "
  "both but with different efficiency; PP hurts TPOT more than TTFT once micro-batching is in play. Falsifiable "
  "hypothesis H32: for a workload whose median output length is under 32 tokens, the topology choice changes end-to-end "
  "p95 latency by less than 10% because prefill dominates, whereas for outputs above 512 tokens the same choice changes it "
  "by much more (ESTIMATE; derivation: total latency is TTFT + out_len * TPOT, so the TPOT term's weight scales with "
  "output length). Controlled experiment: replay the production output-length distribution rather than a fixed length, "
  "and additionally run two synthetic distributions at 32 and 512 tokens to bracket it. Evidence required: the production "
  "prompt and output length histograms; a latency breakdown into TTFT and TPOT*out_len. Rollback gate: reject any "
  "benchmark that uses a fixed output length as the sole basis for a production topology decision."),
 ("Stance 33 - Batching policy is a confounder that can invert the result.",
  "Insist that continuous batching interacts with parallelism strongly enough to invert naive conclusions. A larger "
  "resident batch raises arithmetic intensity and shifts decode away from the bandwidth-bound regime, which shrinks TP's "
  "relative benefit; conversely PP's bubble shrinks as more micro-batches become available, which is exactly what "
  "continuous batching supplies at high load. So a comparison run with a fixed static batch will misrepresent both. "
  "Falsifiable hypothesis H33: the TP-over-PP TPOT advantage measured at concurrency 1 shrinks by more than half at "
  "concurrency 32 under continuous batching (ESTIMATE; derivation: PP utilization M/(M+P-1) rises toward 1 as in-flight "
  "micro-batches grow). Controlled experiment: hold scheduler policy, max batch tokens, and preemption policy identical "
  "across arms; sweep concurrency; log the achieved running-batch size distribution per arm and verify the arms actually "
  "reached comparable batch sizes. Evidence required: per-step running batch size histograms; scheduler configuration "
  "diff proving only parallelism changed. Rollback gate: discard any A/B whose achieved batch-size distributions differ "
  "materially, since that is a confounded comparison rather than evidence."),
 ("Stance 34 - Quantization and precision change the crossover point.",
  "Point out that the TP-versus-PP crossover is not a fixed property of the model but a function of precision. Lowering "
  "weight precision cuts the weight-bytes term of decode latency roughly proportionally while leaving the collective term "
  "unchanged, so it pushes the crossover toward lower TP degrees: after aggressive weight quantization the same model may "
  "no longer need TP at all for latency, only for KV capacity. KV-cache quantization moves the capacity constraint "
  "similarly and independently. Falsifiable hypothesis H34: after halving weight precision, the smallest TP degree that "
  "meets the same p95 TPOT SLO drops by at least one power of two (ESTIMATE; derivation: the memory-traffic term halves "
  "while the synchronous collective term is unchanged). Controlled experiment: fix everything but precision, sweep TP for "
  "each precision, record the SLO-feasible minimum TP; separately validate output quality with a task-level evaluation so "
  "a latency win is not bought with silent accuracy loss. Evidence required: paired latency and quality results per "
  "precision; kernel-level evidence that the quantized path is actually taken and not silently upcast. Rollback gate: no "
  "precision change ships without a quality evaluation at least as strict as the latency evaluation."),
 ("Stance 35 - Multi-node fabric health is a precondition, not a detail.",
  "If the deployment is or may become multi-node, treat fabric correctness as a gate that precedes any latency "
  "comparison. RDMA over converged Ethernet depends on a lossless or near-lossless configuration; misconfigured "
  "flow control or ECN produces intermittent pause storms and retransmissions that appear as unexplained latency spikes "
  "and are frequently misattributed to the parallelism strategy. GPUDirect RDMA must be verified as actually engaged "
  "rather than assumed, since a silent fallback to a staged host-memory copy adds latency and CPU cost without any error. "
  "Falsifiable hypothesis H35: with GPUDirect RDMA disabled, inter-node collective latency for small messages increases "
  "materially and p99 TPOT for any node-spanning TP group degrades far more than p50 (ESTIMATE; derivation: the staged "
  "copy adds a host round trip on the critical path of every collective). Controlled experiment: run the same topology "
  "with the direct path enabled and disabled, capturing transport selection and per-collective timing. Evidence required: "
  "NIC and switch counters for pause frames, ECN marks, discards and retransmits; explicit confirmation of which "
  "transport and which memory path the collective library selected. Rollback gate: any node-spanning configuration whose "
  "fabric counters show nonzero discards under load is rejected before latency is even considered."),
 ("Stance 36 - Correctness and determinism side effects of changing the shard axis.",
  "Raise the axis the latency debate usually ignores: changing parallelism changes the numerics. TP changes reduction "
  "order in every all-reduce, so bitwise-identical outputs across TP degrees should not be expected; PP changes batching "
  "and scheduling, which can change results through different fused-kernel selection. Any regression suite that asserts "
  "exact token equality will fail spuriously and erode trust in the rollout. Falsifiable hypothesis H36: greedy decoding "
  "outputs differ between TP=1 and TP=4 for a nonzero fraction of prompts while task-level metrics stay within noise "
  "(ESTIMATE; derivation: floating-point reduction is non-associative and small differences can flip near-tie argmax "
  "choices). Controlled experiment: run a fixed prompt set greedily under each topology, measure exact-match divergence "
  "rate and a task metric with confidence intervals. Evidence required: divergence rate plus task metric with intervals, "
  "and the kernel/backend selection log for each arm. Rollback gate: ship only if the task metric difference is inside "
  "the pre-registered equivalence margin; do not gate on token-level identity, and fix the test suite rather than the "
  "topology if it asserts identity."),
 ("Stance 37 - Operational blast radius, scheduling and recovery.",
  "Evaluate the choice by what happens on a bad day. A TP group is a gang-scheduled unit: it must be co-scheduled, it "
  "restarts together, and its startup cost includes loading shards and re-establishing collective communicators. PP "
  "stages likewise form a chain where a single stage failure kills the replica. Higher parallelism therefore means "
  "larger, slower-to-recover units and coarser capacity granularity, which hurts rolling upgrades and node draining. "
  "Falsifiable hypothesis H37: mean time to restore full capacity after a single GPU failure grows roughly linearly with "
  "the parallel degree of the replica (ESTIMATE; derivation: the whole group must be rescheduled and reloaded, and "
  "placement constraints get harder to satisfy as the group grows). Controlled experiment: kill one GPU process in each "
  "topology under production-like load and measure time to full capacity, requests failed, and whether the scheduler "
  "could re-place the group at all. Evidence required: timeline of pod/process restarts, communicator re-initialization "
  "duration, and error budget consumed during recovery. Rollback gate: do not adopt a parallel degree whose single-fault "
  "recovery consumes more than a pre-agreed fraction of the monthly error budget."),
 ("Stance 38 - Disaggregated serving and KV transfer as the real modern alternative.",
  "Argue that the TP-versus-PP dichotomy is dated for large deployments, where the meaningful axis is prefill/decode "
  "disaggregation with a KV-cache transfer or shared cache tier between roles. That design lets prefill run at high "
  "parallelism for TTFT and decode run at the minimum parallelism that meets TPOT, and it converts a scheduling conflict "
  "between long prefills and latency-critical decodes into a separation of concerns. Its cost is a new critical path: "
  "moving the KV blocks. Falsifiable hypothesis H38: disaggregation improves p95 TPOT under mixed load because decode is "
  "no longer preempted by long prefills, while adding a KV-transfer term to TTFT that stays below a pre-declared budget "
  "such as 15% of TTFT (ESTIMATE; derivation: KV bytes for a long prompt are large but move over a high-bandwidth path, "
  "so the transfer is bandwidth-bound and predictable). Controlled experiment: compare colocated and disaggregated "
  "deployments at equal total GPUs under a mixed short/long prompt load; instrument KV transfer time separately from "
  "compute. Evidence required: KV transfer bytes and duration per request, cache-hit rate for any reuse tier, and "
  "prefill-induced decode stall counts in the colocated arm. Rollback gate: revert if KV transfer exceeds its declared "
  "TTFT budget or if cache-tier unavailability can stall decode."),
 ("Stance 39 - Pre-registration and anti-overfitting discipline for the comparison itself.",
  "Attack the methodology rather than the topology. Most TP-versus-PP conclusions in practice are artifacts: the sweep is "
  "tuned on the same trace it is evaluated on, the winning arm got more tuning attention, warmup is inconsistent so "
  "compilation or autotuning cost leaks into the first arm, and the reported statistic is a mean over an unstated number "
  "of runs. Pre-register the decision rule before running: primary metric, percentile, load points, number of repeats, "
  "equivalence margin, and the stopping rule. Falsifiable hypothesis H39: re-running the identical comparison on a "
  "held-out day's traffic reproduces the original winner and keeps the measured gap inside the pre-registered margin; if "
  "it does not, the original result was overfitting rather than a topology effect. Controlled experiment: split traffic "
  "traces into tuning and holdout sets, tune only on the former, report only on the latter, with at least five repeats "
  "per arm and identical warmup procedure. Evidence required: the pre-registration document, raw per-run results rather "
  "than summary means, and confidence intervals. Rollback gate: a result that does not replicate on holdout is not "
  "actionable and the incumbent topology stays."),
]

CRITIQUE = (
"Critique of the source item: the prompt is a valid infrastructure question and asks for assumptions, a falsifiable "
"hypothesis and a controlled experiment, but the corpus pair is degenerate - the user turn holds only a generic system "
"instruction while the task text sits in the assistant turn, and there is no actual answer content to evaluate. It is "
"therefore rewritten into a complete answer that supplies mechanism, boundary conditions, an explicit hypothesis, a "
"controlled single-variable experiment, the evidence artifacts required, and a rollback gate. All quantitative claims are "
"labelled ESTIMATE with their derivation; nothing here is MEASURED, because no run was performed for this review. This "
"output is provisional teacher-B review material, not expert gold, and it says nothing about any model's capability."
)


def main():
    with open(CORPUS) as f:
        lines = f.readlines()[START:START + N]
    assert len(lines) == N, len(lines)
    out = []
    for i, line in enumerate(lines):
        d = json.loads(line)
        m = {x["role"]: x["content"] for x in d["messages"]}
        su, sa = m["user"], m["assistant"]
        title, body = STANCES[i]
        ca = (f"Analytical stance under test: {title}\n\n{COMMON}\n{body}\n\n{CRITIQUE}")
        out.append({
            "source_id": d["id"],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": su,
            "source_assistant": sa,
            "corrected_answer": ca,
            "quality_dimensions": {
                "technical_correctness": 3,
                "instruction_coverage": 2,
                "operational_safety": 2,
            },
            "risks": [
                "Source pair is degenerate: the task statement occupies the assistant turn and no answer content is present.",
                "A bare TP-versus-PP verdict without interconnect and concurrency context is not decidable and invites overfitting to one topology.",
                "Latency comparisons are easily confounded by GPU placement, NCCL transport fallback, and head-count divisibility.",
                "Numeric claims about speedups are estimates derived from first principles, not measurements on this hardware.",
            ],
            "evidence_required": [
                "Interconnect topology dump and NCCL transport selection log for every arm of the comparison.",
                "Concurrency-resolved p50/p95/p99 TTFT and TPOT curves, not mean end-to-end latency.",
                "Profiler traces separating GEMM time from collective time per layer.",
                "KV-cache utilization and preemption/eviction counters from the serving engine.",
                "SLO-constrained throughput per GPU for each candidate parallelism configuration.",
            ],
            "confidence": 0.62,
        })
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", OUT, len(out))


main()

import json, os

SRC = "research/ai-infra-expert/corpus/train.jsonl"
OUT = "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0019.jsonl"
START, N = 180, 10

rows = [json.loads(l) for l in open(SRC)][START:START+N]

A = {}

A["corpus-00203"] = (
"Definition: tensor parallelism (TP) shards the parameters of a single layer across N devices so one logical matmul becomes N partial matmuls plus a collective. It is intra-layer parallelism; it does not partition the batch (data parallel) or the layer stack (pipeline parallel).\n"
"Concrete mechanism: in a transformer MLP, the up-projection W1 [h, 4h] is split column-wise into N shards of [h, 4h/N]. Each rank computes its own activation slice with no communication. The down-projection W2 [4h, h] is then split row-wise, each rank produces a full-shape [b, s, h] partial sum, and a single all-reduce over the TP group sums them. Attention is sharded the same way, by heads: each rank owns num_heads/N heads plus their KV, and the output projection is row-parallel with one all-reduce. So one transformer block costs 2 all-reduces in the forward pass (one for attention out-proj, one for MLP down-proj) and 2 more in the backward pass.\n"
"Why it matters: TP is the mechanism that makes a model whose weights plus activations plus KV cache exceed one device's HBM runnable at all, and it cuts per-device weight memory and per-device matmul FLOPs by roughly 1/N, which lowers decode latency for a single request in a way data parallelism cannot.\n"
"Boundary condition: TP is only economical inside one high-bandwidth domain. All-reduce volume per block is O(b * s * h) bytes and is issued twice per block per forward pass, so on 8x A30 24GB connected over PCIe Gen4 (~25-30 GB/s effective peer bandwidth, no NVLink) TP=8 is usually communication-bound and TP=2 or TP=4 plus pipeline or data parallelism wins. Crossing a node boundary with TP is almost always wrong.\n"
"Falsifiable check: if measured decode TPOT does not improve by at least ~1.5x going TP=1 -> TP=2 on a model that fits at both degrees, the workload is already communication- or memory-bandwidth-bound and TP is not buying compute scaling.\n"
"Evidence to collect: per-rank HBM usage, NCCL all-reduce bus bandwidth from nccl-tests at the exact message sizes b*s*h*dtype, nvidia-smi topo -m to confirm the interconnect, and TPOT/TTFT at TP in {1,2,4,8}.\n"
"Rollback gate: revert to the previous TP degree if p99 TTFT or TPOT regresses more than 10 percent, or if any rank OOMs during a warmed benchmark."
)

A["corpus-00204"] = (
"Definition: tensor parallelism splits individual weight tensors across N ranks so that each rank stores 1/N of a layer's parameters and executes 1/N of that layer's matmul work; correctness is restored by a collective (all-reduce or all-gather) inside the layer.\n"
"Concrete mechanism: Megatron-style column-then-row sharding. Layer 1 is column-parallel (each rank holds W1[:, k/N slice], no communication needed because the nonlinearity is elementwise on the sharded dimension). Layer 2 is row-parallel (each rank holds W2[k/N slice, :]) and emits a partial sum, which an all-reduce over the TP group turns into the correct output. Choosing column-then-row is deliberate: it collapses what would be two collectives into one per MLP.\n"
"Why it matters: it converts a hard memory ceiling into a tunable knob. A 70B model in bf16 needs ~140 GB of weights alone, which cannot fit on a 24 GB A30; TP=8 brings per-rank weights to ~17.5 GB and leaves headroom for KV cache. It also shrinks per-rank KV cache by sharding attention heads, which directly raises the number of concurrent sequences a server can hold.\n"
"Boundary condition: num_attention_heads (and for GQA, num_key_value_heads) must be divisible by the TP degree. With GQA and 8 KV heads, TP=16 either fails or forces KV replication, which silently erases the KV memory saving. Vocabulary and hidden sizes must likewise divide cleanly or the framework pads, wasting FLOPs.\n"
"Falsifiable check: per-rank KV cache bytes should fall as 1/TP up to TP = num_key_value_heads and then stop falling. If reported KV blocks do not scale that way, KV is being replicated.\n"
"Evidence to collect: model config num_key_value_heads, per-rank allocator snapshot (torch.cuda.memory_summary or the server's KV block count), and an ablation of max concurrent sequences at TP in {1,2,4,8}.\n"
"Rollback gate: revert if the divisibility change forces padding that costs more than 5 percent of throughput, or if per-rank memory does not drop as predicted, which indicates the shard is not actually taking effect."
)

A["corpus-00205"] = (
"Definition: tensor parallelism is intra-operator model parallelism. A single GEMM is decomposed across N devices, each device holds a slice of the weight and computes a partial result, and an in-layer collective reassembles the mathematically exact output. It is not an approximation: the result is bitwise-comparable to single-device execution up to floating-point reduction-order differences.\n"
"Concrete mechanism: for attention with H heads and TP degree N, ranks own disjoint sets of H/N heads including their Q/K/V projections and their KV cache blocks. Each rank runs full scaled-dot-product attention on its own heads, producing [b, s, H/N * d_head]. The output projection is row-parallel, so each rank's partial [b, s, h] is all-reduced across the TP group. During decode this all-reduce is small in bytes (s=1) but latency-dominated, so it is bounded by interconnect round-trip latency rather than bandwidth.\n"
"Why it matters: for latency-sensitive single-stream decode, TP is the only parallelism that reduces time per output token, because it divides the per-token GEMM work. Data parallelism increases throughput but leaves per-token latency unchanged; pipeline parallelism adds bubbles and can make single-stream latency worse.\n"
"Boundary condition: the crossover is set by the ratio of per-token compute to collective latency. In decode, per-token GEMM time falls as 1/N while all-reduce latency is roughly flat or rising with N, so beyond some N the total gets worse. On PCIe-attached A30s that crossover is often at TP=2 to TP=4; on NVLink/NVSwitch systems it is typically TP=8.\n"
"Falsifiable hypothesis: TPOT(N) = C/N + L(N). Measure at N in {1,2,4,8}; if the measured curve is flat or U-shaped rather than decreasing, the deployment is collective-latency-bound and further TP is a regression.\n"
"Evidence to collect: nsys or torch profiler traces isolating ncclAllReduce duration per decode step, nccl-tests latency at 1 KB-1 MB message sizes, and nvidia-smi topo -m.\n"
"Rollback gate: hold the smallest TP degree that fits in memory; roll back any increase in TP that does not reduce p50 TPOT by at least 15 percent."
)

A["corpus-00206"] = (
"Contrast, single-device (naive) versus tensor-parallel execution.\n"
"Naive: one device holds the full weight matrix W [h, 4h] and computes Y = XW in one GEMM. Zero communication, perfectly simple, but per-device memory must hold all weights plus optimizer state plus activations plus KV cache, and the per-token FLOPs are all served by one SM array. For a 70B bf16 model that is ~140 GB of weights, which simply does not fit on a 24 GB A30 - the naive path fails outright, it is not merely slow.\n"
"Tensor-parallel: W is column-sharded into N pieces of [h, 4h/N], each rank computes a partial activation, the paired down-projection is row-sharded, and one all-reduce per MLP restores the exact output. Per-rank weight memory and per-rank GEMM FLOPs both drop ~N-fold; the added cost is 2 all-reduces per transformer block per forward pass.\n"
"Concrete mechanism separating them: the naive path has arithmetic intensity set only by the GEMM shape, while the TP path adds a fixed communication term of roughly 2 * b * s * h * sizeof(dtype) * 2(N-1)/N bytes moved per block per forward pass. That term is what decides whether TP is a win.\n"
"Boundary condition: TP beats naive only when the model does not fit, or when compute time saved (T_gemm * (1 - 1/N)) exceeds the added all-reduce time. If the model fits on one device and the interconnect is PCIe rather than NVLink, the naive single-device path frequently wins on both latency and cost.\n"
"Falsifiable test: run the same model at TP=1 and TP=2 on a size that fits both; if TP=2 does not improve TPOT, the communication term dominates and the naive baseline should be kept.\n"
"Evidence to collect: end-to-end TPOT/TTFT at TP=1 vs TP=N, profiler breakdown of GEMM time versus NCCL time per layer, per-rank peak memory.\n"
"Rollback gate: keep the naive single-device configuration as the default and only promote TP if it wins on the primary metric by more than the run-to-run noise band (measure that band with 5 repeats before deciding)."
)

A["corpus-00207"] = (
"Contrast at the level of what each design pays for.\n"
"Naive (no TP): correctness needs no synchronization inside a layer, so there is no collective, no NCCL communicator, no risk of a hung all-reduce, and no requirement that head counts divide evenly. The costs are a hard memory ceiling and a hard single-device compute ceiling. Any model or context length that overflows HBM triggers OOM or forces offloading, and CPU offload typically costs an order of magnitude in decode latency because weights must cross PCIe every token.\n"
"With TP: memory and compute ceilings both scale with N, at the price of a synchronous collective on the critical path of every block. This makes the whole TP group a single fate-sharing unit: one slow or faulty GPU stalls all ranks at the next all-reduce, and one crashed rank hangs the rest until the NCCL timeout fires.\n"
"Concrete mechanism: the row-parallel output projection emits per-rank partial sums that are only correct after ncclAllReduce over the TP communicator. That call is blocking, so per-step time equals the slowest rank's time plus collective time - TP converts independent failure into correlated failure.\n"
"Boundary condition: TP is appropriate only within a single node and single high-bandwidth fabric. Once the TP group spans nodes, every block's all-reduce traverses the network (RoCE/IB), and unless RDMA with GPUDirect RDMA is correctly configured the copy path adds host bounce buffers and latency that make the naive fallback or a smaller TP degree preferable.\n"
"Falsifiable check: if per-step time variance grows superlinearly with TP degree, one rank is straggling (thermal cap, ECC retirement, or a noisy neighbour); confirm with per-rank timing rather than assuming a communication problem.\n"
"Evidence to collect: per-rank step timing histogram, NCCL_DEBUG=INFO topology dump, nvidia-smi -q clocks and throttle reasons, ib/roce counters if multi-node.\n"
"Rollback gate: if the NCCL watchdog reports any collective timeout in a soak run, or per-rank step-time spread exceeds 10 percent, drain the deployment back to the last known-good TP degree before serving traffic."
)

A["corpus-00208"] = (
"Contrast by scaling behaviour.\n"
"Naive implementation: throughput and memory both scale as O(1) in device count because there is only one device. Adding GPUs helps only by running independent replicas (data parallel), which multiplies aggregate throughput but leaves per-request latency and the maximum model size completely unchanged.\n"
"Tensor parallel: per-rank weight bytes and per-rank GEMM FLOPs scale as O(1/N), while communication scales as O(b*s*h) per collective and is essentially independent of N in volume for ring all-reduce (2(N-1)/N of the buffer, which asymptotes to 2x). So the compute term shrinks and the communication term does not, which is exactly why TP has a sweet spot rather than scaling forever.\n"
"Concrete mechanism: ring all-reduce moves 2(N-1)/N * S bytes per rank for a buffer of S bytes, in 2(N-1) steps. At N=2 that is 1.0*S in 2 steps; at N=8 it is 1.75*S in 14 steps. The step count is what makes small decode-time buffers latency-bound rather than bandwidth-bound.\n"
"Boundary condition: prefill and decode sit on opposite sides of this. Prefill has large s so the collective is bandwidth-bound and amortizes well against large GEMMs; decode has s=1 so the same collective is latency-bound and amortizes poorly. A TP degree tuned on prefill throughput can therefore be the wrong choice for decode TPOT, which argues for measuring both, or for disaggregating prefill and decode onto separately tuned pools.\n"
"Falsifiable hypothesis: speedup(N) for prefill should track close to N until PCIe saturates; speedup(N) for decode should saturate much earlier. If both curves look identical, the benchmark is not actually separating prefill from decode.\n"
"Evidence to collect: separate TTFT (prefill-dominated) and TPOT (decode-dominated) measurements at each TP degree, achieved all-reduce bus bandwidth per phase, and GEMM-vs-NCCL time split from a profiler.\n"
"Rollback gate: do not adopt a TP change on the strength of an aggregate tokens/s number alone; require both TTFT and TPOT to be non-regressing, otherwise revert."
)

A["corpus-00209"] = (
"Contrast by operational surface, which is where the naive path is genuinely better.\n"
"Naive: one process, one CUDA context, one failure domain. Restart time is one model load. There is no communicator to bootstrap, no rendezvous, no environment variables such as NCCL_SOCKET_IFNAME or NCCL_IB_HCA to get wrong, and no class of bug where the job appears alive but every rank is blocked in a collective.\n"
"Tensor parallel: N processes (or N ranks in one process group) that must all agree on the world size, rank ordering, and device mapping before the first token. Startup now includes NCCL bootstrap over a chosen transport; a misdetected NIC or a blocked port produces a hang, not a clean error. Checkpoints become sharded, so save/load must be resharding-aware if the serving TP degree differs from the training TP degree.\n"
"Concrete mechanism: each rank's partial output from the row-parallel projection is only valid after the group's all-reduce, so ranks are lockstep. A single rank that never reaches the collective blocks the entire group until NCCL's watchdog timeout (default order of 10 minutes) expires, at which point the whole replica dies rather than degrading.\n"
"Boundary condition: this only pays off when the model cannot be served naively, or when latency targets require dividing per-token compute. If a single A30 can hold the model with adequate KV headroom at the target concurrency, adding TP buys complexity and correlated failure for little or negative gain.\n"
"Falsifiable check: measure mean time to recovery for both topologies by killing one process; if TP MTTR is more than 3x the naive MTTR and TP gives less than 1.5x latency benefit, the naive design is the better operational choice.\n"
"Evidence to collect: cold-start and warm-start times, kill-one-rank recovery time, NCCL_DEBUG=INFO bootstrap logs, checkpoint shard layout and whether a resharding path exists and has been exercised.\n"
"Rollback gate: require a rehearsed rollback to the naive or lower-TP configuration, verified in staging with a real rank kill, before any TP degree change reaches production traffic."
)

A["corpus-00210"] = (
"Contrast on memory accounting, which is usually the deciding factor.\n"
"Naive: one device must hold weights + activations + KV cache + fragmentation headroom. For inference in bf16, weights are ~2 bytes per parameter and KV cache is 2 * num_layers * num_kv_heads * head_dim * seq_len * batch * 2 bytes. On a 24 GB A30 the weights alone bound which models are servable, and the KV term bounds concurrency.\n"
"Tensor parallel: weights divide by N exactly; KV cache divides by N as long as N <= num_key_value_heads; activations divide for the sharded dimensions but the all-reduce buffers and any replicated tensors (embeddings if not sharded, layernorm parameters, residual stream) do not. So per-rank memory is (weights/N) + (KV/N) + activations/N + a replicated remainder that does not shrink, which is why observed savings are always somewhat less than 1/N.\n"
"Concrete mechanism: the residual stream tensor [b, s, h] is full-shape on every rank between the two collectives, because column-parallel output is sharded but row-parallel output is all-reduced back to full shape. That full-shape tensor is the floor on activation memory regardless of TP degree.\n"
"Boundary condition: TP does not fix a context-length problem past the point where the full-shape residual and attention workspace dominate. At very long context the KV term still divides, but paged-attention block overhead and workspace do not, so the marginal benefit of raising TP flattens.\n"
"Falsifiable prediction: plot measured per-rank peak memory against 1/N; the intercept of the linear fit is the non-shardable remainder. If that intercept is a large fraction of device memory, further TP will not help and the fix is quantization, KV compression, or a different attention kernel.\n"
"Evidence to collect: torch.cuda.max_memory_allocated per rank at each TP degree, the server's reported KV block count, and model config num_key_value_heads and head_dim.\n"
"Rollback gate: revert any TP increase whose measured per-rank memory reduction is less than 60 percent of the predicted 1/N saving, since that indicates replication or fragmentation rather than real sharding."
)

A["corpus-00211"] = (
"Two failure modes / trade-offs of tensor parallelism.\n"
"1) Communication becomes the bottleneck. Mechanism: every transformer block issues an all-reduce after the attention output projection and another after the MLP down projection, so a 40-layer model performs 80 blocking collectives per forward pass. On a PCIe-only 8x A30 node with roughly 25-30 GB/s effective peer bandwidth and no NVLink, the time spent in ncclAllReduce can exceed the GEMM time saved by sharding, so raising TP from 4 to 8 makes latency worse rather than better. Boundary condition: this flips sign on NVLink/NVSwitch fabrics where intra-node bandwidth is an order of magnitude higher. Falsifiable test: measure GEMM time and NCCL time separately at TP in {2,4,8}; the hypothesis 'more TP is faster' is rejected the moment the NCCL fraction exceeds roughly 30 percent of step time.\n"
"2) Correlated failure and lockstep stalls. Mechanism: TP ranks are synchronous, so per-step time is set by the slowest rank plus collective time. A single GPU that thermal-throttles, hits ECC error retirement, or shares a PCIe switch with a busy NIC will slow every other rank; a rank that dies leaves the rest blocked in the collective until the NCCL watchdog timeout fires, turning a one-GPU fault into a whole-replica outage. Boundary condition: this risk grows with TP degree and becomes severe if the TP group ever spans nodes, because the network adds another correlated failure source. Falsifiable test: kill one rank in staging; if the replica does not fail fast and get replaced within the SLO, the topology is not operationally safe.\n"
"Evidence to collect: per-rank step-time distribution, nvidia-smi throttle reasons and ECC counters, NCCL timeout settings, profiler split of GEMM versus collective time, and a rehearsed single-rank-kill recovery measurement.\n"
"Rollback gate: revert to the previous TP degree if NCCL time exceeds 30 percent of step time, if per-rank step-time spread exceeds 10 percent, or if any collective timeout appears in a soak run."
)

A["corpus-00212"] = (
"Two trade-offs of tensor parallelism, stated as things that can be measured and can go wrong.\n"
"1) Divisibility and silent replication. Mechanism: TP shards attention by heads and the MLP by hidden dimension, so the TP degree must divide num_attention_heads and, under grouped-query attention, num_key_value_heads. With 8 KV heads, TP=8 shards KV one head per rank, but TP=16 cannot - the framework either errors out or replicates KV across pairs of ranks. Replication means per-rank KV memory stops falling while you believe it is still halving, so a capacity plan built on a 1/N assumption overcommits and the server OOMs at the concurrency you sized for. Boundary condition: the KV saving is bounded by min(TP, num_key_value_heads); beyond that only weights keep sharding. Falsifiable check: per-rank KV block count should scale as 1/TP up to num_key_value_heads and then flatten; if it flattens earlier, sharding is not doing what the config claims.\n"
"2) Checkpoint and topology coupling. Mechanism: TP produces sharded weight files whose layout is tied to the TP degree used when saving. Loading a TP=8 checkpoint into a TP=4 server requires an explicit resharding step; skipping it yields either a load error or, worse, a silently misaligned load if shapes coincidentally match after padding, which degrades quality without crashing. This couples an operational choice (how many GPUs a replica gets) to an artifact format, so capacity changes are no longer free. Boundary condition: frameworks that save a consolidated or TP-agnostic checkpoint avoid this at the cost of save time and peak host memory. Falsifiable check: after any TP-degree change, compare greedy generations on a fixed 50-prompt probe set against the previous topology; identical prompts must give matching outputs within tokenizer-level noise, and a perplexity delta over ~1 percent on a held-out set indicates a bad load rather than a benign difference.\n"
"Evidence to collect: model config head counts, per-rank KV block counts across TP degrees, checkpoint shard manifest, and a fixed-seed generation-parity probe run before and after any topology change.\n"
"Rollback gate: block the topology change and restore the prior checkpoint/TP pair if generation parity fails, if perplexity moves more than 1 percent, or if per-rank KV memory does not match the predicted min(TP, num_key_value_heads) scaling."
)

recs = []
for r in rows:
    sid = r["id"]
    msgs = {m["role"]: m["content"] for m in r["messages"]}
    recs.append({
        "source_id": sid,
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": msgs["user"],
        "source_assistant": msgs["assistant"],
        "corrected_answer": A[sid],
        "quality_dimensions": {
            "technical_correctness": 4,
            "instruction_coverage": 2,
            "operational_safety": 3,
        },
        "risks": [
            "source answer is a single generic sentence: it names no concrete sharding mechanism (column/row parallel, head sharding) so it cannot teach the mechanism the prompt explicitly asks for",
            "no boundary condition is given despite the prompt requiring one; a model trained on this learns to assert 'it depends on memory, topology, batch' without stating where the crossover is",
            "identical assistant text is reused across all prompt variants (define / contrast / failure modes), so it does not answer the contrast and failure-mode framings at all",
            "operationally unsafe as guidance: no divisibility constraint on num_key_value_heads, no correlated-failure warning, and no rollback gate for changing TP degree",
        ],
        "evidence_required": [
            "nvidia-smi topo -m to establish whether the TP group is NVLink- or PCIe-connected",
            "nccl-tests all_reduce_perf bus bandwidth and latency at the message sizes implied by b*s*hidden*dtype",
            "profiler split of GEMM time versus ncclAllReduce time per decode step at TP in {1,2,4,8}",
            "per-rank peak memory and KV block counts at each TP degree, checked against the min(TP, num_key_value_heads) prediction",
            "separate TTFT and TPOT measurements so prefill and decode behaviour are not conflated",
        ],
        "confidence": 0.74,
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for rec in recs:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("wrote", OUT, len(recs), recs[0]["source_id"], recs[-1]["source_id"])
